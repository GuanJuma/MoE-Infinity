# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Single-expert FFN runners: ``out = down(act(gate(x)) * up(x))``.

The expert is FP8 (e4m3) everywhere by default: stored, moved and consumed
as the checkpoint's ~18 MiB of FP8 bytes.  BF16/INT8 kernels exist only as
explicitly requested non-FP8 comparisons.

GPU kernels (``GPU_KERNELS``), all run on one expert (E=1, top-1, weight 1):

``sglang_triton`` (default, FP8 W8A8)
    SGLang's Triton ``fused_experts`` -- the path SGLang itself uses for an
    FP8 MoE on sm_120 (no DeepGEMM/CUTLASS-FP8 MoE there).  Per-tensor FP8 is
    prepared exactly like SGLang's ``Fp8MoEMethod``: gate/up requantized to
    one scale (their max), static activation scales = checkpoint
    ``input_scale``.  Block-FP8 and per-channel FP8 use SGLang's block /
    per-channel paths with dynamic activation quantization.
``torch_scaled_mm`` (default, FP8 W8A8)
    Two cuBLASLt FP8 GEMMs (``torch._scaled_mm``, per-tensor scales) plus
    SiLU*up and the FP8 activation casts in PyTorch.  Per-tensor FP8 only.
``torch_bf16`` (opt-in in FP8 mode; cuBLAS reference in BF16 mode)
    Weights dequantized once to BF16, two cuBLAS BF16 GEMMs; 2x the bytes.
``batchgen_triton`` (BF16 mode only, primary there)
    PR #1's vendored BatchGen ``fused_moe_bf16`` Triton kernels, JIT-compiled
    for the running GPU; sm_120 is fine (the AOT cubins are sm_80/90 only).

CPU kernels (``CPU_KERNELS``).  x86 CPUs have no FP8 matmul units, so FP8
weights stay FP8 in memory and are widened to BF16 inside the kernel:

``sglang_fp8_w8a16`` (default)
    Vendored SGLang ``fused_experts_cpu``: e4m3 weights (per-tensor scales
    broadcast exactly to 128x128 block scales), BF16 activations.
``sglang_fp8_w8a8_emu`` (default)
    Numerical W8A8 like the GPU paths: activations rounded to e4m3 with the
    same static input scales, the same FP8 weight codes, SGLang's CPU FP8
    GEMM (``fp8_scaled_mm_cpu``).  Not faster than W8A16; the activation
    quantize steps are timed separately (``act_quant_ms``).
``sglang_bf16`` (BF16 mode default; opt-in non-FP8 comparison in FP8 mode)
    Same kernel on BF16 weights, AMX-BF16 / AVX512-BF16, no dequant.
``sglang_int8_w8a8`` / ``torch_bf16`` (opt-in, non-FP8)
    INT8 W8A8 requantized weights; PyTorch baseline.

Every runner is used through ``bind(x)`` which returns ``(fn, reset)``:
``fn()`` computes the expert on the bound input and returns the output
tensor; ``reset`` (or None) restores the input after an in-place run and is
always called outside the timed region.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from checkpoint_expert import CheckpointError, ExpertWeights

FP8_MAX = 448.0
GPU_KERNELS = (
    "sglang_triton",
    "torch_scaled_mm",
    "torch_bf16",
    "batchgen_triton",
)
CPU_KERNELS = (
    "sglang_fp8_w8a16",
    "sglang_fp8_w8a8_emu",
    "sglang_bf16",
    "sglang_int8_w8a8",
    "torch_bf16",
)
# The expert stays FP8 (stored, moved, consumed as e4m3) only with these.
FP8_GPU_KERNELS = ("sglang_triton", "torch_scaled_mm")
FP8_CPU_KERNELS = ("sglang_fp8_w8a16", "sglang_fp8_w8a8_emu")
W8A8_KERNELS = ("sglang_triton", "torch_scaled_mm", "sglang_fp8_w8a8_emu")
# --precision bf16: the expert is BF16 everywhere (dequantized from FP8).
BF16_GPU_KERNELS = ("batchgen_triton", "torch_bf16", "sglang_triton")
BF16_CPU_KERNELS = ("sglang_bf16",)
PRECISIONS = ("fp8", "bf16")


def default_kernels(precision: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    if precision == "fp8":
        return FP8_GPU_KERNELS, FP8_CPU_KERNELS
    if precision == "bf16":
        return BF16_GPU_KERNELS, BF16_CPU_KERNELS
    raise ValueError(
        f"unknown precision {precision!r}; choose from {PRECISIONS}"
    )


class KernelUnavailable(RuntimeError):
    pass


def _act_mul(h: torch.Tensor, activation: str) -> torch.Tensor:
    d = h.shape[-1] // 2
    g, u = h[..., :d], h[..., d:]
    return (F.silu(g) if activation == "silu" else F.gelu(g)) * u


def reference_expert(x: torch.Tensor, ew: ExpertWeights) -> torch.Tensor:
    """FP32 reference with dequantized weights and unquantized activations."""
    dev = x.device
    g = ew.gate.dequant().to(dev)
    u = ew.up.dequant().to(dev)
    d = ew.down.dequant().to(dev)
    xf = x.float()
    h = torch.cat([xf @ g.t(), xf @ u.t()], dim=-1)
    return _act_mul(h, ew.activation) @ d.t()


def rel_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    ref = ref.float()
    return float((out.float() - ref).norm() / ref.norm().clamp(min=1e-12))


def _requant_fp8(w_fp32: torch.Tensor, scale: float) -> torch.Tensor:
    return (w_fp32 / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)


def merged_per_tensor_w13(ew: ExpertWeights):
    """SGLang Fp8MoEMethod: one w13 scale per expert = max(gate, up)."""
    sg = float(ew.gate.scale.reshape(()))
    su = float(ew.up.scale.reshape(()))
    s = max(sg, su)
    parts = []
    for p, sp in ((ew.gate, sg), (ew.up, su)):
        if sp == s:
            parts.append(p.weight)
        else:
            parts.append(_requant_fp8(p.weight.float() * sp, s))
    return torch.cat(parts, 0).contiguous(), s


# --------------------------------------------------------------------------
# GPU runners
# --------------------------------------------------------------------------


@dataclass
class GpuRunner:
    """A GPU-ready expert: ``weights`` are the tensors scenario 3 copies."""

    kernel: str
    weights: Dict[str, torch.Tensor]
    meta: dict = field(default_factory=dict)
    prep_ms: float = 0.0
    device: torch.device = torch.device("cpu")
    _bind: Callable = None

    @property
    def weight_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.weights.values())

    def to(self, device) -> "GpuRunner":
        self.weights = {
            k: v.to(device).contiguous() for k, v in self.weights.items()
        }
        self.device = torch.device(device)
        return self

    def host_copy(self, pinned: bool) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in self.weights.items():
            h = torch.empty(v.shape, dtype=v.dtype, pin_memory=pinned)
            h.copy_(v.cpu())
            out[k] = h
        return out

    def bind(self, x: torch.Tensor):
        return self._bind(self, x)


def _gpu_prep(
    ew: ExpertWeights, kernel: str
) -> Tuple[Dict[str, torch.Tensor], dict]:
    """Host-side, one-time conversion to the layout the kernel consumes."""
    a1, a2 = ew.static_input_scales()
    if kernel in ("torch_bf16", "batchgen_triton"):
        w13 = torch.cat(
            [ew.gate.dequant(torch.bfloat16), ew.up.dequant(torch.bfloat16)], 0
        )
        fmt = (
            f"bf16 [{ew.source}]"
            if not ew.gate.is_fp8
            else "bf16, dequantized once from the FP8 checkpoint (non-FP8 comparison)"
        )
        return {
            "w13": w13.contiguous(),
            "w2": ew.down.dequant(torch.bfloat16).contiguous(),
        }, {"format": fmt, "numerics": "BF16"}
    fp8 = ew.gate.is_fp8
    scheme = ew.gate.scheme
    if kernel == "torch_scaled_mm":
        if not fp8 or scheme != "per_tensor":
            raise KernelUnavailable(
                f"torch_scaled_mm needs per-tensor FP8 weights (checkpoint is {ew.quant})"
            )
        w13, s13 = merged_per_tensor_w13(ew)
        t = {
            "w13": w13,
            "w2": ew.down.weight.contiguous(),
            "w13_scale": torch.tensor([s13], dtype=torch.float32),
            "w2_scale": ew.down.scale.reshape(1).to(torch.float32),
        }
        if a1 is not None and a2 is not None:
            t["a1_scale"] = torch.tensor([a1], dtype=torch.float32)
            t["a2_scale"] = torch.tensor([a2], dtype=torch.float32)
        return t, {
            "format": "fp8 e4m3 per-tensor, w13 requantized to max(gate,up) scale",
            "act_quant": "static (checkpoint input_scale)"
            if "a1_scale" in t
            else "dynamic per-tensor",
        }
    if kernel == "sglang_triton":
        if not fp8:
            return {
                "w13": torch.cat([ew.gate.weight, ew.up.weight], 0)[None]
                .to(torch.bfloat16)
                .contiguous(),
                "w2": ew.down.weight[None].to(torch.bfloat16).contiguous(),
            }, {
                "format": f"bf16 [{ew.source}]",
                "mode": "bf16",
                "numerics": "BF16",
            }
        if scheme == "per_tensor":
            w13, s13 = merged_per_tensor_w13(ew)
            t = {
                "w13": w13[None].contiguous(),
                "w2": ew.down.weight[None].contiguous(),
                "w13_scale": torch.tensor([s13], dtype=torch.float32),
                "w2_scale": ew.down.scale.reshape(1).to(torch.float32),
            }
            meta = {
                "format": "fp8 e4m3 per-tensor (SGLang Fp8MoEMethod layout)",
                "mode": "per_tensor",
            }
            if a1 is not None and a2 is not None:
                # 0-dim, as Fp8MoEMethod stores w13/w2_input_scale.max().
                t["a1_scale"] = torch.tensor(a1, dtype=torch.float32)
                t["a2_scale"] = torch.tensor(a2, dtype=torch.float32)
                meta["act_quant"] = "static (checkpoint input_scale)"
            else:
                meta["act_quant"] = "dynamic per-tensor"
            return t, meta
        if scheme == "per_channel":
            return {
                "w13": torch.cat([ew.gate.weight, ew.up.weight], 0)[
                    None
                ].contiguous(),
                "w2": ew.down.weight[None].contiguous(),
                "w13_scale": torch.cat([ew.gate.scale, ew.up.scale])[
                    None, :, None
                ].contiguous(),
                "w2_scale": ew.down.scale[None, :, None].contiguous(),
            }, {
                "format": "fp8 per-channel",
                "mode": "per_channel",
                "act_quant": "dynamic per-token",
            }
        if scheme == "block":
            blk = ew.gate.block
            return {
                "w13": torch.cat([ew.gate.weight, ew.up.weight], 0)[
                    None
                ].contiguous(),
                "w2": ew.down.weight[None].contiguous(),
                "w13_scale": torch.cat([ew.gate.scale, ew.up.scale], 0)[
                    None
                ].contiguous(),
                "w2_scale": ew.down.scale[None].contiguous(),
            }, {
                "format": f"fp8 block {list(blk)}",
                "mode": "block",
                "block_shape": list(blk),
                "act_quant": f"dynamic per-token-group (1x{blk[1]})",
            }
    raise KernelUnavailable(f"unknown GPU kernel {kernel!r}")


def _bind_torch_bf16(r: GpuRunner, x):
    w13, w2, act = r.weights["w13"], r.weights["w2"], r.meta["activation"]

    def fn():
        return _act_mul(x @ w13.t(), act) @ w2.t()

    return fn, None


def _fp8_cast(t: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (t.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)


def _bind_torch_scaled_mm(r: GpuRunner, x):
    w = r.weights
    w13, w2, s13, s2 = w["w13"], w["w2"], w["w13_scale"], w["w2_scale"]
    a1, a2 = w.get("a1_scale"), w.get("a2_scale")
    act = r.meta["activation"]
    out_dtype = x.dtype

    def dyn(t):
        return (t.abs().amax().float() / FP8_MAX).clamp(min=1e-12).reshape(())

    if x.device.type == "cuda":
        mm = torch._scaled_mm
    else:
        # CPU debug runs only: oneDNN rejects most FP8 matmul shapes, so
        # emulate _scaled_mm's math to keep the quantization logic testable.
        def mm(a, b, scale_a, scale_b, out_dtype):
            return ((a.float() * scale_a) @ (b.float() * scale_b)).to(out_dtype)

    def fn():
        sa = a1.reshape(()) if a1 is not None else dyn(x)
        h = mm(
            _fp8_cast(x, sa),
            w13.t(),
            scale_a=sa,
            scale_b=s13.reshape(()),
            out_dtype=out_dtype,
        )
        a = _act_mul(h, act)
        sb = a2.reshape(()) if a2 is not None else dyn(a)
        return mm(
            _fp8_cast(a, sb),
            w2.t(),
            scale_a=sb,
            scale_b=s2.reshape(()),
            out_dtype=out_dtype,
        )

    return fn, None


_SGLANG = {}


def _load_sglang():
    """Import SGLang's Triton fused_experts and publish dummy server args."""
    if "fused_experts" in _SGLANG:
        return _SGLANG
    try:
        from sglang.srt.layers.moe.fused_moe_triton import fused_experts
    except Exception as e:  # noqa: BLE001
        raise KernelUnavailable(f"cannot import SGLang fused_experts: {e!r}")
    _SGLANG["fused_experts"] = fused_experts
    params = inspect.signature(fused_experts).parameters
    _SGLANG["new_api"] = "topk_output" in params
    if _SGLANG["new_api"]:
        try:
            from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
            from sglang.srt.layers.moe.topk import StandardTopKOutput
        except Exception as e:  # noqa: BLE001
            raise KernelUnavailable(
                f"SGLang MoE runner types not importable: {e!r}"
            )
        _SGLANG["MoeRunnerConfig"] = MoeRunnerConfig
        _SGLANG["StandardTopKOutput"] = StandardTopKOutput
    # fused_experts reads process-wide config (get_exec()); outside a server
    # it must be published once. model_path="dummy" skips model resolution.
    try:
        from sglang.srt.runtime_context import get_exec

        get_exec()
        _SGLANG["published"] = "already"
    except ImportError:
        _SGLANG["published"] = "n/a"
    except Exception:  # noqa: BLE001
        try:
            from sglang.srt.server_args import (
                ServerArgs,
                set_global_server_args_for_scheduler,
            )

            set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
            _SGLANG["published"] = "dummy ServerArgs"
        except Exception as e:  # noqa: BLE001
            raise KernelUnavailable(
                f"SGLang needs published server args and publishing failed: {e!r}"
            )
    try:
        import sglang

        _SGLANG["version"] = getattr(sglang, "__version__", "?")
    except Exception:  # noqa: BLE001
        _SGLANG["version"] = "?"
    return _SGLANG


def _bind_sglang_triton(r: GpuRunner, x):
    sg = _load_sglang()
    w = r.weights
    meta = r.meta
    M, K = x.shape
    dev = x.device
    ids = torch.zeros(M, 1, dtype=torch.int32, device=dev)
    tw = torch.ones(M, 1, dtype=torch.float32, device=dev)
    logits = torch.zeros(M, 1, dtype=torch.float32, device=dev)
    # In-place keeps fused_experts off the TP-group / symmetric-memory
    # allocation path that needs an initialized distributed environment.
    work = x.clone()
    mode = meta["mode"]
    kw = {}
    if mode != "bf16":
        kw = dict(
            use_fp8_w8a8=True,
            w1_scale=w["w13_scale"],
            w2_scale=w["w2_scale"],
            a1_scale=w.get("a1_scale"),
            a2_scale=w.get("a2_scale"),
            per_channel_quant=(mode == "per_channel"),
            block_shape=meta.get("block_shape"),
        )
    fe = sg["fused_experts"]
    if sg["new_api"]:
        cfg = sg["MoeRunnerConfig"](
            num_experts=1,
            num_local_experts=1,
            hidden_size=K,
            intermediate_size_per_partition=w["w2"].shape[-1],
            top_k=1,
            params_dtype=x.dtype,
            activation=meta["activation"],
            inplace=True,
        )
        topk = sg["StandardTopKOutput"](tw, ids, logits)

        def fn():
            fe(work, w["w13"], w["w2"], topk, cfg, **kw)
            return work

    else:

        def fn():
            fe(
                work,
                w["w13"],
                w["w2"],
                tw,
                ids,
                inplace=True,
                activation=meta["activation"],
                **kw,
            )
            return work

    def reset():
        work.copy_(x)

    return fn, reset


_BATCHGEN = {}


def _load_batchgen():
    if "expert_ffn" not in _BATCHGEN:
        try:
            from moe_infinity.kernel.batchgen.expert_ffn import expert_ffn
        except Exception as e:  # noqa: BLE001
            raise KernelUnavailable(
                f"BatchGen Triton kernels (PR #1) not importable: {e!r}"
            )
        _BATCHGEN["expert_ffn"] = expert_ffn
    return _BATCHGEN["expert_ffn"]


def _bind_batchgen(r: GpuRunner, x):
    """PR #1's vendored BatchGen ``fused_moe_bf16`` Triton kernels, JIT
    compiled by Triton for the running GPU (so sm_120 works; the AOT cubins
    PR #1 embeds are sm_80/sm_90).  Single-expert form: packed w13 GEMM,
    ``silu_and_mul``, down GEMM."""
    expert_ffn = _load_batchgen()
    w = r.weights
    if r.meta["activation"] != "silu":
        raise KernelUnavailable("BatchGen expert_ffn implements silu only")

    def fn():
        w13 = w["w13"]
        n = w13.shape[0] // 2
        return expert_ffn(x, w13[:n], w13[n:], w["w2"])

    return fn, None


_BINDERS = {
    "torch_bf16": _bind_torch_bf16,
    "torch_scaled_mm": _bind_torch_scaled_mm,
    "sglang_triton": _bind_sglang_triton,
    "batchgen_triton": _bind_batchgen,
}


def build_gpu_runner(ew: ExpertWeights, kernel: str, device) -> GpuRunner:
    if kernel not in _BINDERS:
        raise KernelUnavailable(
            f"unknown GPU kernel {kernel!r}; choose from {GPU_KERNELS}"
        )
    if kernel == "sglang_triton":
        _load_sglang()
    if kernel == "batchgen_triton":
        _load_batchgen()
        if ew.gate.is_fp8:
            raise KernelUnavailable(
                "batchgen_triton is BF16-only; use --precision bf16"
            )
    t0 = time.perf_counter()
    weights, meta = _gpu_prep(ew, kernel)
    prep_ms = (time.perf_counter() - t0) * 1e3
    meta["activation"] = ew.activation
    r = GpuRunner(kernel, weights, meta, prep_ms, _bind=_BINDERS[kernel])
    return r.to(device)


def resolve_gpu_kernels(names: List[str], ew: ExpertWeights) -> List[str]:
    if names != ["auto"]:
        return names
    out = []
    try:
        _load_sglang()
        out.append("sglang_triton")
    except KernelUnavailable:
        pass
    if ew.gate.is_fp8 and ew.gate.scheme == "per_tensor":
        out.append("torch_scaled_mm")
    if not out:
        raise KernelUnavailable(
            "no FP8 GPU kernel applies (SGLang not importable and the checkpoint "
            f"is {ew.quant}); pass --gpu-kernels torch_bf16 for a non-FP8 comparison"
        )
    return out


# --------------------------------------------------------------------------
# CPU runners
# --------------------------------------------------------------------------


@dataclass
class CpuRunner:
    """A CPU expert: ``tensors`` hold the packed weights (and scales).

    ``tensors`` can be swapped for views of the same bytes elsewhere (e.g.
    the MoE-Infinity pinned host pool) before ``bind``; nothing is copied.
    ``last`` holds the per-call breakdown of kernels that time sub-steps.
    """

    kernel: str
    pack_ms: float
    meta: dict
    tensors: Dict[str, torch.Tensor]
    _make: Callable = None
    last: dict = field(default_factory=dict)

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors.values())

    @property
    def fp8(self) -> bool:
        return self.kernel in FP8_CPU_KERNELS

    def bind(self, x_host: torch.Tensor):
        """``x_host``: contiguous BF16 [M, K]; the result is written into it."""
        return self._make(self, x_host)


def _fp8_block_for(ew: ExpertWeights) -> Tuple[int, int]:
    if ew.gate.scheme == "block":
        return tuple(ew.gate.block)
    N = ew.intermediate_size
    for bn in (128, 64, 32):
        if N % bn == 0:
            return (bn, 128)
    raise KernelUnavailable(
        f"intermediate size {N} not a multiple of 32: per-tensor gate/up scales "
        f"cannot be mapped to CPU FP8 block scales"
    )


def _cpu_ops():
    try:
        from moe_infinity.kernel.cpu import load_cpu_moe

        return load_cpu_moe()
    except Exception as e:  # noqa: BLE001
        raise KernelUnavailable(
            f"MoE-Infinity CPU MoE kernels unavailable: {e!r}"
        )


def _make_fused(packed_template):
    from moe_infinity.kernel.cpu import fused_experts

    def make(r: CpuRunner, xh):
        t = r.tensors
        packed = replace(
            packed_template,
            w13=t["w13"],
            w2=t["w2"],
            w13_scale=t.get("w13_scale"),
            w2_scale=t.get("w2_scale"),
        )
        M = xh.shape[0]
        ids = torch.zeros(M, 1, dtype=torch.int32)
        tw = torch.ones(M, 1, dtype=torch.float32)

        def fn():
            return fused_experts(xh, packed, ids, tw, inplace=True)

        return fn, None

    return make


def _fp8_round(t: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """e4m3 code of ``t / scale`` as BF16 (exact: e4m3 values fit in BF16)."""
    q = (t.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.to(torch.bfloat16)


def _build_w8a8_emu(
    ew: ExpertWeights,
) -> Tuple[Dict[str, torch.Tensor], dict, Callable]:
    """Static per-tensor W8A8 on the CPU, numerically like the GPU paths.

    Same FP8 weights as the GPU kernels (w13 requantized to max(gate, up)
    scale, like SGLang's Fp8MoEMethod), activations rounded to e4m3 with the
    same static input scales and clamp (x for gate/up, act(gate)*up for
    down).  The e4m3 codes are fed to SGLang's CPU FP8 GEMM
    (``fp8_scaled_mm_cpu``) as exact BF16 values, with the activation scale
    folded into the weight block scales, so the products are the W8A8
    products; the kernel still computes in BF16 (no FP8 units on x86).
    """
    ops = _cpu_ops()
    if not ew.gate.is_fp8 or ew.gate.scheme != "per_tensor":
        raise KernelUnavailable(
            f"sglang_fp8_w8a8_emu needs per-tensor FP8 weights (checkpoint is {ew.quant})"
        )
    a1, a2 = ew.static_input_scales()
    w13, s13 = merged_per_tensor_w13(ew)
    s2 = float(ew.down.scale.reshape(()))
    bn, bk = _fp8_block_for(ew)
    N, K = ew.intermediate_size, ew.hidden_size

    def grid(rows, cols, value):
        return torch.full(
            (-(-rows // bn), -(-cols // bk)), float(value), dtype=torch.float32
        )

    static = a1 is not None and a2 is not None
    tensors = {
        "w13": ops.convert_weight_packed(w13.contiguous()),
        "w2": ops.convert_weight_packed(ew.down.weight.contiguous()),
        # Static: activation scale folded in.  Dynamic: weight scale only,
        # the per-call activation scale is applied to a copy of the grid.
        "w13_scale": grid(2 * N, K, s13 * (a1 if static else 1.0)),
        "w2_scale": grid(K, N, s2 * (a2 if static else 1.0)),
    }
    if static:
        tensors["a1_scale"] = torch.tensor(a1, dtype=torch.float32)
        tensors["a2_scale"] = torch.tensor(a2, dtype=torch.float32)
    meta = {
        "format": f"fp8 e4m3 weights (w13 at max(gate,up) scale) + e4m3-rounded "
        f"activations, {'static' if static else 'dynamic per-tensor'} scales; "
        f"SGLang fp8_scaled_mm_cpu, block {[bn, bk]}",
        "act_quant": "static (checkpoint input_scale)"
        if static
        else "dynamic per-tensor",
        "numerics": "W8A8 (activations and weights FP8), computed in BF16",
    }
    act = ew.activation

    def make(r: CpuRunner, xh):
        t = r.tensors
        w13p, w2p = t["w13"], t["w2"]
        if w13p.dim() == 3:
            w13p, w2p = w13p[0], w2p[0]
        blk = [bn, bk]

        def scales(x, key, w_grid):
            if static:
                return t[key], w_grid
            s = (x.abs().amax().float() / FP8_MAX).clamp(min=1e-12)
            return s, w_grid * s

        def fn():
            t0 = time.perf_counter()
            sa, g13 = scales(xh, "a1_scale", t["w13_scale"])
            xq = _fp8_round(xh, sa)
            t1 = time.perf_counter()
            h = ops.fp8_scaled_mm_cpu(
                xq, w13p, g13, blk, None, torch.bfloat16, True
            )
            a = _act_mul(h, act)
            t2 = time.perf_counter()
            sb, g2 = scales(a, "a2_scale", t["w2_scale"])
            aq = _fp8_round(a, sb)
            t3 = time.perf_counter()
            xh.copy_(
                ops.fp8_scaled_mm_cpu(
                    aq, w2p, g2, blk, None, torch.bfloat16, True
                )
            )
            t4 = time.perf_counter()
            r.last = {
                "act_quant_ms": ((t1 - t0) + (t3 - t2)) * 1e3,
                "gemm_ms": ((t2 - t1) + (t4 - t3)) * 1e3,
            }
            return xh

        return fn, None

    return tensors, meta, make


def build_cpu_runner(ew: ExpertWeights, kernel: str) -> CpuRunner:
    if kernel not in CPU_KERNELS:
        raise KernelUnavailable(
            f"unknown CPU kernel {kernel!r}; choose from {CPU_KERNELS}"
        )
    act = ew.activation
    t0 = time.perf_counter()
    if kernel == "torch_bf16":
        tensors = {
            "w13": torch.cat(
                [
                    ew.gate.dequant(torch.bfloat16),
                    ew.up.dequant(torch.bfloat16),
                ],
                0,
            ).contiguous(),
            "w2": ew.down.dequant(torch.bfloat16).contiguous(),
        }

        def make(r, xh):
            w13, w2 = r.tensors["w13"], r.tensors["w2"]

            def fn():
                xh.copy_(_act_mul(xh @ w13.t(), act) @ w2.t())
                return xh

            return fn, None

        meta = {
            "format": "bf16, oneDNN matmul"
            + (
                " (non-FP8 comparison)" if ew.gate.is_fp8 else f" [{ew.source}]"
            ),
            "numerics": "BF16",
        }
        return CpuRunner(
            kernel, (time.perf_counter() - t0) * 1e3, meta, tensors, make
        )

    _cpu_ops()
    from moe_infinity.kernel.cpu import (
        CpuExpertQuant,
        pack_experts,
        pack_fp8_experts,
    )

    if kernel == "sglang_fp8_w8a8_emu":
        tensors, meta, make = _build_w8a8_emu(ew)
        return CpuRunner(
            kernel, (time.perf_counter() - t0) * 1e3, meta, tensors, make
        )

    if kernel == "sglang_fp8_w8a16":
        if not ew.gate.is_fp8:
            raise KernelUnavailable(f"checkpoint is {ew.quant}, not FP8")
        if ew.gate.scheme not in ("per_tensor", "block"):
            raise KernelUnavailable(
                f"{ew.gate.scheme} FP8 scales have no exact CPU block-scale form"
            )
        blk = _fp8_block_for(ew)
        if ew.gate.scheme == "block" and ew.intermediate_size % blk[0]:
            raise KernelUnavailable(
                "gate/up block scales not aligned to the block size"
            )
        w13 = torch.cat([ew.gate.weight, ew.up.weight], 0)[None]
        s13 = torch.cat([ew.gate.scale_grid(blk), ew.up.scale_grid(blk)], 0)[
            None
        ]
        w2 = ew.down.weight[None]
        s2 = ew.down.scale_grid(blk)[None]
        packed = pack_fp8_experts(w13, w2, s13, s2, blk, activation=act)
        meta = {
            "format": f"fp8 e4m3 weights kept in memory, block scales {list(blk)}"
            + (
                " (broadcast from per-tensor)"
                if ew.gate.scheme == "per_tensor"
                else ""
            )
            + "; dequantized to BF16 inside the kernel (W8A16)",
            "numerics": "W8A16 (BF16 activations)",
        }
    else:
        w13 = torch.cat([ew.gate.dequant(), ew.up.dequant()], 0)[None]
        w2 = ew.down.dequant()[None]
        quant = (
            CpuExpertQuant.BF16
            if kernel == "sglang_bf16"
            else CpuExpertQuant.INT8_W8A8
        )
        if quant == CpuExpertQuant.INT8_W8A8 and act != "silu":
            raise KernelUnavailable("int8_w8a8 CPU experts support silu only")
        packed = pack_experts(w13, w2, quant, activation=act)
        meta = {
            "format": (
                "bf16 (VNNI-packed), AMX-BF16 / AVX512-BF16, no dequant"
                if quant == CpuExpertQuant.BF16
                else "int8 per-channel weights, dynamic per-token int8 activations"
            )
            + (
                " (non-FP8 comparison)" if ew.gate.is_fp8 else f" [{ew.source}]"
            ),
            "numerics": "BF16" if quant == CpuExpertQuant.BF16 else "INT8 W8A8",
        }
    tensors = {"w13": packed.w13, "w2": packed.w2}
    if packed.w13_scale is not None:
        tensors["w13_scale"] = packed.w13_scale
        tensors["w2_scale"] = packed.w2_scale
    return CpuRunner(
        kernel,
        (time.perf_counter() - t0) * 1e3,
        meta,
        tensors,
        _make_fused(packed),
    )


def reference_expert_w8a8(x: torch.Tensor, ew: ExpertWeights) -> torch.Tensor:
    """FP32 reference of static per-tensor W8A8 (what the GPU paths compute).

    Weights: the FP8 codes the GPU kernels use (w13 at max(gate, up) scale);
    activations: x and act(gate)*up rounded to e4m3 with the checkpoint's
    static input scales (dynamic per-tensor if the checkpoint has none).
    """
    dev = x.device
    w13, s13 = merged_per_tensor_w13(ew)
    w13 = w13.to(dev).float() * s13
    w2 = ew.down.weight.to(dev).float() * float(ew.down.scale.reshape(()))
    a1, a2 = ew.static_input_scales()

    def rnd(t, s):
        if s is None:
            s = float((t.abs().amax() / FP8_MAX).clamp(min=1e-12))
        return (t / s).clamp(-FP8_MAX, FP8_MAX).to(
            torch.float8_e4m3fn
        ).float() * s

    h = rnd(x.float(), a1) @ w13.t()
    return rnd(_act_mul(h, ew.activation), a2) @ w2.t()


def expert_flops(M: int, K: int, N: int) -> float:
    return 2.0 * M * K * (2 * N) + 2.0 * M * N * K


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GiB"


__all__ = [
    "CPU_KERNELS",
    "GPU_KERNELS",
    "CheckpointError",
    "KernelUnavailable",
    "build_cpu_runner",
    "build_gpu_runner",
    "expert_flops",
    "reference_expert",
    "reference_expert_w8a8",
    "rel_err",
    "resolve_gpu_kernels",
]
