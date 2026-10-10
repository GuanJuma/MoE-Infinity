# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Single-expert FFN runners: ``out = down(act(gate(x)) * up(x))``.

GPU kernels (``GPU_KERNELS``), all run on one expert (E=1, top-1, weight 1):

``sglang_triton``
    SGLang's Triton ``fused_experts`` -- the path SGLang itself uses for an
    FP8 MoE on sm_120 (no DeepGEMM/CUTLASS-FP8 MoE there).  Per-tensor FP8 is
    prepared exactly like SGLang's ``Fp8MoEMethod``: gate/up requantized to
    one scale (their max), static activation scales = checkpoint
    ``input_scale``.  Block-FP8 and per-channel FP8 use SGLang's block /
    per-channel paths with dynamic activation quantization.
``torch_scaled_mm``
    Two cuBLASLt FP8 GEMMs (``torch._scaled_mm``, per-tensor scales) plus
    SiLU*up and the FP8 activation casts in PyTorch.  Per-tensor FP8 only.
``torch_bf16``
    Weights dequantized once to BF16, two cuBLAS BF16 GEMMs.  Works for every
    checkpoint format and GPU; moves 2x the bytes of FP8 over PCIe.

CPU kernels (``CPU_KERNELS``):

``sglang_fp8_w8a16``
    MoE-Infinity's vendored SGLang ``fused_experts_cpu``, FP8 weights kept
    as e4m3 (per-tensor scales broadcast to 128x128 block scales, exact),
    dequantized to BF16 inside the kernel; AMX-BF16 / AVX512-BF16.
``sglang_bf16`` / ``sglang_int8_w8a8``
    Same kernel on BF16 weights / INT8 W8A8 requantized weights.
``torch_bf16``
    Plain PyTorch BF16 GEMMs (oneDNN) as a baseline.

Every runner is used through ``bind(x)`` which returns ``(fn, reset)``:
``fn()`` computes the expert on the bound input and returns the output
tensor; ``reset`` (or None) restores the input after an in-place run and is
always called outside the timed region.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from checkpoint_expert import CheckpointError, ExpertWeights

FP8_MAX = 448.0
GPU_KERNELS = ("sglang_triton", "torch_scaled_mm", "torch_bf16")
CPU_KERNELS = (
    "sglang_fp8_w8a16",
    "sglang_bf16",
    "sglang_int8_w8a8",
    "torch_bf16",
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
    if kernel == "torch_bf16":
        w13 = torch.cat(
            [ew.gate.dequant(torch.bfloat16), ew.up.dequant(torch.bfloat16)], 0
        )
        return {
            "w13": w13.contiguous(),
            "w2": ew.down.dequant(torch.bfloat16).contiguous(),
        }, {"format": "bf16 (dequantized once from the checkpoint)"}
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
            }, {"format": "bf16", "mode": "bf16"}
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


_BINDERS = {
    "torch_bf16": _bind_torch_bf16,
    "torch_scaled_mm": _bind_torch_scaled_mm,
    "sglang_triton": _bind_sglang_triton,
}


def build_gpu_runner(ew: ExpertWeights, kernel: str, device) -> GpuRunner:
    if kernel not in _BINDERS:
        raise KernelUnavailable(
            f"unknown GPU kernel {kernel!r}; choose from {GPU_KERNELS}"
        )
    if kernel == "sglang_triton":
        _load_sglang()
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
    out.append("torch_bf16")
    return out


# --------------------------------------------------------------------------
# CPU runners
# --------------------------------------------------------------------------


@dataclass
class CpuRunner:
    kernel: str
    pack_ms: float
    nbytes: int
    meta: dict
    _bind: Callable = None

    def bind(self, x_host: torch.Tensor):
        """``x_host``: contiguous BF16 [M, K]; the result is written into it."""
        return self._bind(x_host)


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


def build_cpu_runner(ew: ExpertWeights, kernel: str) -> CpuRunner:
    if kernel not in CPU_KERNELS:
        raise KernelUnavailable(
            f"unknown CPU kernel {kernel!r}; choose from {CPU_KERNELS}"
        )
    act = ew.activation
    t0 = time.perf_counter()
    if kernel == "torch_bf16":
        w13 = torch.cat(
            [ew.gate.dequant(torch.bfloat16), ew.up.dequant(torch.bfloat16)], 0
        ).contiguous()
        w2 = ew.down.dequant(torch.bfloat16).contiguous()
        pack_ms = (time.perf_counter() - t0) * 1e3

        def bind(xh):
            def fn():
                xh.copy_(_act_mul(xh @ w13.t(), act) @ w2.t())
                return xh

            return fn, None

        nbytes = (w13.numel() + w2.numel()) * 2
        return CpuRunner(
            kernel, pack_ms, nbytes, {"format": "bf16, oneDNN matmul"}, bind
        )

    try:
        from moe_infinity.kernel.cpu import (
            CpuExpertQuant,
            fused_experts,
            load_cpu_moe,
            pack_experts,
            pack_fp8_experts,
        )

        load_cpu_moe()
    except Exception as e:  # noqa: BLE001
        raise KernelUnavailable(
            f"MoE-Infinity CPU MoE kernels unavailable: {e!r}"
        )

    if kernel == "sglang_fp8_w8a16":
        if not ew.gate.is_fp8:
            raise KernelUnavailable(f"checkpoint is {ew.quant}, not FP8")
        if ew.gate.scheme not in ("per_tensor", "block"):
            raise KernelUnavailable(
                f"{ew.gate.scheme} FP8 scales have no exact CPU block-scale form; use sglang_bf16"
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
            "format": f"fp8 e4m3 W8A16, block scales {list(blk)}"
            + (
                " (broadcast from per-tensor)"
                if ew.gate.scheme == "per_tensor"
                else ""
            )
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
            "format": "bf16 (VNNI-packed)"
            if quant == CpuExpertQuant.BF16
            else "int8 per-channel weights, dynamic per-token int8 activations"
        }
    pack_ms = (time.perf_counter() - t0) * 1e3

    def bind(xh):
        M = xh.shape[0]
        ids = torch.zeros(M, 1, dtype=torch.int32)
        tw = torch.ones(M, 1, dtype=torch.float32)

        def fn():
            return fused_experts(xh, packed, ids, tw, inplace=True)

        return fn, None

    return CpuRunner(kernel, pack_ms, packed.nbytes, meta, bind)


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
    "rel_err",
    "resolve_gpu_kernels",
]
