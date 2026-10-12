# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Expert FFN on CPU with SGLang's fused MoE kernels (AMX / AVX512-BF16).

Weights use SGLang's layout: ``w13 = cat([gate, up], dim=1)`` of shape
``[E, 2N, K]`` and ``w2`` (down) of shape ``[E, K, N]``; the expert computes
``down(act(gate(x)) * up(x))``.  ``pack_experts`` converts them once into the
kernel's VNNI-blocked layout (same byte size as the input for BF16), after
which ``fused_experts`` runs gate/up GEMM + activation + down GEMM + top-k
weighted sum in one call.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from ._ext import load_cpu_moe

FP8_E4M3_MAX = 448.0
DEFAULT_FP8_BLOCK = (128, 128)


class CpuExpertQuant(str, enum.Enum):
    BF16 = "bf16"
    INT8_W8A8 = "int8_w8a8"
    FP8_W8A16 = "fp8_w8a16"

    @property
    def comp_method(self) -> int:
        # Mirrors CPUQuantMethod in extensions/kernel/cpu/sglang/gemm.h.
        return {"bf16": 0, "int8_w8a8": 1, "fp8_w8a16": 2}[self.value]


@dataclass
class PackedCpuExperts:
    w13: torch.Tensor
    w2: torch.Tensor
    quant: CpuExpertQuant
    num_experts: int
    hidden_size: int
    intermediate_size: int
    activation: str = "silu"
    w13_scale: Optional[torch.Tensor] = None
    w2_scale: Optional[torch.Tensor] = None
    block_size: Optional[list] = None

    @property
    def nbytes(self) -> int:
        total = self.w13.numel() * self.w13.element_size()
        total += self.w2.numel() * self.w2.element_size()
        for s in (self.w13_scale, self.w2_scale):
            if s is not None:
                total += s.numel() * s.element_size()
        return total


def _quant_int8_per_channel(w: torch.Tensor):
    w = w.float()
    scale = w.abs().amax(dim=-1).clamp(min=1e-8) / 127.0
    q = torch.round(w / scale.unsqueeze(-1)).clamp(-128, 127).to(torch.int8)
    return q.contiguous(), scale.contiguous()


def _quant_fp8_block(w: torch.Tensor, block: Sequence[int]):
    E, OC, IC = w.shape
    bn, bk = block
    nb_n, nb_k = math.ceil(OC / bn), math.ceil(IC / bk)
    pad = torch.zeros(E, nb_n * bn, nb_k * bk, dtype=torch.float32)
    pad[:, :OC, :IC] = w.float()
    blocks = pad.view(E, nb_n, bn, nb_k, bk)
    amax = blocks.abs().amax(dim=(2, 4)).clamp(min=1e-8)
    scale = amax / FP8_E4M3_MAX
    q = (blocks / scale[:, :, None, :, None]).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    q = q.view(E, nb_n * bn, nb_k * bk)[:, :OC, :IC]
    return q.to(torch.float8_e4m3fn).contiguous(), scale.contiguous()


def dequant_fp8_block(q: torch.Tensor, scale: torch.Tensor, block):
    E, OC, IC = q.shape
    bn, bk = block
    s = scale.repeat_interleave(bn, dim=1)[:, :OC]
    s = s.repeat_interleave(bk, dim=2)[:, :, :IC]
    return q.float() * s


def pack_experts(
    w13: torch.Tensor,
    w2: torch.Tensor,
    quant: CpuExpertQuant | str = CpuExpertQuant.BF16,
    activation: str = "silu",
    fp8_block: Sequence[int] = DEFAULT_FP8_BLOCK,
) -> PackedCpuExperts:
    """Prepack stacked expert weights for ``fused_experts``.

    ``w13``: ``[E, 2N, K]`` (gate rows first, then up), ``w2``: ``[E, K, N]``.
    INT8 uses symmetric per-output-channel weight scales with dynamic per-token
    activation quantization (W8A8); FP8 stores e4m3 weights with block scales
    and computes in BF16 (W8A16).
    """
    ops = load_cpu_moe()
    quant = CpuExpertQuant(quant)
    if w13.dim() != 3 or w2.dim() != 3:
        raise ValueError("w13 and w2 must be [E, OC, IC]")
    E, two_n, K = w13.shape
    N = two_n // 2
    if w2.shape != (E, K, N):
        raise ValueError(
            f"w2 shape {tuple(w2.shape)} != expected {(E, K, N)} for w13 "
            f"{tuple(w13.shape)}"
        )
    if quant != CpuExpertQuant.BF16 and activation != "silu":
        if quant == CpuExpertQuant.INT8_W8A8:
            raise ValueError("int8_w8a8 CPU experts support silu only")

    w13_scale = w2_scale = block_size = None
    if quant == CpuExpertQuant.BF16:
        w13_p = w13.to(torch.bfloat16).contiguous()
        w2_p = w2.to(torch.bfloat16).contiguous()
    elif quant == CpuExpertQuant.INT8_W8A8:
        w13_p, w13_scale = _quant_int8_per_channel(w13)
        w2_p, w2_scale = _quant_int8_per_channel(w2)
    else:
        w13_p, w13_scale = _quant_fp8_block(w13, fp8_block)
        w2_p, w2_scale = _quant_fp8_block(w2, fp8_block)
        block_size = list(fp8_block)

    return PackedCpuExperts(
        w13=ops.convert_weight_packed(w13_p),
        w2=ops.convert_weight_packed(w2_p),
        quant=quant,
        num_experts=E,
        hidden_size=K,
        intermediate_size=N,
        activation=activation,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        block_size=block_size,
    )


def fused_experts(
    hidden_states: torch.Tensor,
    packed: PackedCpuExperts,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """``sum_k w[t,k] * expert[ids[t,k]](x[t])`` for every token, in BF16.

    ``topk_ids`` index into ``packed`` (``0 .. E-1``); every slot must be a
    valid local expert.
    """
    ops = load_cpu_moe()
    x = hidden_states.to(torch.bfloat16).contiguous()
    return ops.fused_experts_cpu(
        x,
        packed.w13,
        packed.w2,
        topk_weights.to(torch.float32).contiguous(),
        topk_ids.to(torch.int32).contiguous(),
        False,
        packed.quant.comp_method,
        packed.w13_scale,
        packed.w2_scale,
        None,
        None,
        packed.block_size,
        None,
        None,
        None,
        None,
        True,
        packed.activation,
    )


def amx_selfcheck(repeats: int = 8, seed: int = 0) -> dict:
    """Run a brgemm-sized BF16 expert batch repeatedly and compare.

    The BF16 kernel switches to oneDNN brgemm (AMX when the CPU has it) above
    4 rows per expert. Some virtualized hosts do not preserve AMX tile state
    across vCPU preemption, which shows up as run-to-run differences and
    errors far above BF16 rounding. If ``ok`` is False, start the process
    with ``ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16`` (AVX512 path, no AMX).
    """
    g = torch.Generator().manual_seed(seed)
    E, K, N, M, topk = 4, 2048, 1024, 64, 2
    w13 = (torch.randn(E, 2 * N, K, generator=g) / K**0.5).to(torch.bfloat16)
    w2 = (torch.randn(E, K, N, generator=g) / N**0.5).to(torch.bfloat16)
    x = torch.randn(M, K, generator=g).to(torch.bfloat16)
    tw, ids = torch.topk(torch.randn(M, E, generator=g).softmax(-1), topk)
    tw = (tw / tw.sum(-1, keepdim=True)).float()
    packed = pack_experts(w13, w2, CpuExpertQuant.BF16)
    ref = reference_experts(x, w13, w2, ids, tw)
    first = None
    max_err = 0.0
    deterministic = True
    for _ in range(repeats):
        out = fused_experts(x, packed, ids, tw).float()
        max_err = max(max_err, float((out - ref).norm() / ref.norm()))
        if first is None:
            first = out
        elif not torch.equal(out, first):
            deterministic = False
    return {
        "ok": deterministic and max_err < 1e-2,
        "deterministic": deterministic,
        "max_rel_err": max_err,
        "repeats": repeats,
    }


def _act_and_mul(x: torch.Tensor, activation: str) -> torch.Tensor:
    d = x.shape[-1] // 2
    if activation == "silu":
        return F.silu(x[..., :d]) * x[..., d:]
    if activation == "gelu":
        return F.gelu(x[..., :d]) * x[..., d:]
    raise ValueError(f"unsupported activation {activation!r}")


def reference_experts(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    activation: str = "silu",
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Per-expert loop in plain PyTorch (oneDNN/MKL GEMMs on CPU)."""
    x = hidden_states.to(compute_dtype)
    out = torch.zeros_like(x, dtype=torch.float32)
    flat_ids = topk_ids.reshape(-1)
    flat_w = topk_weights.reshape(-1).float()
    token_idx = torch.arange(x.shape[0]).repeat_interleave(topk_ids.shape[1])
    for e in torch.unique(flat_ids).tolist():
        sel = flat_ids == e
        rows = token_idx[sel]
        h = x[rows] @ w13[e].to(compute_dtype).t()
        h = _act_and_mul(h, activation) @ w2[e].to(compute_dtype).t()
        out.index_add_(0, rows, h.float() * flat_w[sel, None])
    return out
