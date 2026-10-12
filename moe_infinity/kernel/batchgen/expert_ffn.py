# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Per-expert gated FFN on BatchGen's grouped-GEMM kernels.

MoE-Infinity fetches experts one at a time and computes each one as soon as
its weights land on the GPU, so BatchGen's grouped ``fused_moe_bf16`` runs
here in its single-expert form: identity token order, expert id 0, top_k 1.
This is the Python twin of ``batchgen::ExpertFFN`` in
``core/model/batchgen_moe.cpp`` (same launches, same config choice) and is
used to validate that decomposition against the original expert math.
"""

from __future__ import annotations

import torch
import triton

from .aot import SILU_BLOCK_N, SILU_NUM_WARPS, load_vendor_module


def pick_config(rows: int):
    return load_vendor_module()._pick_config(rows)


def _identity_routing(rows: int, block_m: int, device):
    padded = triton.cdiv(rows, block_m) * block_m
    sorted_ids = torch.arange(padded, dtype=torch.int32, device=device)
    expert_ids = torch.zeros(
        padded // block_m, dtype=torch.int32, device=device
    )
    num_post = torch.tensor([padded], dtype=torch.int32, device=device)
    return sorted_ids, expert_ids, num_post, padded


def _gemm(a, b, c, routing, rows, n, k, config, stride_cm):
    vendor = load_vendor_module()
    block_m, block_n, block_k, group_m, warps, stages = config
    sorted_ids, expert_ids, num_post, padded = routing
    grid = (triton.cdiv(padded, block_m) * triton.cdiv(n, block_n),)
    vendor._fused_moe_gemm[grid](
        a,
        b,
        c,
        sorted_ids,
        expert_ids,
        num_post,
        n,
        k,
        padded,
        rows,
        1,
        a.stride(0),
        1,
        0,
        1,
        k,
        stride_cm,
        1,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        GROUP_SIZE_M=group_m,
        EVEN_K=(k % block_k == 0),
        num_warps=warps,
        num_stages=stages,
    )


def gate_up_packed(gate_w: torch.Tensor, up_w: torch.Tensor) -> bool:
    """True when ``up_w`` directly follows ``gate_w`` in one storage.

    That is BatchGen's packed ``w13 = [gate; up]`` layout, so stage 1 can run
    as a single GEMM over ``2N`` output columns.
    """
    return (
        gate_w.is_contiguous()
        and up_w.is_contiguous()
        and gate_w.shape == up_w.shape
        and gate_w.untyped_storage().data_ptr()
        == up_w.untyped_storage().data_ptr()
        and up_w.data_ptr()
        == gate_w.data_ptr() + gate_w.numel() * gate_w.element_size()
    )


def expert_ffn(
    x: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
) -> torch.Tensor:
    """``(silu(x @ gate_w.T) * (x @ up_w.T)) @ down_w.T`` for one expert.

    Args:
        x: ``[M, K]`` tokens routed to this expert.
        gate_w, up_w: ``[N, K]`` gate / up projections.
        down_w: ``[H, N]`` down projection.
    """
    if x.dim() != 2 or x.shape[0] == 0:
        raise ValueError(f"x must be a non-empty [M, K] tensor, got {x.shape}")
    rows, k = x.shape
    n = gate_w.shape[0]
    h = down_w.shape[0]
    if gate_w.shape != (n, k) or up_w.shape != (n, k):
        raise ValueError("gate_w and up_w must both be [N, K]")
    if down_w.shape != (h, n):
        raise ValueError("down_w must be [H, N]")
    x = x.contiguous()
    down_w = down_w.contiguous()

    config = pick_config(rows)
    routing = _identity_routing(rows, config[0], x.device)
    c1 = torch.empty((rows, 2 * n), device=x.device, dtype=x.dtype)
    if gate_up_packed(gate_w, up_w):
        w13 = torch.as_strided(gate_w, (2 * n, k), (k, 1))
        _gemm(x, w13, c1, routing, rows, 2 * n, k, config, 2 * n)
    else:
        gate_w = gate_w.contiguous()
        up_w = up_w.contiguous()
        _gemm(x, gate_w, c1, routing, rows, n, k, config, 2 * n)
        _gemm(x, up_w, c1[:, n:], routing, rows, n, k, config, 2 * n)

    act = torch.empty((rows, n), device=x.device, dtype=x.dtype)
    vendor = load_vendor_module()
    vendor._silu_and_mul_kernel[(rows, triton.cdiv(n, SILU_BLOCK_N))](
        c1,
        act,
        n,
        c1.stride(0),
        act.stride(0),
        BLOCK_N=SILU_BLOCK_N,
        num_warps=SILU_NUM_WARPS,
    )

    out = torch.empty((rows, h), device=x.device, dtype=x.dtype)
    _gemm(act, down_w, out, routing, rows, h, n, config, h)
    return out
