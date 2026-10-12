# Copyright (c) 2025-2026 BatchGen Team.
# SPDX-License-Identifier: Apache-2.0
#
# Vendored from BatchGen (https://github.com/batchgen-project/batchgen),
# commit dc170ac27b7ac1fe2e6470da49112339ca5438cf, file
# batchgen_kernels/triton/moe_weighted_sum.py.
# Licensed under the Apache License, Version 2.0; see LICENSE and NOTICE in
# this directory.
#
# Modified for MoE-Infinity: trimmed to the v1 kernels and the
# moe_weighted_sum_triton launcher used by fused_moe_bf16 (lines 1-209 of the
# upstream file). Kernel code is unchanged.

import torch
import triton
import triton.language as tl

@triton.jit
def moe_weighted_sum_kernel(
    global_results_ptr,
    topk_weight_ptr,
    output_ptr,
    num_tokens,
    topk,
    hidden_size,
    BLOCK_SIZE_TOKENS: tl.constexpr,
    BLOCK_SIZE_HIDDEN: tl.constexpr,
):
    """
    Fused kernel for: weighted_output = global_results * topk_weight.unsqueeze(-1)
                     output = weighted_output.sum(dim=1)

    Args:
        global_results: [num_tokens, topk, hidden_size]
        topk_weight: [num_tokens, topk]
        output: [num_tokens, hidden_size]
    """
    # Program IDs
    pid_token = tl.program_id(0)
    pid_hidden = tl.program_id(1)

    # Token range for this CTA
    token_start = pid_token * BLOCK_SIZE_TOKENS

    # Hidden dimension range
    hidden_start = pid_hidden * BLOCK_SIZE_HIDDEN

    # Create offset arrays
    token_offsets = token_start + tl.arange(0, BLOCK_SIZE_TOKENS)
    hidden_offsets = hidden_start + tl.arange(0, BLOCK_SIZE_HIDDEN)

    # Create masks
    token_mask = token_offsets < num_tokens
    hidden_mask = hidden_offsets < hidden_size

    # Initialize output accumulator for this block
    output_acc = tl.zeros((BLOCK_SIZE_TOKENS, BLOCK_SIZE_HIDDEN), dtype=tl.float32)

    # Sum across all experts
    for expert_idx in range(topk):
        # Load weights for all tokens in this block
        weight_indices = token_offsets * topk + expert_idx
        weights = tl.load(
            topk_weight_ptr + weight_indices,
            mask=token_mask,
            other=0.0
        )

        # Load global_results for all tokens and current expert using vectorized operations
        # Calculate base indices for all tokens
        token_base_indices = token_offsets * topk * hidden_size + expert_idx * hidden_size

        # Create 2D indices: [BLOCK_SIZE_TOKENS, BLOCK_SIZE_HIDDEN]
        load_indices = token_base_indices[:, None] + hidden_offsets[None, :]
        load_mask = token_mask[:, None] & hidden_mask[None, :]

        # Load values for all tokens and hidden dims in this block
        values = tl.load(
            global_results_ptr + load_indices,
            mask=load_mask,
            other=0.0
        )

        # Multiply by weights (broadcast weights to match values shape)
        weighted_values = values * weights[:, None]

        # Accumulate
        output_acc += weighted_values

    # Store results
    store_indices = token_offsets[:, None] * hidden_size + hidden_offsets[None, :]
    store_mask = token_mask[:, None] & hidden_mask[None, :]

    tl.store(
        output_ptr + store_indices,
        output_acc,
        mask=store_mask
    )


@triton.jit
def moe_weighted_sum_kernel_optimized(
    global_results_ptr,
    topk_weight_ptr,
    output_ptr,
    num_tokens,
    topk,
    hidden_size,
    BLOCK_SIZE_TOKENS: tl.constexpr,
    BLOCK_SIZE_HIDDEN: tl.constexpr,
):
    """
    More optimized version that processes multiple tokens vectorially.
    """
    # Program IDs
    pid_token = tl.program_id(0)
    pid_hidden = tl.program_id(1)

    # Token and hidden dimension ranges
    token_start = pid_token * BLOCK_SIZE_TOKENS
    hidden_start = pid_hidden * BLOCK_SIZE_HIDDEN

    # Create offset arrays
    token_offsets = token_start + tl.arange(0, BLOCK_SIZE_TOKENS)
    hidden_offsets = hidden_start + tl.arange(0, BLOCK_SIZE_HIDDEN)

    # Create masks
    token_mask = token_offsets < num_tokens
    hidden_mask = hidden_offsets < hidden_size

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_TOKENS, BLOCK_SIZE_HIDDEN), dtype=tl.float32)

    # Process each expert
    for expert_idx in range(topk):
        # Load weights for current expert across all tokens in block
        weight_ptrs = token_offsets[:, None] * topk + expert_idx
        weights = tl.load(
            topk_weight_ptr + weight_ptrs,
            mask=token_mask[:, None],
            other=0.0
        )  # Shape: [BLOCK_SIZE_TOKENS, 1]

        # Load global results for current expert
        # Calculate base pointers for each token
        token_base_ptrs = (token_offsets[:, None] * topk + expert_idx) * hidden_size
        result_ptrs = token_base_ptrs + hidden_offsets[None, :]

        # Load values with proper masking
        mask_2d = token_mask[:, None] & hidden_mask[None, :]
        values = tl.load(
            global_results_ptr + result_ptrs,
            mask=mask_2d,
            other=0.0
        )  # Shape: [BLOCK_SIZE_TOKENS, BLOCK_SIZE_HIDDEN]

        # Multiply by weights and accumulate
        weighted_values = values * weights
        acc += weighted_values

    # Store results
    output_ptrs = token_offsets[:, None] * hidden_size + hidden_offsets[None, :]
    mask_2d = token_mask[:, None] & hidden_mask[None, :]

    tl.store(
        output_ptr + output_ptrs,
        acc,
        mask=mask_2d
    )


def moe_weighted_sum_triton(global_results, topk_weight, use_optimized=True, version="v1"):
    """
    Triton implementation of fused multiply and sum operation.

    Args:
        global_results: torch.Tensor of shape [num_tokens, topk, hidden_size]
        topk_weight: torch.Tensor of shape [num_tokens, topk]
        use_optimized: bool, whether to use the optimized kernel version
        version: str, "v1" for original optimized, "v2" for truly optimized

    Returns:
        output: torch.Tensor of shape [num_tokens, hidden_size]
    """
    num_tokens, topk_val, hidden_size = global_results.shape
    assert topk_weight.shape == (num_tokens, topk_val), f"Shape mismatch: {topk_weight.shape} vs {(num_tokens, topk_val)}"

    # Create output tensor
    output = torch.empty(
        (num_tokens, hidden_size),
        device=global_results.device,
        dtype=global_results.dtype
    )

    # Tunable block sizes
    BLOCK_SIZE_TOKENS = min(32, triton.next_power_of_2(num_tokens)) if num_tokens < 32 else 32
    BLOCK_SIZE_HIDDEN = min(256, triton.next_power_of_2(hidden_size))

    # Ensure minimum block sizes for efficiency
    BLOCK_SIZE_TOKENS = max(4, BLOCK_SIZE_TOKENS)
    BLOCK_SIZE_HIDDEN = max(32, BLOCK_SIZE_HIDDEN)

    # Calculate grid dimensions
    grid_tokens = triton.cdiv(num_tokens, BLOCK_SIZE_TOKENS)
    grid_hidden = triton.cdiv(hidden_size, BLOCK_SIZE_HIDDEN)

    # Choose kernel
    kernel = moe_weighted_sum_kernel_optimized if use_optimized else moe_weighted_sum_kernel

    # Launch kernel
    kernel[(grid_tokens, grid_hidden)](
        global_results,
        topk_weight,
        output,
        num_tokens,
        topk_val,
        hidden_size,
        BLOCK_SIZE_TOKENS=BLOCK_SIZE_TOKENS,
        BLOCK_SIZE_HIDDEN=BLOCK_SIZE_HIDDEN,
    )

    return output
