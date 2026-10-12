# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""CPU expert-compute kernels (vendored from SGLang; see
extensions/kernel/cpu/sglang/NOTICE and docs/cpu-expert-offload.md)."""

from ._ext import CpuMoeUnavailable, cpu_isa_flags, is_available, load_cpu_moe
from .expert_ffn import (
    CpuExpertQuant,
    PackedCpuExperts,
    amx_selfcheck,
    dequant_fp8_block,
    fused_experts,
    pack_experts,
    reference_experts,
)

__all__ = [
    "CpuExpertQuant",
    "amx_selfcheck",
    "CpuMoeUnavailable",
    "PackedCpuExperts",
    "cpu_isa_flags",
    "dequant_fp8_block",
    "fused_experts",
    "is_available",
    "load_cpu_moe",
    "pack_experts",
    "reference_experts",
]
