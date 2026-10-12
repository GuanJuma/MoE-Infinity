# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""BatchGen MoE kernels (https://github.com/batchgen-project/batchgen).

The vendored Triton kernels live in ``vendor/`` under BatchGen's Apache-2.0
license.  The native engine runs them through AOT-compiled cubins when
``MOE_EXPERT_KERNEL=batchgen`` (or ``ArcherConfig.expert_kernel="batchgen"``);
see ``docs/batchgen-expert-kernels.md``.
"""

from __future__ import annotations

from typing import Any

EXPERT_KERNEL_ENV = "MOE_EXPERT_KERNEL"
EXPERT_KERNEL_CHOICES = ("default", "batchgen")


def fused_moe_bf16(*args: Any, **kwargs: Any):
    """BatchGen's grouped BF16 MoE forward (all experts in one launch)."""
    from .vendor.fused_moe_bf16 import fused_moe_bf16 as _impl

    return _impl(*args, **kwargs)


def expert_ffn(*args: Any, **kwargs: Any):
    """Single-expert gated FFN on BatchGen kernels; see ``expert_ffn.py``."""
    from .expert_ffn import expert_ffn as _impl

    return _impl(*args, **kwargs)


def native_kernel_stats(store_module: Any | None = None) -> dict[str, Any]:
    """Counters from the native launcher (calls, fallbacks, loaded archs)."""
    if store_module is None:
        import importlib

        store_module = importlib.import_module("moe_infinity._store")
    return dict(store_module.batchgen_expert_kernel_stats())


__all__ = [
    "EXPERT_KERNEL_CHOICES",
    "EXPERT_KERNEL_ENV",
    "expert_ffn",
    "fused_moe_bf16",
    "native_kernel_stats",
]
