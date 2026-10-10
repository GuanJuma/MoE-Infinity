# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Loader for the SGLang-derived CPU MoE kernels (``moe_infinity._cpu_moe``).

The extension is normally built by ``setup.py`` (``MOE_BUILD_CPU_MOE``).  When
it is missing -- e.g. a source checkout on a CPU-only box where the CUDA
extensions cannot be built -- it is JIT-compiled from
``extensions/kernel/cpu`` with ``torch.utils.cpp_extension.load`` unless
``MOE_CPU_MOE_JIT=0``.
"""

from __future__ import annotations

import functools
import os
import platform
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
CPU_KERNEL_DIR = _REPO_ROOT / "extensions" / "kernel" / "cpu"

CPU_MOE_SOURCES = [
    "cpu_moe_bindings.cpp",
    "sglang/gemm.cpp",
    "sglang/gemm_int8.cpp",
    "sglang/gemm_fp8.cpp",
    "sglang/gemm_int4.cpp",
    "sglang/moe.cpp",
    "sglang/moe_int8.cpp",
    "sglang/moe_fp8.cpp",
    "sglang/moe_int4.cpp",
]

# Same ISA baseline as upstream SGLang's CPU build: the AMX tiles are only
# executed when the CPU reports them (cpublas falls back to AVX512 brgemm).
CPU_MOE_CXX_FLAGS = [
    "-O3",
    "-std=c++20",
    "-fPIC",
    "-fopenmp",
    "-Wno-unknown-pragmas",
    "-march=x86-64-v4",
    "-mavx512bf16",
    "-mavx512vnni",
    "-mamx-tile",
    "-mamx-bf16",
    "-mamx-int8",
    "-DSGLANG_CPU_FP8_CVT_FTZ",
]


class CpuMoeUnavailable(RuntimeError):
    pass


def cpu_isa_flags() -> dict:
    flags = set()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    flags = set(line.split(":", 1)[1].split())
                    break
    except OSError:
        pass
    return {
        "avx512f": "avx512f" in flags,
        "avx512_bf16": "avx512_bf16" in flags,
        "avx512_vnni": "avx512_vnni" in flags,
        "amx_bf16": "amx_bf16" in flags,
        "amx_int8": "amx_int8" in flags,
        "avx512_fp16": "avx512_fp16" in flags,
    }


def _check_host() -> None:
    if platform.machine() not in ("x86_64", "AMD64"):
        raise CpuMoeUnavailable(
            f"CPU MoE kernels need x86_64, got {platform.machine()}"
        )
    isa = cpu_isa_flags()
    missing = [k for k in ("avx512f", "avx512_bf16") if not isa[k]]
    if missing:
        raise CpuMoeUnavailable(
            "CPU MoE kernels need AVX512F + AVX512_BF16 (missing: "
            + ", ".join(missing)
            + ")"
        )


def _jit_build():
    from torch.utils.cpp_extension import load

    sources = [str(CPU_KERNEL_DIR / s) for s in CPU_MOE_SOURCES]
    missing = [s for s in sources if not os.path.exists(s)]
    if missing:
        raise CpuMoeUnavailable(f"CPU MoE sources not found: {missing}")
    load(
        name="moe_infinity_cpu_moe_jit",
        sources=sources,
        extra_cflags=CPU_MOE_CXX_FLAGS,
        extra_ldflags=["-fopenmp"],
        extra_include_paths=[str(CPU_KERNEL_DIR / "sglang")],
        is_python_module=False,
        verbose=os.environ.get("MOE_CPU_MOE_JIT_VERBOSE") == "1",
    )


@functools.lru_cache(maxsize=1)
def load_cpu_moe():
    """Register ``torch.ops.moe_infinity_cpu`` and return that namespace."""
    _check_host()
    try:
        import moe_infinity._cpu_moe  # noqa: F401
    except ImportError:
        if os.environ.get("MOE_CPU_MOE_JIT", "1") == "0":
            raise CpuMoeUnavailable(
                "moe_infinity._cpu_moe is not built and MOE_CPU_MOE_JIT=0"
            )
        _jit_build()
    return torch.ops.moe_infinity_cpu


def is_available() -> bool:
    try:
        load_cpu_moe()
    except Exception:
        return False
    return True
