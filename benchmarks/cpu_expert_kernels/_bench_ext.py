# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""JIT build of bench_bindings.cpp against external MoE-Gen / SGLang trees.

MOEGEN_SRC: directory holding MoE-Gen's core/Hetero_Attn/CPU_Kernels
SGLANG_SRC: SGLang's CPU kernel directory (python/sglang/kernels/aot/csrc/cpu)
"""

from __future__ import annotations

import functools
import os
import platform
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _flags():
    from moe_infinity.kernel.cpu._ext import CPU_MOE_CXX_FLAGS

    return list(CPU_MOE_CXX_FLAGS)


@functools.lru_cache(maxsize=1)
def load_bench_ext(moegen_src: str, sglang_src: str):
    from torch.utils.cpp_extension import load

    moegen_kernel = (
        Path(moegen_src) / "grouped_query_attention_cpu_avx2_omp.cpp"
    )
    sglang_decode = Path(sglang_src) / "decode.cpp"
    for p in (moegen_kernel, sglang_decode):
        if not p.exists():
            raise FileNotFoundError(p)
    t0 = time.time()
    mod = load(
        name="moe_infinity_cpu_kernel_bench",
        sources=[
            str(HERE / "bench_bindings.cpp"),
            str(moegen_kernel),
            str(sglang_decode),
        ],
        extra_cflags=_flags(),
        extra_ldflags=["-fopenmp"],
        extra_include_paths=[str(moegen_src), str(sglang_src)],
        verbose=os.environ.get("BENCH_VERBOSE") == "1",
    )
    print(f"[bench ext built/loaded in {time.time() - t0:.1f}s]")
    return mod


def host_info() -> dict:
    import torch

    from moe_infinity.kernel.cpu import cpu_isa_flags

    model = ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {
        "cpu": model,
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "isa": cpu_isa_flags(),
    }
