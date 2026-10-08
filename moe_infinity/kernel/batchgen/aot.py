# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

# EfficientMoE Team

"""Ahead-of-time build of BatchGen's expert-FFN kernels for the native engine.

MoE-Infinity computes each fetched expert inside C++ dispatcher threads, so
BatchGen's Triton kernels are compiled to cubins at build time and embedded
into the ``_store`` extension (``core/model/batchgen_moe.cpp`` launches them).
The variant table below is shared by the generated header and the Python
per-expert adapter in ``expert_ffn.py``.  The C++ launcher packs kernel
parameters in the order of ``GEMM_PARAMS`` / ``SILU_PARAMS``; keep the three
in sync.

Usage (run by setup.py and core/CMakeLists.txt)::

    python moe_infinity/kernel/batchgen/aot.py --archs 80,90 --out <header>
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PKG_DIR = Path(__file__).resolve().parent


def load_vendor_module() -> Any:
    """Import the vendored ``fused_moe_bf16`` module.

    Works both inside the installed package and when this file runs as a
    standalone build script (setup.py runs it before ``moe_infinity`` is
    importable).
    """
    if __package__:
        return importlib.import_module(f"{__package__}.vendor.fused_moe_bf16")
    name = "_moe_infinity_batchgen_vendor"
    if f"{name}.fused_moe_bf16" in sys.modules:
        return sys.modules[f"{name}.fused_moe_bf16"]
    vendor_dir = _PKG_DIR / "vendor"
    spec = importlib.util.spec_from_file_location(
        name,
        vendor_dir / "__init__.py",
        submodule_search_locations=[str(vendor_dir)],
    )
    assert spec is not None and spec.loader is not None
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    return importlib.import_module(f"{name}.fused_moe_bf16")


# Non-constexpr parameters of the per-expert specialisation of BatchGen's
# ``_fused_moe_gemm``, in kernel order.  Triton appends its scratch pointers
# after these (see ``scratch_arg_count``).
GEMM_PARAMS = (
    ("a_ptr", "*bf16"),
    ("b_ptr", "*bf16"),
    ("c_ptr", "*bf16"),
    ("sorted_token_ids_ptr", "*i32"),
    ("expert_ids_ptr", "*i32"),
    ("num_tokens_post_padded_ptr", "*i32"),
    ("N", "i32"),
    ("K", "i32"),
    ("EM", "i32"),
    ("num_valid_tokens", "i32"),
    ("stride_am", "i32"),
    ("stride_be", "i64"),
    ("stride_bn", "i32"),
    ("stride_cm", "i32"),
)
# One expert per launch: token slot == row, and A, B, C are row-major
# contiguous.  The launcher checks these before using the cubins.
GEMM_FIXED = {"top_k": 1, "stride_ak": 1, "stride_bk": 1, "stride_cn": 1}
# 16-element / 16-byte divisibility hints, mirroring what Triton's JIT infers
# for typical model shapes.  The launcher falls back when they do not hold.
GEMM_DIVISIBLE = (
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "sorted_token_ids_ptr",
    "expert_ids_ptr",
    "num_tokens_post_padded_ptr",
    "N",
    "K",
    "stride_am",
    "stride_bn",
    "stride_cm",
)

SILU_PARAMS = (
    ("x_ptr", "*bf16"),
    ("y_ptr", "*bf16"),
    ("N", "i32"),
    ("stride_xm", "i32"),
    ("stride_ym", "i32"),
)
SILU_DIVISIBLE = ("x_ptr", "y_ptr", "N", "stride_xm", "stride_ym")
SILU_BLOCK_N = 1024
SILU_NUM_WARPS = 4


@dataclass(frozen=True)
class Variant:
    name: str
    kernel: str  # "_fused_moe_gemm" or "_silu_and_mul_kernel"
    constexprs: dict[str, Any] = field(default_factory=dict)
    num_warps: int = 4
    num_stages: int = 3
    block_m: int = 0
    block_n: int = 0
    block_k: int = 0


def gemm_variants() -> list[Variant]:
    """BatchGen's two GEMM configs, each with and without ``EVEN_K``."""
    vendor = load_vendor_module()
    variants = []
    for label, config in (
        ("small", vendor._CONFIG_SMALL),
        ("large", vendor._CONFIG_LARGE),
    ):
        block_m, block_n, block_k, group_m, warps, stages = config
        for even_k in (True, False):
            constexprs = dict(GEMM_FIXED)
            constexprs.update(
                BLOCK_SIZE_M=block_m,
                BLOCK_SIZE_N=block_n,
                BLOCK_SIZE_K=block_k,
                GROUP_SIZE_M=group_m,
                EVEN_K=even_k,
            )
            variants.append(
                Variant(
                    name=f"gemm_{label}_{'evenk' if even_k else 'oddk'}",
                    kernel="_fused_moe_gemm",
                    constexprs=constexprs,
                    num_warps=warps,
                    num_stages=stages,
                    block_m=block_m,
                    block_n=block_n,
                    block_k=block_k,
                )
            )
    return variants


def silu_variant() -> Variant:
    return Variant(
        name="silu_and_mul",
        kernel="_silu_and_mul_kernel",
        constexprs={"BLOCK_N": SILU_BLOCK_N},
        num_warps=SILU_NUM_WARPS,
        num_stages=3,
        block_n=SILU_BLOCK_N,
    )


def all_variants() -> list[Variant]:
    return gemm_variants() + [silu_variant()]


def _params_for(variant: Variant):
    if variant.kernel == "_fused_moe_gemm":
        return GEMM_PARAMS, GEMM_DIVISIBLE
    return SILU_PARAMS, SILU_DIVISIBLE


def compile_variant(variant: Variant, arch: int):
    import triton
    import triton.compiler
    from triton.backends.compiler import GPUTarget

    vendor = load_vendor_module()
    fn = getattr(vendor, variant.kernel)
    params, divisible = _params_for(variant)
    types = dict(params)
    signature = {}
    for name in fn.arg_names:
        if name in variant.constexprs:
            signature[name] = "constexpr"
        elif name in types:
            signature[name] = types[name]
        else:
            raise RuntimeError(
                f"{variant.kernel}: parameter {name!r} has no AOT type"
            )
    attrs = {
        (fn.arg_names.index(name),): [["tt.divisibility", 16]]
        for name in divisible
    }
    src = triton.compiler.ASTSource(
        fn=fn,
        signature=signature,
        constexprs=dict(variant.constexprs),
        attrs=attrs,
    )
    target = GPUTarget("cuda", arch, 32)
    backend = triton.compiler.make_backend(target)
    options = backend.parse_options(
        {"num_warps": variant.num_warps, "num_stages": variant.num_stages}
    )
    compiled = triton.compile(src, target=target, options=options.__dict__)
    metadata = compiled.metadata
    for scratch in ("global_scratch_size", "profile_scratch_size"):
        if getattr(metadata, scratch, 0):
            raise RuntimeError(
                f"{variant.name} sm_{arch}: {scratch} > 0 is not supported"
            )
    return compiled


def scratch_arg_count(metadata: Any) -> int:
    # Triton's launcher appends one pointer per scratch kind that its
    # compiled-kernel metadata reports, whether or not the size is zero.
    return sum(
        hasattr(metadata, name)
        for name in ("global_scratch_size", "profile_scratch_size")
    )


def _c_bytes(data: bytes) -> str:
    rows = []
    for offset in range(0, len(data), 16):
        chunk = data[offset : offset + 16]
        rows.append("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",")
    return "\n".join(rows)


def render_header(archs: list[int]) -> str:
    import triton

    variants = all_variants()
    blobs = []
    entries = []
    scratch_args = None
    for variant in variants:
        for arch in archs:
            compiled = compile_variant(variant, arch)
            count = scratch_arg_count(compiled.metadata)
            if scratch_args is None:
                scratch_args = count
            elif scratch_args != count:
                raise RuntimeError("inconsistent Triton scratch arguments")
            cubin = compiled.asm["cubin"]
            symbol = f"k_{variant.name}_sm{arch}"
            blobs.append(
                f"alignas(16) static const unsigned char {symbol}[] = {{\n"
                f"{_c_bytes(cubin)}\n}};"
            )
            entries.append(
                f'  {{"{variant.name}", {arch}, "{compiled.metadata.name}", '
                f"{symbol}, sizeof({symbol}), {compiled.metadata.shared}u, "
                f"{variant.num_warps}, {variant.block_m}, {variant.block_n}, "
                f"{variant.block_k}}},"
            )
    archs_list = ", ".join(str(a) for a in archs)
    return "\n".join(
        [
            "// Generated by moe_infinity/kernel/batchgen/aot.py; do not edit.",
            "// Embeds cubins of BatchGen's Triton expert-FFN kernels",
            "// (https://github.com/batchgen-project/batchgen, Apache-2.0).",
            "#pragma once",
            "",
            "#include <cstddef>",
            "",
            "namespace moe_batchgen_aot {",
            "",
            f'constexpr const char* kTritonVersion = "{triton.__version__}";',
            f"constexpr int kTritonScratchArgs = {scratch_args or 0};",
            f"constexpr int kNumGemmParams = {len(GEMM_PARAMS)};",
            f"constexpr int kNumSiluParams = {len(SILU_PARAMS)};",
            f"constexpr int kArchs[] = {{{archs_list}}};",
            "",
            "struct CubinEntry {",
            "  const char* variant;",
            "  int arch;",
            "  const char* function;",
            "  const unsigned char* data;",
            "  std::size_t size;",
            "  unsigned shared_bytes;",
            "  int num_warps;",
            "  int block_m;",
            "  int block_n;",
            "  int block_k;",
            "};",
            "",
            *blobs,
            "",
            "static const CubinEntry kCubins[] = {",
            *entries,
            "};",
            "",
            "}  // namespace moe_batchgen_aot",
            "",
        ]
    )


def write_header(archs: list[int], out: Path) -> bool:
    """Write the header; returns False when the content is unchanged."""
    text = render_header(archs)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.read_text() == text:
        return False
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(out)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archs",
        default="80,90",
        help="Comma-separated SM versions, e.g. 80,90,120",
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    archs = sorted({int(a) for a in args.archs.split(",") if a.strip()})
    changed = write_header(archs, args.out)
    state = "wrote" if changed else "unchanged"
    print(f"[batchgen-aot] {state} {args.out} (sm_{archs})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
