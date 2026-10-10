# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""CPU expert-FFN benchmark on MoE model shapes.

Backends (all on the same routed batch, same weights):
  sglang_bf16 / sglang_int8 / sglang_fp8
      vendored SGLang fused_experts_cpu (moe_infinity.kernel.cpu)
  torch_bf16
      per-expert PyTorch loop (oneDNN BF16 GEMMs; what MoE-Infinity would run
      on CPU without a dedicated kernel)
  moegen_avx2
      expert GEMVs written with MoE-Gen's AVX2 helpers (needs --moegen-src and
      --sglang-src for the bench extension; decode batches only)

Example:
  python benchmarks/cpu_expert_kernels/bench_cpu_moe.py --json out.json \
      --moegen-src /path/MoE-Gen/core/Hetero_Attn/CPU_Kernels \
      --sglang-src /path/sglang/python/sglang/kernels/aot/csrc/cpu
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench_ext import host_info, load_bench_ext  # noqa: E402

from moe_infinity.kernel.cpu import (  # noqa: E402
    CpuExpertQuant,
    fused_experts,
    pack_experts,
    reference_experts,
)

# name: (hidden K, expert intermediate N, total experts, top-k, resident E)
# "resident E" < total experts keeps the working set inside this VM's RAM;
# decode M=1 always touches exactly top-k experts, so it is unaffected.
MODELS = {
    "olmoe-1b-7b": (2048, 1024, 64, 8, 64),
    "qwen3-30b-a3b": (2048, 768, 128, 8, 128),
    "deepseek-v2-lite": (2048, 1408, 64, 6, 64),
    "mixtral-8x7b": (4096, 14336, 8, 2, 2),
    "deepseek-v3": (7168, 2048, 256, 8, 8),
}
DECODE_M = [1, 4, 16, 64]
PREFILL_M = [256, 1024]


_FLUSH = None


def set_cold_cache(nbytes):
    """Evict the LLC before every timed call (0 = keep caches warm).

    Offloaded experts are streamed from DRAM; without a flush the few experts
    a repeated decode step touches stay in the (here 320 MB) L3.
    """
    global _FLUSH
    _FLUSH = torch.empty(nbytes // 4, dtype=torch.float32) if nbytes else None


def bench(fn, min_iters=5, min_time=0.5, warmup=1):
    for _ in range(warmup):
        fn()
    times = []
    spent = 0.0
    while len(times) < min_iters or spent < min_time:
        if _FLUSH is not None:
            _FLUSH.add_(1.0)
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        times.append(dt)
        spent += dt
        if len(times) >= 100:
            break
    return statistics.median(times)


def active_weight_bytes(ids, K, N, bytes_per_w):
    n_active = int(torch.unique(ids).numel())
    return n_active * 3 * K * N * bytes_per_w, n_active


def rel_err(out, ref):
    return float((out.float() - ref).norm() / ref.norm().clamp(min=1e-12))


def run_model(name, args, ext, results):
    K, N, E_total, topk, E = MODELS[name]
    g = torch.Generator().manual_seed(0)
    w13 = (torch.randn(E, 2 * N, K, generator=g) / K**0.5).to(torch.bfloat16)
    w2 = (torch.randn(E, K, N, generator=g) / N**0.5).to(torch.bfloat16)

    cases = []
    for M in args.batches:
        if args.decode_only and M not in DECODE_M:
            continue
        x = torch.randn(M, K, generator=g).to(torch.bfloat16)
        logits = torch.randn(M, E, generator=g)
        tw, ids = torch.topk(logits.softmax(-1), min(topk, E))
        tw = (tw / tw.sum(-1, keepdim=True)).float()
        ids = ids.to(torch.int32)
        ref = None
        if args.check and M <= 64:
            ref = reference_experts(x, w13, w2, ids, tw)
        cases.append((M, x, ids, tw, ref))

    def record(backend, M, t, ids, bytes_per_w, err):
        wb, n_act = active_weight_bytes(ids, K, N, bytes_per_w)
        flops = 2 * 3 * K * N * M * min(topk, E)
        rec = {
            "model": name,
            "phase": "decode" if M in DECODE_M else "prefill",
            "M": M,
            "backend": backend,
            "ms": t * 1e3,
            "tflops": flops / t / 1e12,
            "weight_gbps": wb / t / 1e9,
            "active_experts": n_act,
            "rel_err_vs_fp32": err,
            "K": K,
            "N": N,
            "E_resident": E,
            "E_total": E_total,
            "topk": topk,
        }
        results.append(rec)
        err_s = f"{err:.4f}" if err is not None else "-"
        print(
            f"{name:17s} {rec['phase']:7s} M={M:<5d} {backend:12s} "
            f"{t * 1e3:9.3f} ms  {rec['tflops']:6.3f} TFLOP/s  "
            f"{rec['weight_gbps']:7.1f} GB/s(w)  act={n_act:<3d} "
            f"err={err_s}",
            flush=True,
        )

    for q in args.quants:
        pk = pack_experts(w13, w2, q)
        bpw = 2 if q == "bf16" else 1
        for M, x, ids, tw, ref in cases:
            t = bench(lambda: fused_experts(x, pk, ids, tw))
            err = None
            if ref is not None:
                err = rel_err(fused_experts(x, pk, ids, tw), ref)
            record("sglang_" + q.split("_")[0], M, t, ids, bpw, err)
        del pk
        gc.collect()

    if args.torch_baseline:
        for M, x, ids, tw, ref in cases:

            def run():
                return reference_experts(
                    x, w13, w2, ids, tw, compute_dtype=torch.bfloat16
                )

            t = bench(run, min_time=0.5)
            err = rel_err(run(), ref) if ref is not None else None
            record("torch_bf16", M, t, ids, 2, err)

    if ext is not None:
        for M, x, ids, tw, ref in cases:
            if M > args.moegen_max_m:
                continue

            def run():
                return ext.moegen_style_expert_ffn(x, w13, w2, ids, tw)

            t = bench(run, min_iters=3, min_time=0.5, warmup=1)
            err = rel_err(run(), ref) if ref is not None else None
            record("moegen_avx2", M, t, ids, 2, err)

    del w13, w2, cases
    gc.collect()


def measure_bandwidth():
    a = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32).fill_(1)
    b = torch.empty_like(a)
    t = bench(lambda: b.copy_(a), min_iters=5, min_time=0.5)
    gbps = 2 * a.numel() * 4 / t / 1e9
    del a, b
    return gbps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument(
        "--batches", nargs="+", type=int, default=DECODE_M + PREFILL_M
    )
    ap.add_argument(
        "--quants",
        nargs="+",
        default=[q.value for q in CpuExpertQuant],
        choices=[q.value for q in CpuExpertQuant],
    )
    ap.add_argument(
        "--no-torch-baseline", dest="torch_baseline", action="store_false"
    )
    ap.add_argument("--no-check", dest="check", action="store_false")
    ap.add_argument("--decode-only", action="store_true")
    ap.add_argument("--moegen-src")
    ap.add_argument("--sglang-src")
    ap.add_argument("--moegen-max-m", type=int, default=4)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--json")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    ext = None
    if args.moegen_src and args.sglang_src:
        ext = load_bench_ext(args.moegen_src, args.sglang_src)
    info = host_info()
    info["copy_bandwidth_gbps"] = measure_bandwidth()
    print(json.dumps(info, indent=1))
    results = []
    for name in args.models:
        run_model(name, args, ext, results)
    if args.json:
        Path(args.json).write_text(
            json.dumps({"host": info, "results": results}, indent=1)
        )


if __name__ == "__main__":
    main()
