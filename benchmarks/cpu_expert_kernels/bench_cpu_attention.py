# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Decode GQA attention on CPU: MoE-Gen AVX2 kernel vs SGLang decode kernel.

Both read the same BF16 KV data (MoE-Gen: one [L, H_kv, D] buffer per
sequence; SGLang: the same rows in a token pool addressed by req_to_token)
and are checked against FP32 SDPA.

  python benchmarks/cpu_expert_kernels/bench_cpu_attention.py \
      --moegen-src /path/MoE-Gen/core/Hetero_Attn/CPU_Kernels \
      --sglang-src /path/sglang/python/sglang/kernels/aot/csrc/cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench_ext import host_info, load_bench_ext  # noqa: E402
from bench_cpu_moe import bench  # noqa: E402

# name: (q heads, kv heads, head dim)
ATTN = {
    "mixtral-8x7b": (32, 8, 128),
    "qwen3-30b-a3b": (32, 4, 128),
    "olmoe-1b-7b": (16, 16, 128),
}


def reference(q, keys, values, group):
    outs = []
    for b in range(q.shape[0]):
        k = keys[b].float().transpose(0, 1).repeat_interleave(group, 0)
        v = values[b].float().transpose(0, 1).repeat_interleave(group, 0)
        qb = q[b].float()  # [H, D]
        s = torch.einsum("hd,hld->hl", qb, k) / qb.shape[-1] ** 0.5
        outs.append(torch.einsum("hl,hld->hd", s.softmax(-1), v))
    return torch.stack(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moegen-src", required=True)
    ap.add_argument("--sglang-src", required=True)
    ap.add_argument("--models", nargs="+", default=list(ATTN))
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 4, 16, 64])
    ap.add_argument("--cache-lens", nargs="+", type=int, default=[512, 2048])
    ap.add_argument("--kv-splits", type=int, default=8)
    ap.add_argument("--json")
    args = ap.parse_args()

    ext = load_bench_ext(args.moegen_src, args.sglang_src)
    threads = torch.get_num_threads()
    info = host_info()
    print(json.dumps(info, indent=1))
    results = []
    for name in args.models:
        H, Hkv, D = ATTN[name]
        group = H // Hkv
        for L in args.cache_lens:
            for B in args.batches:
                g = torch.Generator().manual_seed(B * 7 + L)
                q = torch.randn(B, H, D, generator=g).to(torch.bfloat16)
                pool_k = torch.randn(B * L, Hkv, D, generator=g).to(
                    torch.bfloat16
                )
                pool_v = torch.randn(B * L, Hkv, D, generator=g).to(
                    torch.bfloat16
                )
                keys = [pool_k[b * L : (b + 1) * L] for b in range(B)]
                values = [pool_v[b * L : (b + 1) * L] for b in range(B)]
                mask = torch.zeros(B, L, dtype=torch.bfloat16)
                ref = reference(q, keys, values, group)

                q4 = q.view(B, H, 1, D).contiguous()

                def run_moegen():
                    return ext.moegen_gqa(
                        q4, keys, values, mask, L, group, H, Hkv, D, threads
                    )

                req_to_token = torch.arange(B * L, dtype=torch.int32).view(B, L)
                req_pool_indices = torch.arange(B, dtype=torch.int64)
                seq_lens = torch.full((B,), L, dtype=torch.int64)
                loc = torch.zeros(B, dtype=torch.int64)
                attn_logits = torch.empty(
                    B, H, args.kv_splits, D + 1, dtype=torch.float32
                )
                out_sgl = torch.empty(B, H, D, dtype=torch.bfloat16)

                def run_sglang():
                    ext.sglang_decode(
                        q,
                        pool_k,
                        pool_v,
                        out_sgl,
                        loc,
                        attn_logits,
                        req_to_token,
                        req_pool_indices,
                        seq_lens,
                        1.0 / D**0.5,
                    )
                    return out_sgl

                kv_bytes = 2 * B * L * Hkv * D * 2
                for backend, fn in (
                    ("moegen_avx2_omp", run_moegen),
                    ("sglang_decode", run_sglang),
                ):
                    out = fn().reshape(B, H, D).float()
                    err = float((out - ref).norm() / ref.norm())
                    t = bench(fn, min_iters=5, min_time=0.5)
                    rec = {
                        "model": name,
                        "batch": B,
                        "cache_len": L,
                        "backend": backend,
                        "ms": t * 1e3,
                        "kv_gbps": kv_bytes / t / 1e9,
                        "rel_err_vs_fp32": err,
                    }
                    results.append(rec)
                    print(
                        f"{name:14s} L={L:<5d} B={B:<3d} {backend:16s} "
                        f"{t * 1e3:9.3f} ms  {rec['kv_gbps']:7.1f} GB/s(kv) "
                        f"err={err:.4f}",
                        flush=True,
                    )
    if args.json:
        Path(args.json).write_text(
            json.dumps({"host": info, "results": results}, indent=1)
        )


if __name__ == "__main__":
    main()
