#!/usr/bin/env python3
"""A/B: default (CUTLASS) vs BatchGen expert FFN kernels.

micro  Per-expert kernel latency through the native bindings, on the expert
       shapes of common MoE models (no checkpoint needed).
e2e    End-to-end offloaded generation, once per kernel in a fresh process,
       comparing greedy outputs and latency / throughput.

Usage:
    python benchmarks/ab_batchgen_expert_kernel.py micro
    python benchmarks/ab_batchgen_expert_kernel.py e2e \
        --model allenai/OLMoE-1B-7B-0924-Instruct --offload-dir /tmp/moe-offload
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

KERNELS = ("default", "batchgen")
MODEL_SHAPES = {
    # name: (hidden K, expert intermediate N)
    "olmoe-1b-7b": (2048, 1024),
    "deepseek-v2-lite": (2048, 1408),
    "qwen3-30b-a3b": (2048, 768),
    "mixtral-8x7b": (4096, 14336),
}
PROMPTS = [
    "Explain why the sky is blue in two sentences.",
    "Write a Python function that checks whether a number is prime.",
    "List three differences between TCP and UDP.",
    "What is a mixture-of-experts model?",
]


def _time_cuda(fn, warmup: int, iters: int) -> float:
    import torch

    for _ in range(warmup):
        fn()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) * 1e3 / iters


def run_micro(args: argparse.Namespace) -> dict:
    import torch

    from moe_infinity import _store

    stats = _store.batchgen_expert_kernel_stats()
    if not stats["compiled_in"]:
        sys.exit("_store was built without BatchGen cubins")
    results = []
    for model, (k, n) in MODEL_SHAPES.items():
        w13 = torch.randn(2 * n, k, device="cuda", dtype=torch.bfloat16) * 0.02
        gate, up = w13[:n], w13[n:]
        down = torch.randn(k, n, device="cuda", dtype=torch.bfloat16) * 0.02
        for rows in args.rows:
            x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
            row = {"model": model, "K": k, "N": n, "rows": rows}
            for layout, (g, u) in {
                "w13": (gate, up),
                "split": (gate.clone(), up.clone()),
            }.items():
                bg = _time_cuda(
                    lambda: _store.batchgen_expert_ffn(x, g, u, down),
                    args.warmup,
                    args.iters,
                )
                row[f"batchgen_{layout}_us"] = round(bg, 2)
            row["default_us"] = round(
                _time_cuda(
                    lambda: _store.default_expert_ffn(x, gate, up, down),
                    args.warmup,
                    args.iters,
                ),
                2,
            )
            ref = _store.default_expert_ffn(x, gate, up, down).float()
            out = _store.batchgen_expert_ffn(x, gate, up, down).float()
            row["max_rel_diff"] = float(
                (out - ref).abs().max() / ref.abs().max().clamp_min(1e-6)
            )
            results.append(row)
            print(json.dumps(row))
    return {
        "mode": "micro",
        "device": torch.cuda.get_device_name(),
        "results": results,
    }


def run_child(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoTokenizer

    from moe_infinity import MoE

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = MoE(
        args.model,
        {
            "offload_path": args.offload_dir,
            "device_memory_ratio": args.device_memory_ratio,
            "expert_kernel": args.child,
        },
    )
    encoded = [
        tokenizer(p, return_tensors="pt").input_ids.to("cuda:0")
        for p in PROMPTS
    ]

    def generate(ids, max_new_tokens):
        with torch.no_grad():
            return model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                min_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

    generate(encoded[0], 4)  # warm up the expert cache and kernels
    prefill_s, total_s, outputs = [], [], []
    for _ in range(args.rounds):
        for ids in encoded:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            generate(ids, 1)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            out = generate(ids, args.max_new_tokens)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            prefill_s.append(t1 - t0)
            total_s.append(t2 - t1)
            outputs.append(out[0, ids.shape[1] :].tolist())

    from moe_infinity import _store

    new_tokens = args.max_new_tokens * len(total_s)
    decode_s = sum(t - p for t, p in zip(total_s, prefill_s))
    result = {
        "kernel": args.child,
        "prefill_latency_s": sum(prefill_s) / len(prefill_s),
        "request_latency_s": sum(total_s) / len(total_s),
        "decode_tokens_per_s": (new_tokens - len(total_s)) / decode_s,
        "end_to_end_tokens_per_s": new_tokens / sum(total_s),
        "outputs": outputs[: len(PROMPTS)],
        "texts": [
            tokenizer.decode(o, skip_special_tokens=True)
            for o in outputs[: len(PROMPTS)]
        ],
        "native_kernel_stats": dict(_store.batchgen_expert_kernel_stats()),
    }
    print("RESULT_JSON " + json.dumps(result))


def _first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def run_e2e(args: argparse.Namespace) -> dict:
    results = {}
    for kernel in KERNELS:
        cmd = [
            sys.executable,
            __file__,
            "e2e",
            "--child",
            kernel,
            "--model",
            args.model,
            "--offload-dir",
            args.offload_dir,
            "--device-memory-ratio",
            str(args.device_memory_ratio),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--rounds",
            str(args.rounds),
        ]
        env = dict(os.environ, MOE_EXPERT_KERNEL=kernel)
        print(f"== {kernel}: {' '.join(cmd)}", flush=True)
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        lines = [
            l for l in proc.stdout.splitlines() if l.startswith("RESULT_JSON ")
        ]
        if proc.returncode != 0 or not lines:
            print(proc.stdout[-4000:], proc.stderr[-4000:], file=sys.stderr)
            sys.exit(f"{kernel} run failed (exit {proc.returncode})")
        results[kernel] = json.loads(lines[-1][len("RESULT_JSON ") :])

    base, bg = results["default"], results["batchgen"]
    divergence = [
        _first_divergence(a, b) for a, b in zip(base["outputs"], bg["outputs"])
    ]
    summary = {
        "mode": "e2e",
        "model": args.model,
        "identical_outputs": sum(d is None for d in divergence),
        "num_prompts": len(divergence),
        "first_divergence_token": divergence,
        "speedup_decode": bg["decode_tokens_per_s"]
        / base["decode_tokens_per_s"],
        "speedup_prefill": base["prefill_latency_s"] / bg["prefill_latency_s"],
        "runs": results,
    }
    for kernel, r in results.items():
        print(
            f"{kernel:9s} prefill {r['prefill_latency_s'] * 1e3:8.1f} ms  "
            f"decode {r['decode_tokens_per_s']:7.2f} tok/s  "
            f"e2e {r['end_to_end_tokens_per_s']:7.2f} tok/s  "
            f"batchgen calls {r['native_kernel_stats']['expert_calls']}  "
            f"fallbacks {r['native_kernel_stats']['fallback_calls']}"
        )
    print(
        f"identical greedy outputs: {summary['identical_outputs']}/"
        f"{summary['num_prompts']}  first divergence: {divergence}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    micro = sub.add_parser("micro")
    micro.add_argument(
        "--rows", type=int, nargs="+", default=[1, 4, 16, 64, 256, 2048]
    )
    micro.add_argument("--warmup", type=int, default=20)
    micro.add_argument("--iters", type=int, default=100)
    e2e = sub.add_parser("e2e")
    e2e.add_argument("--model", default="allenai/OLMoE-1B-7B-0924-Instruct")
    e2e.add_argument("--offload-dir", required=True)
    e2e.add_argument("--device-memory-ratio", type=float, default=0.3)
    e2e.add_argument("--max-new-tokens", type=int, default=32)
    e2e.add_argument("--rounds", type=int, default=2)
    e2e.add_argument("--child", choices=KERNELS, help=argparse.SUPPRESS)
    for p in (micro, e2e):
        p.add_argument("--output-json")
    args = parser.parse_args()

    if args.mode == "e2e" and args.child:
        run_child(args)
        return
    summary = run_micro(args) if args.mode == "micro" else run_e2e(args)
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
