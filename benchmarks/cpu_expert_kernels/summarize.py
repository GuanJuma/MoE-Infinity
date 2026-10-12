# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Markdown tables from bench_cpu_moe.py / bench_cpu_attention.py JSON.

python summarize.py moe results.json [--baseline torch_bf16]
python summarize.py attn results.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict


def _fmt_err(err):
    return "-" if err is None else f"{err * 100:.2f}%"


def moe_tables(data, baseline):
    rows = data["results"]
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    out = []
    for model, recs in by_model.items():
        backends = sorted({r["backend"] for r in recs}, key=_backend_order)
        ms = sorted({r["M"] for r in recs})
        r0 = recs[0]
        out.append(
            f"\n**{model}** (K={r0['K']}, N={r0['N']}, top-{r0['topk']}, "
            f"{r0['E_resident']}/{r0['E_total']} experts resident)\n"
        )
        out.append(
            "| M | "
            + " | ".join(f"{b} ms (GB/s)" for b in backends)
            + f" | sglang_bf16 vs {baseline} | best vs {baseline} |"
        )
        out.append("|" + "---|" * (len(backends) + 3))
        for m in ms:
            cells = []
            cell_ms = {}
            for b in backends:
                rec = next(
                    (r for r in recs if r["M"] == m and r["backend"] == b),
                    None,
                )
                if rec is None:
                    cells.append("-")
                    continue
                cell_ms[b] = rec["ms"]
                cells.append(f"{rec['ms']:.2f} ({rec['weight_gbps']:.0f})")
            base = cell_ms.get(baseline)
            sgl = {b: t for b, t in cell_ms.items() if b.startswith("sglang")}
            if base and sgl:
                best = min(sgl, key=sgl.get)
                speed = f"{base / sgl[best]:.2f}x ({best})"
            else:
                speed = "-"
            bf16 = (
                f"{base / cell_ms['sglang_bf16']:.2f}x"
                if base and "sglang_bf16" in cell_ms
                else "-"
            )
            out.append(
                f"| {m} | " + " | ".join(cells) + f" | {bf16} | {speed} |"
            )
        errs = defaultdict(list)
        for r in recs:
            if r["rel_err_vs_fp32"] is not None:
                errs[r["backend"]].append(r["rel_err_vs_fp32"])
        if errs:
            out.append(
                "\nmax rel. error vs FP32: "
                + ", ".join(
                    f"{b} {_fmt_err(max(v))}"
                    for b, v in sorted(
                        errs.items(), key=lambda kv: _backend_order(kv[0])
                    )
                )
            )
    return "\n".join(out)


def _backend_order(b):
    order = [
        "sglang_bf16",
        "sglang_int8",
        "sglang_fp8",
        "torch_bf16",
        "moegen_avx2",
    ]
    return order.index(b) if b in order else len(order)


def attn_tables(data):
    rows = data["results"]
    keys = sorted({(r["model"], r["cache_len"]) for r in rows})
    out = [
        "| model | L | B | MoE-Gen AVX2 ms (err) | SGLang ms (err) | "
        "SGLang speedup |",
        "|---|---|---|---|---|---|",
    ]
    for model, L in keys:
        for B in sorted(
            {
                r["batch"]
                for r in rows
                if (r["model"], r["cache_len"]) == (model, L)
            }
        ):
            sel = {
                r["backend"]: r
                for r in rows
                if r["model"] == model
                and r["cache_len"] == L
                and r["batch"] == B
            }
            mg, sg = sel.get("moegen_avx2_omp"), sel.get("sglang_decode")
            if not (mg and sg):
                continue
            out.append(
                f"| {model} | {L} | {B} | {mg['ms']:.3f} "
                f"({_fmt_err(mg['rel_err_vs_fp32'])}) | {sg['ms']:.3f} "
                f"({_fmt_err(sg['rel_err_vs_fp32'])}) | "
                f"{mg['ms'] / sg['ms']:.1f}x |"
            )
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["moe", "attn"])
    ap.add_argument("path")
    ap.add_argument("--baseline", default="torch_bf16")
    args = ap.parse_args()
    with open(args.path) as f:
        data = json.load(f)
    if args.kind == "moe":
        print(moe_tables(data, args.baseline))
    else:
        print(attn_tables(data))


if __name__ == "__main__":
    main()
