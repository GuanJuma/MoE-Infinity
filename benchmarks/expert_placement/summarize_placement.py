# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Summarize expert_placement_bench.py results: Markdown tables + plots.

    python summarize_placement.py RESULTS_DIR_OR_JSON [--md summary.md] [--plot-dir DIR]

Tables: total latency per token count for every scenario/variant (best one
in bold), the compute/overhead split of each, and the crossover token counts
between placements.  Plots (needs matplotlib): total vs tokens, stacked
compute/overhead per scenario, overhead share.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ORDER = ("gpu_resident", "cpu_compute", "cpu_store_gpu_compute")
LABEL = {
    "gpu_resident": "S1 GPU-resident",
    "cpu_compute": "S2 CPU compute",
    "cpu_store_gpu_compute": "S3 CPU-stored->GPU",
}


def load(path: str) -> dict:
    p = Path(path)
    if p.is_dir():
        p = p / "results.json"
    with open(p) as f:
        return json.load(f)


def series(result):
    """(scenario, variant) -> {tokens: row}."""
    out = defaultdict(dict)
    for r in result["rows"]:
        out[(r["scenario"], r["variant"])][r["tokens"]] = r
    return dict(
        sorted(out.items(), key=lambda kv: (ORDER.index(kv[0][0]), kv[0][1]))
    )


def _f(x, nd=3):
    if x is None:
        return "-"
    if abs(x) >= 100:
        return f"{x:.1f}"
    if abs(x) >= 1:
        return f"{x:.{nd - 1}f}"
    return f"{x:.{nd + 1}f}"


def header(result) -> list:
    e, env = result["expert"], result["env"]
    gpu = env.get("gpu", {})
    cpu = env.get("cpu", {})
    setup = result.get("setup", {})
    lines = [
        f"# Expert placement: layer {e['layer']} expert {e['expert']} "
        f"(precision {result.get('precision', 'fp8')})",
        "",
        f"- expert: K={e['hidden_size']} N={e['intermediate_size']} {e['quant']}, "
        f"{e['checkpoint_bytes'] / 2**20:.1f} MiB per copy; source: "
        f"{setup.get('expert_source', e.get('source', 'checkpoint'))}",
        f"- GPU: {gpu.get('name', '-')} {gpu.get('capability', '')}; "
        f"torch {env['versions'].get('torch')} CUDA {env['versions'].get('torch_cuda')}; "
        f"sglang {env['versions'].get('sglang')}; triton {env['versions'].get('triton')}",
        f"- CPU threads: {cpu.get('torch_threads')} (allowed CPUs {cpu.get('affinity_cpus')}, "
        f"{cpu.get('Cpus_allowed_list', '-')}; cpuset mems {cpu.get('Mems_allowed_list', '-')}; "
        f"mempolicy {_mempol(env)}); "
        f"ONEDNN_MAX_CPU_ISA={env.get('env', {}).get('ONEDNN_MAX_CPU_ISA', '-')}; "
        f"amx_selfcheck={_amx(env.get('amx_selfcheck'))}"
        + (
            " (CPU has no AMX: oneDNN AVX512-BF16 path)"
            if isinstance(cpu.get("isa"), dict)
            and not cpu["isa"].get("amx_bf16")
            else ""
        ),
        f"- flush: {result.get('flush')}; x_std={result.get('x_std'):.4g}",
    ]
    for kind in ("gpu", "cpu"):
        for k, meta in result.get("kernels", {}).get(kind, {}).items():
            lines.append(
                f"- {kind} kernel `{k}`: {meta.get('format')}"
                + (
                    f", activations {meta['act_quant']}"
                    if meta.get("act_quant")
                    else ""
                )
            )
    ms = setup.get("mi_store")
    if ms:
        lines.append(
            f"- MoE-Infinity store ({ms['lib']}): {len(ms['nodes'])} expert nodes in the "
            f"pinned host pool; offload {ms['offload_ms']:.0f} ms, topology init "
            f"{ms['topology_init_ms']:.0f} ms"
        )
    for p in result.get("pcie_probe", []) or []:
        lines.append(
            f"- PCIe {p['kind']} {p['bytes'] / 2**20:.0f} MiB: {p['GBps']:.1f} GB/s"
        )
    for w in result.get("warnings", []):
        lines.append(f"- **warning**: {w}")
    for err in result.get("errors", []):
        lines.append(f"- **error** in {err['stage']}: {err['error'][:200]}")
    return lines


def _mempol(env):
    mp = (env.get("host_setup") or {}).get("mempolicy")
    if mp:
        return f"{mp.get('mode')} {mp.get('nodes')}"
    show = env.get("numactl_show") or ""
    for line in show.splitlines():
        if line.startswith("membind:"):
            return "bind " + line.split(":", 1)[1].strip()
    return "-"


def _amx(v):
    if isinstance(v, dict):
        return (
            "ok"
            if v.get("ok")
            else f"FAILED (max_rel_err {v.get('max_rel_err'):.3g})"
        )
    return v or "-"


def total_table(result, curves) -> list:
    keys = list(curves)
    tokens = sorted({t for s in curves.values() for t in s})
    cols = [f"{LABEL[s]} `{v}`" for s, v in keys]
    lines = [
        "## Total latency (ms, median)",
        "",
        "| tokens | " + " | ".join(cols) + " | best |",
        "|---:|" + "---:|" * len(cols) + "---|",
    ]
    for t in tokens:
        vals = [curves[k].get(t, {}).get("total_ms") for k in keys]
        present = [(v, i) for i, v in enumerate(vals) if v is not None]
        best = min(present)[1] if present else None
        cells = []
        for i, v in enumerate(vals):
            c = _f(v)
            cells.append(f"**{c}**" if i == best else c)
        lines.append(
            f"| {t} | "
            + " | ".join(cells)
            + f" | {cols[best] if best is not None else '-'} |"
        )
    return lines


def split_tables(result, curves) -> list:
    lines = []
    for (s, v), pts in curves.items():
        lines += ["", f"## {LABEL[s]} `{v}`: compute vs overhead (ms)", ""]
        if s == "gpu_resident":
            hdr = "| tokens | total | compute (kernel) | overhead (launch+sync) | overhead % | TFLOPS | p10-p90 total | err |"
        elif s == "cpu_compute":
            hdr = "| tokens | total | compute (CPU) | of which act quant | overhead | = d2h + h2d | overhead % | TFLOPS | p10-p90 total | err | err vs W8A8 ref |"
        else:
            hdr = "| tokens | total | xfer+kernel (serial) | compute (kernel) | overhead | = xfer + release + launch | xfer GB/s | overhead % | pipelined max(xfer,compute) | p10-p90 total | err | err vs W8A8 ref |"
        lines += [hdr, "|" + "---:|" * (hdr.count("|") - 1)]
        for t in sorted(pts):
            r = pts[t]
            share = (
                100 * r["overhead_ms"] / r["total_ms"]
                if r["total_ms"]
                else None
            )
            rng = f"{_f(r['total_p10_ms'])}-{_f(r['total_p90_ms'])}"
            err = f"{r['rel_err']:.1e}" if r.get("rel_err") is not None else "-"
            err8 = (
                f"{r['rel_err_w8a8']:.1e}"
                if r.get("rel_err_w8a8") is not None
                else "-"
            )
            if s == "gpu_resident":
                cells = [
                    _f(r["total_ms"]),
                    _f(r["compute_ms"]),
                    _f(r["overhead_ms"]),
                    f"{share:.0f}%",
                    _f(r.get("compute_TFLOPS"), 2),
                    rng,
                    err,
                ]
            elif s == "cpu_compute":
                cells = [
                    _f(r["total_ms"]),
                    _f(r["compute_ms"]),
                    _f(r.get("act_quant_ms")),
                    _f(r["overhead_ms"]),
                    f"{_f(r['d2h_ms'])} + {_f(r['h2d_ms'])}",
                    f"{share:.0f}%",
                    _f(r.get("compute_TFLOPS"), 2),
                    rng,
                    err,
                    err8,
                ]
            else:
                cells = [
                    _f(r["total_ms"]),
                    _f(r.get("xfer_plus_kernel_ms")),
                    _f(r["compute_ms"]),
                    _f(r["overhead_ms"]),
                    f"{_f(r['xfer_ms'])} + {_f(r.get('release_ms') or 0)} + {_f(r['launch_ms'])}",
                    _f(r.get("xfer_GBps"), 2),
                    f"{share:.0f}%",
                    _f(r.get("pipelined_ms")),
                    rng,
                    err,
                    err8,
                ]
            lines.append(f"| {t} | " + " | ".join(cells) + " |")
    return lines


def best_per_scenario(curves):
    best = defaultdict(dict)
    for (s, v), pts in curves.items():
        for t, r in pts.items():
            cur = best[s].get(t)
            if cur is None or r["total_ms"] < cur[0]:
                best[s][t] = (r["total_ms"], v)
    return best


def crossovers(curves) -> list:
    best = best_per_scenario(curves)
    lines = ["", "## Crossovers (best variant of each scenario, by total)", ""]
    pairs = [
        ("cpu_compute", "cpu_store_gpu_compute"),
        ("cpu_compute", "gpu_resident"),
        ("cpu_store_gpu_compute", "gpu_resident"),
    ]
    for a, b in pairs:
        if a not in best or b not in best:
            continue
        ts = sorted(set(best[a]) & set(best[b]))
        wins = [(t, best[a][t][0] <= best[b][t][0]) for t in ts]
        if not wins:
            continue
        flips = [t for (t, w), (_, w0) in zip(wins[1:], wins[:-1]) if w != w0]
        a_range = [t for t, w in wins if w]
        desc = (
            f"{LABEL[a]} faster at tokens {a_range}"
            if a_range
            else f"{LABEL[a]} never faster"
        )
        lines.append(
            f"- {LABEL[a]} vs {LABEL[b]}: {desc}; switches at {flips or 'none'}"
        )
    return lines


S3_NOTE = (
    "S3 note: compare placements on `xfer+kernel`. `raw_pinned` copies asynchronously, "
    "so its `total` hides the eager call's Python dispatch under the DMA; MoE-Infinity's "
    "`begin()`/prefetch wait (and `raw_pageable` copies synchronously), so their `total` "
    "also contains that dispatch (~S1 overhead)."
)


def mi_vs_raw(curves) -> list:
    """MoE-Infinity load path vs the raw cudaMemcpy reference, per kernel."""
    lines = []
    for (s, v), pts in curves.items():
        if s != "cpu_store_gpu_compute" or "+mi_" not in v:
            continue
        kernel = v.split("+")[0]
        raw = curves.get((s, f"{kernel}+raw_pinned"))
        if not raw:
            continue
        common = sorted(set(pts) & set(raw))
        if not common:
            continue
        mi = [pts[t]["xfer_ms"] for t in common]
        rw = [raw[t]["xfer_ms"] for t in common]
        med = sorted(a - b for a, b in zip(mi, rw))[len(common) // 2]
        lines.append(
            f"- `{v}`: MoE-Infinity move {_f(sorted(mi)[len(mi) // 2])} ms vs raw pinned "
            f"copy {_f(sorted(rw)[len(rw) // 2])} ms -> bookkeeping/sync overhead about "
            f"{_f(med)} ms (median over token counts)"
        )
    if lines:
        lines = [
            "",
            "## MoE-Infinity load path vs raw cudaMemcpy (reference)",
            "",
            S3_NOTE,
            "",
        ] + lines
    return lines


def parity_lines(result) -> list:
    pairs = (result.get("parity") or {}).get("pairs") or {}
    if not pairs:
        return []
    out = ["", "## Kernel parity on identical inputs (relative difference)", ""]
    for pair, vals in pairs.items():
        out.append(
            f"- {pair}: "
            + ", ".join(
                f"M={m} {v:.1e}"
                for m, v in sorted(vals.items(), key=lambda kv: int(kv[0]))
            )
        )
    return out


def plots(result, curves, out_dir: Path) -> list:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ["(matplotlib not installed: pip install matplotlib for plots)"]
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    e = result["expert"]
    title = f"L{e['layer']} E{e['expert']} K={e['hidden_size']} N={e['intermediate_size']} {e['quant']}"

    fig, ax = plt.subplots(figsize=(9, 6))
    for (s, v), pts in curves.items():
        ts = sorted(pts)
        ax.plot(
            ts,
            [pts[t]["total_ms"] for t in ts],
            marker="o",
            label=f"{LABEL[s]} {v}",
        )
        ax.plot(
            ts,
            [pts[t]["compute_ms"] for t in ts],
            ls=":",
            alpha=0.6,
            color=ax.lines[-1].get_color(),
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("tokens routed to the expert")
    ax.set_ylabel("ms (solid = total, dotted = compute)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    p = out_dir / "total_vs_tokens.png"
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    made.append(p)

    n = len(curves)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.6 * n), squeeze=False)
    for ax, ((s, v), pts) in zip(axes[:, 0], curves.items()):
        ts = sorted(pts)
        xs = range(len(ts))
        comp = [pts[t]["compute_ms"] for t in ts]
        over = [max(pts[t]["overhead_ms"], 0) for t in ts]
        ax.bar(xs, comp, label="compute")
        ax.bar(xs, over, bottom=comp, label="overhead")
        ax.set_yscale("log")
        ax.set_xticks(list(xs), [str(t) for t in ts], fontsize=7)
        ax.set_title(f"{LABEL[s]} {v}", fontsize=9)
        ax.set_ylabel("ms")
        ax.legend(fontsize=7)
    fig.tight_layout()
    p = out_dir / "compute_vs_overhead.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    made.append(p)

    fig, ax = plt.subplots(figsize=(9, 5))
    for (s, v), pts in curves.items():
        ts = sorted(pts)
        ax.plot(
            ts,
            [100 * pts[t]["overhead_ms"] / pts[t]["total_ms"] for t in ts],
            marker=".",
            label=f"{LABEL[s]} {v}",
        )
    ax.set_xscale("log", base=2)
    ax.set_ylabel("overhead share of total (%)")
    ax.set_xlabel("tokens")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    p = out_dir / "overhead_share.png"
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    made.append(p)
    return [f"![{p.stem}]({p.name})" for p in made]


def render(result, plot_dir=None) -> str:
    curves = series(result)
    lines = (
        header(result)
        + [""]
        + total_table(result, curves)
        + split_tables(result, curves)
        + crossovers(curves)
        + mi_vs_raw(curves)
        + parity_lines(result)
    )
    if plot_dir is not None:
        lines += ["", "## Plots", ""] + plots(result, curves, Path(plot_dir))
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("results", help="results dir or results.json")
    p.add_argument(
        "--md",
        default=None,
        help="write Markdown here (default: <dir>/summary.md)",
    )
    p.add_argument(
        "--plot-dir",
        default=None,
        help="write PNGs here (default: results dir)",
    )
    p.add_argument("--no-plots", action="store_true")
    a = p.parse_args(argv)
    result = load(a.results)
    base = (
        Path(a.results) if Path(a.results).is_dir() else Path(a.results).parent
    )
    plot_dir = None if a.no_plots else Path(a.plot_dir or base)
    md = render(result, plot_dir)
    out = Path(a.md or base / "summary.md")
    out.write_text(md)
    print(md)
    print(f"-> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
