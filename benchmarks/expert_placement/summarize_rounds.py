# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Repeat-protocol summary for S2: per kernel and M, the median of every
round, min-of-medians and spread (max/min - 1), plus interference flags, so
time-varying noise is visible.  With ``--ab A B`` also an A/B table (e.g.
shared vs isolated cores).

Layout: <dir>/<precision>/[<leg>/]round_<i>/results.json

    python3 summarize_rounds.py OUT [--ab shared isolated] [--metric compute_ms]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def load_rounds(base: Path) -> dict:
    """{(precision, leg): {round: results}} (leg '' without A/B legs)."""
    out = {}
    for res in sorted(base.glob("**/round_*/results.json")):
        rel = res.parent.relative_to(base).parts
        m = re.fullmatch(r"round_(\d+)", rel[-1])
        if not m or len(rel) < 2:
            continue
        prec, leg = rel[0], "/".join(rel[1:-1])
        try:
            out.setdefault((prec, leg), {})[int(m.group(1))] = json.loads(
                res.read_text()
            )
        except ValueError:
            continue
    return out


def table(rounds: dict, metric: str) -> dict:
    """{kernel: {M: {round: (value, interfered)}}} for cpu_compute rows."""
    out = {}
    for r, res in rounds.items():
        for row in res.get("rows", []):
            if row.get("scenario") != "cpu_compute" or row.get(metric) is None:
                continue
            out.setdefault(row["variant"], {}).setdefault(row["tokens"], {})[
                r
            ] = (
                row[metric],
                bool(row.get("interfered")),
            )
    return out


def stats(cells: dict) -> dict:
    vals = [v for v, _ in cells.values()]
    lo, hi = min(vals), max(vals)
    return {
        "min": lo,
        "max": hi,
        "spread_pct": 100.0 * (hi / lo - 1) if lo > 0 else None,
        "flagged": sorted(r for r, (_, f) in cells.items() if f),
        "n": len(vals),
    }


def _f(x):
    if x is None:
        return "-"
    return (
        f"{x:.1f}"
        if abs(x) >= 100
        else (f"{x:.3f}" if abs(x) >= 1 else f"{x:.4f}")
    )


def _pct(x):
    return "-" if x is None else f"{x:.0f}%"


def meta_lines(rounds: dict) -> list:
    lines = [
        "| round | start | noise check (mean max % / worst window %) | flagged rows | layout |",
        "|---:|---|---|---:|---|",
    ]
    for r in sorted(rounds):
        res = rounds[r]
        nc = res.get("noise_check") or {}
        iso = (res.get("env") or {}).get("isolation") or {}
        flagged = sum(1 for row in res.get("rows", []) if row.get("interfered"))
        lines.append(
            f"| {r} | {(res.get('env') or {}).get('time', '-')} | "
            f"{nc.get('foreign_mean_max_pct', '-')} / {nc.get('foreign_max_pct', '-')} | "
            f"{flagged} | compute {iso.get('compute_cpus', '-')}, housekeeping "
            f"{iso.get('housekeeping_cpus') or '-'} |"
        )
    return lines


def render(base: Path, metric: str = "compute_ms", ab=None) -> str:
    data = load_rounds(base)
    out = [
        f"# S2 repeat protocol: {base}",
        "",
        f"metric: `{metric}` (median per round, ms)",
        "",
    ]
    summary = {}
    for (prec, leg), rounds in sorted(data.items()):
        title = f"{prec}" + (f" / {leg}" if leg else "")
        out += [f"## {title}", ""] + meta_lines(rounds) + [""]
        for kernel, by_m in sorted(table(rounds, metric).items()):
            rs = sorted({r for c in by_m.values() for r in c})
            out += [
                f"### {kernel}",
                "",
                "| M | "
                + " | ".join(f"round {r}" for r in rs)
                + " | min of medians | spread | flagged rounds |",
                "|---:|" + "---:|" * len(rs) + "---:|---:|---|",
            ]
            for M in sorted(by_m):
                cells = by_m[M]
                s = stats(cells)
                summary.setdefault((prec, kernel), {}).setdefault(leg, {})[
                    M
                ] = s
                vals = " | ".join(
                    (_f(cells[r][0]) + ("*" if cells[r][1] else ""))
                    if r in cells
                    else "-"
                    for r in rs
                )
                out.append(
                    f"| {M} | {vals} | **{_f(s['min'])}** | {_pct(s['spread_pct'])} | "
                    f"{','.join(map(str, s['flagged'])) or '-'} |"
                )
            out.append("")
    out.append(
        "`*` = row flagged for CPU interference in that round (see results.json)."
    )
    if ab:
        a, b = ab
        out += ["", f"## A/B: `{a}` vs `{b}` (min of medians, ms)", ""]
        for (prec, kernel), legs in sorted(summary.items()):
            if a not in legs or b not in legs:
                continue
            out += [
                f"### {prec} {kernel}",
                "",
                f"| M | {a} | {b} | {b}/{a} | {a} spread | {b} spread | flagged ({a} / {b}) |",
                "|---:|---:|---:|---:|---:|---:|---|",
            ]
            for M in sorted(set(legs[a]) & set(legs[b])):
                sa, sb = legs[a][M], legs[b][M]
                ratio = sb["min"] / sa["min"] if sa["min"] else None
                out.append(
                    f"| {M} | {_f(sa['min'])} | {_f(sb['min'])} | "
                    f"{'-' if ratio is None else f'{ratio:.2f}'} | "
                    f"{_pct(sa['spread_pct'])} | {_pct(sb['spread_pct'])} | "
                    f"{len(sa['flagged'])} / {len(sb['flagged'])} |"
                )
            out.append("")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("dir")
    p.add_argument(
        "--metric", default="compute_ms", choices=("compute_ms", "total_ms")
    )
    p.add_argument("--ab", nargs=2, metavar=("A", "B"))
    a = p.parse_args(argv)
    md = render(Path(a.dir), a.metric, a.ab)
    print(md, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
