# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Side-by-side table of cpu_diag.sh variants (S2 CPU compute medians).

python summarize_diag.py <cpu_diag OUT dir>
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def load(root: Path):
    runs = {}
    for res in sorted(root.glob("*/*/results.json")):
        prec, variant = res.parts[-3], res.parts[-2]
        try:
            runs[(prec, variant)] = json.loads(res.read_text())
        except json.JSONDecodeError:
            continue
    return runs


def onednn_impls(log: Path) -> list:
    """Aggregate ONEDNN_VERBOSE ukernel lines: per (event, shape) count,
    median and total ms -- shows whether brgemm execution itself is slow and
    whether kernels are re-created (``create``) on every call."""
    if not log.exists():
        return []
    groups = {}
    warnings = {}
    for line in log.read_text(errors="replace").splitlines():
        if not line.startswith("onednn_verbose"):
            continue
        f = line.split(",")
        if len(f) > 4 and f[3] == "warning":
            warnings[f[5] if len(f) > 5 else "?"] = (
                warnings.get(f[5] if len(f) > 5 else "?", 0) + 1
            )
            continue
        if len(f) < 8 or f[2] not in ("ukernel", "primitive"):
            continue
        try:
            ms = float(f[-1])
        except ValueError:
            ms = None
        key = (f[2], f[3], f[5], f[-2] if ms is not None else "")
        groups.setdefault(key, []).append(ms)
    out = []
    for (kind, ev, name, shape), vals in sorted(
        groups.items(), key=lambda kv: -len(kv[1])
    ):
        xs = sorted(v for v in vals if v is not None)
        med = xs[len(xs) // 2] if xs else float("nan")
        out.append(
            f"{kind} {ev} {name} {shape}: n={len(vals)} median {med:.4f} ms total {sum(xs):.1f} ms"
        )
    out += [f"warning x{n}: {w}" for w, n in warnings.items()]
    return out[:16]


def render(root: Path) -> str:
    runs = load(root)
    out = [f"# CPU diagnostics: {root}", ""]
    if not runs:
        return "\n".join(out + ["(no results.json found)"])
    for prec in sorted({p for p, _ in runs}):
        variants = [v for p, v in runs if p == prec]
        out += [f"## {prec}", ""]
        meta = [
            "| variant | threads | allowed cores | mempolicy | busy CPUs before run | weight pages by node | malloc | errors |",
            "|---|---:|---:|---|---|---|---|---:|",
        ]
        table = defaultdict(dict)
        kernels = set()
        for v in variants:
            r = runs[(prec, v)]
            hs = r["env"].get("host_setup") or {}
            busy = (hs.get("host_cpu_busy_before_run") or {}).get(
                "cpus_over_50pct"
            )
            pages = {
                k: c.get("weight_page_nodes")
                for k, c in r["setup"].get("cpu", {}).items()
            }
            mp = hs.get("mempolicy") or {}
            meta.append(
                f"| {v} | {hs.get('threads', r['env']['cpu'].get('torch_threads'))} | "
                f"{hs.get('allowed_physical_cores', '-')} | {mp.get('mode', '-')} {mp.get('nodes', '')} | "
                f"{len(busy) if busy is not None else '-'} | {pages} | "
                f"{'on' if (hs.get('malloc') or {}).get('ok') else 'off'} | {len(r.get('errors', []))} |"
            )
            for row in r["rows"]:
                if row["scenario"] != "cpu_compute":
                    continue
                kernels.add(row["kernel"])
                s = row.get("samples_ms", {}).get("compute") or []
                spread = f" [{min(s):.2f}-{max(s):.2f}]" if s else ""
                table[(row["kernel"], row["tokens"])][v] = (
                    f"{row['compute_ms']:.3f}{spread}"
                )
        out += meta + [""]
        for k in sorted(kernels):
            ms = sorted({m for kk, m in table if kk == k})
            out += [
                f"### {k}: CPU compute median ms [min-max over repeats]",
                "",
                "| M | " + " | ".join(variants) + " |",
                "|---:|" + "---:|" * len(variants),
            ]
            for m in ms:
                out.append(
                    f"| {m} | "
                    + " | ".join(table[(k, m)].get(v, "-") for v in variants)
                    + " |"
                )
            out.append("")
        impls = onednn_impls(root / f"{prec}.onednn_verbose.log")
        if impls:
            out += (
                ["oneDNN implementations seen (ONEDNN_VERBOSE):", ""]
                + [f"- `{i}`" for i in impls]
                + [""]
            )
    return "\n".join(out) + "\n"


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    root = Path(argv[0] if argv else ".")
    print(render(root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
