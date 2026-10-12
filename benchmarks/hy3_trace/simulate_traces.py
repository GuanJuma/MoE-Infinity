# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Run archer-rel-simulation's expert-cache simulators on Hy3 traces.

Uses the simulator's own code from the unpacked ``archer-rel-simulation.tar``
(``--sim-dir .../simulation``):

* ``sequence_belady_cache_simulator.py`` -- ``extract_access_sequence`` +
  ``belady_simulate`` (Belady-optimal hit rate over the token-level
  ``(layer, expert)`` access stream, decode only unless
  ``--include-prefill``);
* ``cache_strategy.py`` (``--cache-strategy``) -- ``AccessSequence`` +
  ``HeapExpertCache``: Belady over per-request prefill (unique experts per
  layer) + decode accesses, reporting overall and decode hit rates.

LRU and LFU on the same access stream are added as online baselines.
``--scope per_request`` simulates every trace file on its own (what the
simulator's CLI prints); ``global`` concatenates all requests in file order
(one cache shared across requests, like a serving process).

    python benchmarks/hy3_trace/simulate_traces.py \
        --traces /scratch/$ME/hy3_traces/run1/traces \
        --sim-dir /scratch/$ME/archer-rel-simulation/simulation \
        --out /scratch/$ME/hy3_traces/run1/sim
"""

from __future__ import annotations

import argparse
import csv
import heapq
import importlib
import json
import math
import sys
from collections import Counter, OrderedDict, defaultdict, deque
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

Key = Tuple[int, int]
# 32 GiB = the engine's default GPU expert cache (1,820 Hy3 experts).
DEFAULT_CAPACITY_GIB = (4, 8, 16, 32, 64, 128)


def find_traces(paths: Sequence[str]) -> List[Path]:
    out = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            out.extend(sorted(p.rglob("*.pt")))
        elif p.exists():
            out.append(p)
    return out


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


# -- fallbacks identical to sequence_belady_cache_simulator.py ------------


def _extract(records, *, ignore_layers: int, include_prefill: bool):
    acc: List[Key] = []
    for rec in records[0 if include_prefill else 1 :]:
        t = rec["topk_ids_of_layer"]
        L = t.shape[0]
        flat = t.reshape(L, -1)
        for layer in range(ignore_layers, L):
            for e in flat[layer].tolist():
                if e >= 0:
                    acc.append((layer, int(e)))
    return acc


def _belady(accesses: Sequence[Key], capacity: int):
    total = len(accesses)
    if total == 0:
        return 0, 0, 0.0
    if capacity <= 0:
        return 0, total, 0.0
    future: Dict[Key, deque] = defaultdict(deque)
    for i in range(total - 1, -1, -1):
        future[accesses[i]].appendleft(i)
    nxt: Dict[Key, float] = {}
    heap: list = []
    hits = 0
    for key in accesses:
        pos = future[key]
        pos.popleft()
        nu = pos[0] if pos else math.inf
        if key in nxt:
            hits += 1
        elif len(nxt) >= capacity:
            while heap:
                neg, k = heapq.heappop(heap)
                if nxt.get(k) == -neg:
                    del nxt[k]
                    break
        nxt[key] = nu
        heapq.heappush(heap, (-nu, key))
    return hits, total - hits, hits / total


def lru(accesses: Sequence[Key], capacity: int):
    cache: "OrderedDict[Key, None]" = OrderedDict()
    hits = 0
    for k in accesses:
        if k in cache:
            hits += 1
            cache.move_to_end(k)
        else:
            if len(cache) >= capacity:
                cache.popitem(last=False)
            cache[k] = None
    n = len(accesses)
    return hits, n - hits, hits / n if n else 0.0


def lfu(accesses: Sequence[Key], capacity: int):
    freq: Counter = Counter()
    cache: Dict[Key, int] = {}
    heap: list = []
    hits = 0
    for t, k in enumerate(accesses):
        freq[k] += 1
        if k in cache:
            hits += 1
        elif len(cache) >= capacity:
            while heap:
                f, ts, v = heapq.heappop(heap)
                if v in cache and cache[v] == ts:
                    del cache[v]
                    break
        cache[k] = t
        heapq.heappush(heap, (freq[k], t, k))
    n = len(accesses)
    return hits, n - hits, hits / n if n else 0.0


def load_sim(sim_dir: str | None):
    if not sim_dir:
        return None, "builtin (no --sim-dir)"
    sys.path.insert(0, str(Path(sim_dir).resolve()))
    try:
        mod = importlib.import_module("sequence_belady_cache_simulator")
        return mod, f"archer-rel-simulation: {mod.__file__}"
    except Exception as e:  # noqa: BLE001
        return None, f"builtin (import failed: {e!r})"


def cache_strategy_hit(
    sim_dir: str, traces: List[dict], layers: List[int], capacity: int
):
    cs = importlib.import_module("cache_strategy")
    pre, dec = {}, {}
    for i, tr in enumerate(traces):
        recs = tr["records"]
        ids0 = recs[0]["topk_ids_of_layer"][layers].numpy()
        pre[i] = ids0
        dec[i] = [r["topk_ids_of_layer"][layers].numpy() for r in recs[1:]]
    seq = cs.AccessSequence(pre, dec, None)
    cache = cs.HeapExpertCache(capacity)
    cache.set_access_sequence(seq.get_access_sequence())
    while True:
        req = seq.get_next_request()
        if req is None:
            break
        for k in req.prefillAccessSequence:
            cache.access(k, value=1)
        for k in req.decodeAccessSequence:
            cache.access(k, value=1, decode=True)
    return cache.get_hit_rate(), cache.get_decode_hit_rate()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--traces", nargs="+", required=True)
    p.add_argument(
        "--sim-dir",
        default=None,
        help="archer-rel-simulation/simulation directory",
    )
    p.add_argument(
        "--capacities",
        nargs="*",
        type=int,
        default=[],
        help="cache sizes in experts",
    )
    p.add_argument(
        "--capacity-gib",
        nargs="*",
        type=float,
        default=[],
        help="cache sizes in GiB (expert size from trace meta); default "
        "4 8 16 32 64 128 when no size is given",
    )
    p.add_argument(
        "--ignore-layers",
        type=int,
        default=None,
        help="leading non-MoE layers (default: from trace, 1)",
    )
    p.add_argument("--include-prefill", action="store_true")
    p.add_argument(
        "--policies",
        nargs="+",
        default=["belady", "lru", "lfu"],
        choices=["belady", "lru", "lfu"],
    )
    p.add_argument(
        "--scope",
        nargs="+",
        default=["global", "per_request"],
        choices=["global", "per_request"],
    )
    p.add_argument(
        "--cache-strategy",
        action="store_true",
        help="also run cache_strategy.py's HeapExpertCache",
    )
    p.add_argument("--max-files", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    files = find_traces(a.traces)
    if a.max_files:
        files = files[: a.max_files]
    if not files:
        raise SystemExit("no trace files found")
    traces = [load(f) for f in files]
    meta = traces[0].get("meta", {})
    ignore = (
        a.ignore_layers
        if a.ignore_layers is not None
        else int(meta.get("first_k_dense_replace", 1))
    )
    expert_bytes = int(meta.get("expert_bytes", 18 * 2**20))
    caps = set(a.capacities)
    for g in a.capacity_gib:
        caps.add(int(g * 2**30 // expert_bytes))
    if not caps:
        caps = {int(g * 2**30 // expert_bytes) for g in DEFAULT_CAPACITY_GIB}
    caps = sorted(caps)
    sim, sim_src = load_sim(a.sim_dir)
    extract = sim.extract_access_sequence if sim else _extract
    belady = sim.belady_simulate if sim else _belady
    fn = {"belady": belady, "lru": lru, "lfu": lfu}

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    per_file = [
        extract(
            t["records"],
            ignore_layers=ignore,
            include_prefill=a.include_prefill,
        )
        for t in traces
    ]
    rows = []
    n_moe = len(meta.get("moe_layers") or []) or None
    n_exp = int(meta.get("num_experts", 0)) or None
    total_experts = n_moe * n_exp if n_moe and n_exp else None
    for cap in caps:
        for pol in a.policies:
            if "global" in a.scope:
                acc = [k for f in per_file for k in f]
                h, m, r = fn[pol](acc, cap)
                rows.append(
                    dict(
                        scope="global",
                        policy=pol,
                        capacity=cap,
                        capacity_gib=cap * expert_bytes / 2**30,
                        hits=h,
                        misses=m,
                        hit_rate=r,
                        files=len(files),
                    )
                )
            if "per_request" in a.scope:
                rates, H, M = [], 0, 0
                for acc in per_file:
                    if not acc:
                        continue
                    h, m, r = fn[pol](acc, cap)
                    rates.append(r)
                    H, M = H + h, M + m
                rows.append(
                    dict(
                        scope="per_request",
                        policy=pol,
                        capacity=cap,
                        capacity_gib=cap * expert_bytes / 2**30,
                        hits=H,
                        misses=M,
                        hit_rate=sum(rates) / len(rates) if rates else 0.0,
                        files=len(rates),
                    )
                )
            print(
                f"[sim] cap {cap} {pol}: "
                + ", ".join(
                    f"{r['scope']} {r['hit_rate']:.4f}"
                    for r in rows[-len(a.scope) :]
                ),
                flush=True,
            )
        if a.cache_strategy and a.sim_dir:
            layers = list(
                meta.get("moe_layers")
                or range(
                    ignore,
                    traces[0]["records"][0]["topk_ids_of_layer"].shape[0],
                )
            )
            try:
                hr, dhr = cache_strategy_hit(a.sim_dir, traces, layers, cap)
                rows.append(
                    dict(
                        scope="global",
                        policy="cache_strategy_belady",
                        capacity=cap,
                        capacity_gib=cap * expert_bytes / 2**30,
                        hits=None,
                        misses=None,
                        hit_rate=hr,
                        decode_hit_rate=dhr,
                        files=len(files),
                    )
                )
                print(
                    f"[sim] cap {cap} cache_strategy: {hr:.4f} "
                    f"decode {dhr:.4f}",
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001
                print(f"[sim] cache_strategy failed: {e!r}", flush=True)

    keys = [
        "scope",
        "policy",
        "capacity",
        "capacity_gib",
        "hits",
        "misses",
        "hit_rate",
        "decode_hit_rate",
        "files",
    ]
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    tokens = sum(t["meta"].get("prefill_tokens", 0) for t in traces)
    steps = sum(t["meta"].get("decode_steps", 0) for t in traces)
    lines = [
        "# Hy3 expert-cache simulation",
        "",
        f"- traces: {len(files)} files from {', '.join(a.traces)}",
        f"- prefill tokens {tokens}, decode steps {steps}",
        f"- simulator: {sim_src}",
        f"- accesses: {'prefill + decode' if a.include_prefill else 'decode only'}"
        f", ignore_layers={ignore}",
        f"- expert size {expert_bytes / 2**20:.1f} MiB; routed experts "
        f"{total_experts}",
        "",
        "| scope | policy | capacity (experts) | GiB | % of experts | hit rate |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        frac = (
            f"{100 * r['capacity'] / total_experts:.1f}"
            if total_experts
            else ""
        )
        hr = f"{r['hit_rate']:.4f}"
        if r.get("decode_hit_rate") is not None:
            hr += f" (decode {r['decode_hit_rate']:.4f})"
        lines.append(
            f"| {r['scope']} | {r['policy']} | {r['capacity']} | "
            f"{r['capacity_gib']:.1f} | {frac} | {hr} |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    with open(out / "sim_commands.sh", "w") as f:
        f.write("# The simulator's own CLI on the same files (per file):\n")
        for cap in caps:
            f.write(
                f"python {Path(a.sim_dir or '.') / 'sequence_belady_cache_simulator.py'}"
                f" -f {' '.join(str(x) for x in files[:3])}"
                f"{' ...' if len(files) > 3 else ''} -c {cap}"
                f" --ignore-layers {ignore}"
                f"{' --include-prefill' if a.include_prefill else ''}\n"
            )
    print((out / "summary.md").read_text())
    json.dump(rows, open(out / "results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
