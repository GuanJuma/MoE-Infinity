# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Fallback path: split SGLang per-token recorder dumps into request traces.

SGLang (v0.5.19) with ``--expert-distribution-recorder-mode per_token``
dumps ``expert_distribution_recorder_<time>_<rank>.pt`` on
``POST /dump_expert_distribution_record``: one record per forward pass with
``topk_ids_of_layer`` [num_layers, num_tokens, 8], ``positions``,
``forward_mode`` (1 EXTEND, 2 DECODE, 3 MIXED, 4 IDLE) -- but no request
ids.  With one request in flight at a time (tokenhub ``concurrency 1``),
passes split into requests: an EXTEND pass after a DECODE pass, or with
``positions[0] == 0``, starts a new request; consecutive EXTEND passes
(chunked prefill) are merged.  Output files use the same layout as the
MoE-Infinity traces (``moe_infinity/hy3/routing_trace.py``).

    python benchmarks/hy3_trace/sglang_dump_to_traces.py \
        /scratch/$ME/sgl_dump/expert_distribution_recorder_*_0.pt \
        --out /scratch/$ME/hy3_traces/sglang/traces --expert-bytes 18874368
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from moe_infinity.hy3.routing_trace import (  # noqa: E402
    FORMAT,
    merge_prefill_chunks,
    save_trace,
)

EXTEND, DECODE, MIXED, IDLE = 1, 2, 3, 4


def _mode(v) -> int:
    return int(getattr(v, "value", v))


def split_requests(records):
    reqs, cur, prev = [], None, None
    for r in records:
        mode = _mode(r.get("forward_mode", EXTEND))
        ids = r["topk_ids_of_layer"]
        if mode == IDLE or ids.shape[1] == 0:
            continue
        if mode == MIXED:
            raise ValueError(
                "MIXED forward pass found: more than one request was in "
                "flight; rerun with concurrency 1"
            )
        pos = r.get("positions") or [0]
        rec = {
            "forward_mode": "extend" if mode == EXTEND else "decode",
            "topk_ids_of_layer": ids.to(torch.int32),
            "positions_start": int(pos[0]),
            "num_tokens": int(ids.shape[1]),
        }
        new = cur is None or (
            mode == EXTEND and (prev == DECODE or int(pos[0]) == 0)
        )
        if new:
            if cur:
                reqs.append(cur)
            cur = []
        cur.append(rec)
        prev = mode
    if cur:
        reqs.append(cur)
    return [merge_prefill_chunks(r) for r in reqs]


def remap(records, p2l):
    if p2l is None:
        return records
    p2l = torch.as_tensor(p2l)
    for r in records:
        ids = r["topk_ids_of_layer"]
        out = ids.clone()
        for layer in range(ids.shape[0]):
            m = ids[layer] >= 0
            if m.any():
                out[layer][m] = p2l[layer][ids[layer][m].long()].to(ids.dtype)
        r["topk_ids_of_layer"] = out
    return records


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("dumps", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--prefix", default="sglang")
    p.add_argument("--expert-bytes", type=int, default=3 * 1536 * 4096)
    p.add_argument("--first-k-dense", type=int, default=1)
    a = p.parse_args(argv)
    files = sorted({f for d in a.dumps for f in glob.glob(d)})
    out = Path(a.out)
    n = 0
    for f in files:
        dump = torch.load(f, map_location="cpu", weights_only=False)
        p2l = dump.get("last_physical_to_logical_map")
        for recs in split_requests(dump["records"]):
            recs = remap(recs, p2l)
            L, _, K = recs[0]["topk_ids_of_layer"].shape
            moe = [
                i
                for i in range(L)
                if bool((recs[0]["topk_ids_of_layer"][i] >= 0).any())
            ]
            meta = {
                "engine": "sglang per_token recorder",
                "source_dump": f,
                "rid": f"{a.prefix}_{n:05d}",
                "num_layers": L,
                "top_k": K,
                "moe_layers": moe,
                "first_k_dense_replace": a.first_k_dense,
                "expert_bytes": a.expert_bytes,
                "prefill_tokens": recs[0]["num_tokens"],
                "decode_steps": len(recs) - 1,
            }
            save_trace(
                {
                    "format": FORMAT,
                    "records": recs,
                    "meta": meta,
                    "last_physical_to_logical_map": None,
                },
                out / f"{a.prefix}_{n:05d}.pt",
            )
            n += 1
        print(f"{f}: {n} requests so far")
    print(f"wrote {n} traces to {out}")


if __name__ == "__main__":
    main()
