# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Per-request routing traces (top-k expert ids of every MoE layer).

One ``.pt`` file per request, in the layout of SGLang's per-token expert
distribution recorder (``--expert-distribution-recorder-mode per_token``)
that archer-rel-simulation's ``sequence_belady_cache_simulator.py`` reads::

    {
      "records": [                       # records[0] = whole prefill
        {"forward_mode": "extend", "topk_ids_of_layer": int32 [L, T, K],
         "positions_start": 0, "num_tokens": T},
        {"forward_mode": "decode", "topk_ids_of_layer": int32 [L, 1, K], ...},
        ...
      ],
      "meta": {...},                     # model / request description
      "last_physical_to_logical_map": None,
    }

``L`` is ``num_hidden_layers`` (80 for Hy3); rows of non-MoE layers (Hy3
layer 0) are ``-1``, like SGLang's recorder.  Chunked prefill passes are
merged into one ``extend`` record.  Optional ``topk_weights_of_layer``
(float16, same shape) holds the routing weights.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

FORMAT = "moe_infinity.hy3.routing_trace/v1"


class RoutingTraceRecorder:
    def __init__(
        self, num_layers: int, top_k: int, record_weights: bool = False
    ):
        self.num_layers = num_layers
        self.top_k = top_k
        self.record_weights = record_weights
        self._request: Optional[Dict[str, Any]] = None
        self._pass: Optional[Dict[str, Any]] = None

    @property
    def active(self) -> bool:
        return self._pass is not None

    def begin_request(self, rid: str, meta: Optional[dict] = None) -> None:
        self._request = {"rid": rid, "meta": dict(meta or {}), "records": []}

    def begin_pass(self, mode: str, num_tokens: int, start_pos: int) -> None:
        if self._request is None:
            return
        self._pass = {
            "mode": mode,
            "T": num_tokens,
            "start": start_pos,
            "ids": {},
            "w": {},
        }

    def on_route(
        self,
        layer: int,
        topk_ids: torch.Tensor,
        topk_weights: Optional[torch.Tensor] = None,
    ) -> None:
        p = self._pass
        if p is None:
            return
        p["ids"][layer] = topk_ids.detach()
        if self.record_weights and topk_weights is not None:
            p["w"][layer] = topk_weights.detach()

    def end_pass(self) -> None:
        p, self._pass = self._pass, None
        if p is None or self._request is None:
            return
        T, K, L = p["T"], self.top_k, self.num_layers
        dev = next(iter(p["ids"].values())).device if p["ids"] else "cpu"
        ids = torch.full((L, T, K), -1, dtype=torch.int32, device=dev)
        for layer, t in p["ids"].items():
            ids[layer] = t.reshape(T, K).to(torch.int32)
        rec = {
            "forward_mode": p["mode"],
            "topk_ids_of_layer": ids.cpu(),
            "positions_start": p["start"],
            "num_tokens": T,
        }
        if self.record_weights:
            w = torch.zeros((L, T, K), dtype=torch.float16, device=dev)
            for layer, t in p["w"].items():
                w[layer] = t.reshape(T, K).to(torch.float16)
            rec["topk_weights_of_layer"] = w.cpu()
        self._request["records"].append(rec)

    def end_request(self, extra_meta: Optional[dict] = None) -> dict:
        req, self._request = self._request, None
        self._pass = None
        if req is None:
            raise RuntimeError("end_request without begin_request")
        records = merge_prefill_chunks(req["records"])
        meta = dict(req["meta"])
        meta.update(extra_meta or {})
        meta.update(
            rid=req["rid"],
            num_layers=self.num_layers,
            top_k=self.top_k,
            prefill_tokens=records[0]["num_tokens"] if records else 0,
            decode_steps=max(len(records) - 1, 0),
        )
        return {
            "format": FORMAT,
            "records": records,
            "meta": meta,
            "last_physical_to_logical_map": None,
        }


def merge_prefill_chunks(records: List[dict]) -> List[dict]:
    out: List[dict] = []
    for r in records:
        prev = out[-1] if out else None
        if (
            prev is not None
            and r["forward_mode"] == "extend"
            and prev["forward_mode"] == "extend"
        ):
            merged = dict(prev)
            for k in ("topk_ids_of_layer", "topk_weights_of_layer"):
                if k in prev and k in r:
                    merged[k] = torch.cat([prev[k], r[k]], dim=1)
            merged["num_tokens"] = prev["num_tokens"] + r["num_tokens"]
            out[-1] = merged
        else:
            out.append(r)
    return out


def save_trace(trace: dict, path: str | os.PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(trace, tmp)
    os.replace(tmp, path)
    return path


def load_trace(path: str | os.PathLike) -> dict:
    trace = torch.load(path, map_location="cpu", weights_only=False)
    if "records" not in trace:
        raise ValueError(f"{path}: not a routing trace (no 'records')")
    return trace


def moe_layer_ids(trace: dict) -> List[int]:
    """Layers that carry routing (rows not all -1 in the first record)."""
    ids = trace["records"][0]["topk_ids_of_layer"]
    return [i for i in range(ids.shape[0]) if bool((ids[i] >= 0).any())]
