# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Export Hy3 traces to archer-rel-simulation's pickle trace format.

``cache_strategy.py`` (``MODEL_CONFIG[...]["prefilling_trace"/
"decoding_trace"]``) and the Archer-era scripts (``expert_caching.py``,
``prefetch_performance.py``) read two pickled dicts keyed by request:

* prefill: ``{key: int16 ndarray [n_moe_layers, prompt_tokens, top_k]}``
* decode:  ``{key: int16 ndarray [decode_steps, n_moe_layers, 1, top_k]}``

Only MoE layers are kept, so layer index 0 is Hy3 layer 1.

    python benchmarks/hy3_trace/export_legacy_pickles.py \
        --traces /scratch/$ME/hy3_traces/run1/traces \
        --out /scratch/$ME/hy3_traces/run1/legacy --name hy3_tokenhub
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--traces", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="hy3")
    a = p.parse_args(argv)
    files = []
    for t in a.traces:
        t = Path(t)
        files.extend(sorted(t.rglob("*.pt")) if t.is_dir() else [t])
    pre, dec = {}, {}
    for f in files:
        tr = torch.load(f, map_location="cpu", weights_only=False)
        recs = tr["records"]
        layers = tr["meta"].get("moe_layers") or [
            i
            for i in range(recs[0]["topk_ids_of_layer"].shape[0])
            if bool((recs[0]["topk_ids_of_layer"][i] >= 0).any())
        ]
        key = tr["meta"].get("rid", f.stem)
        pre[key] = recs[0]["topk_ids_of_layer"][layers].numpy().astype(np.int16)
        steps = [r["topk_ids_of_layer"][layers].numpy() for r in recs[1:]]
        dec[key] = (
            np.stack(steps).astype(np.int16)
            if steps
            else np.zeros((0, len(layers), 1, pre[key].shape[-1]), np.int16)
        )
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pp = out / f"{a.name}_prefilling.pkl"
    dp = out / f"{a.name}_decoding.pkl"
    pickle.dump(pre, open(pp, "wb"))
    pickle.dump(dec, open(dp, "wb"))
    print(f"{len(pre)} requests -> {pp}, {dp}")


if __name__ == "__main__":
    main()
