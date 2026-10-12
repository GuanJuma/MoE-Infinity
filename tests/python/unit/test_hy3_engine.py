# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""moe_infinity.hy3 on a tiny Hy3-FP8 checkpoint (CPU-only).

The engine must route and compute exactly like transformers' HYV3ForCausalLM
holding the same dequantized weights; traces must follow the SGLang
per-token recorder layout archer-rel-simulation reads.
"""

from __future__ import annotations

import importlib
import json
import sys
import threading
import urllib.request
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers.models.hy_v3")
pytest.importorskip("tokenizers")

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "benchmarks" / "hy3_trace"))
fake_mi = importlib.import_module("fake_moe_infinity_store")

from moe_infinity.hy3 import Hy3Engine, Hy3EngineConfig  # noqa: E402
from moe_infinity.hy3.checkpoint import Hy3Checkpoint  # noqa: E402
from moe_infinity.hy3.routing_trace import load_trace, save_trace  # noqa: E402
from moe_infinity.hy3.tiny_checkpoint import (  # noqa: E402
    make_tiny_hy3,
    reference_model,
)

PROMPT = list(range(5, 41))


@pytest.fixture(scope="module")
def ckpt_dir(tmp_path_factory):
    return make_tiny_hy3(tmp_path_factory.mktemp("tiny-hy3"))


@pytest.fixture(scope="module")
def ref(ckpt_dir):
    model = reference_model(ckpt_dir)
    from transformers.models.hy_v3 import modeling_hy_v3 as m

    routes = {}
    for i, layer in enumerate(model.model.layers):
        if isinstance(layer.mlp, m.HYV3MoE):

            def hook(mod, args, out, i=i):
                routes[i] = out[2]

            layer.mlp.gate.register_forward_hook(hook)
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([PROMPT])).logits
    first_routes = dict(routes)
    greedy, cur = [], list(PROMPT)
    with torch.no_grad():
        for _ in range(5):
            t = int(model(input_ids=torch.tensor([cur])).logits[0, -1].argmax())
            greedy.append(t)
            cur.append(t)
    return {"logits": logits, "routes": first_routes, "greedy": greedy}


def _engine(ckpt_dir, tmp_path=None, **kw):
    e_bytes = Hy3Checkpoint(ckpt_dir).expert_nbytes()
    cfg = dict(
        model_dir=str(ckpt_dir),
        device="cpu",
        dtype="float32",
        store="pinned",
        expert_cache_gib=9 * e_bytes / 2**30,
        expert_compute="bf16",
        prefill_chunk=7,
    )
    cfg.update(kw)
    lib = fake_mi if cfg["store"] == "mi" else None
    return Hy3Engine(Hy3EngineConfig(**cfg), log=lambda *a: None, store_lib=lib)


def test_checkpoint_layout(ckpt_dir):
    ck = Hy3Checkpoint(ckpt_dir)
    assert ck.moe_layers == [1, 2]  # layer 0 dense, layer 3 = MTP
    assert ck.expert_dtype() == torch.float8_e4m3fn
    names = ck.resident_names()
    assert not any(".layers.3." in n for n in names)
    assert not any(
        n.endswith(("input_scale", "weight_scale", "k_scale")) for n in names
    )
    sc = ck.expert_scales()
    assert sc.weight.shape == (2, 8, 3) and bool((sc.input > 0).all())


def test_matches_hf_reference(ckpt_dir, ref):
    eng = _engine(ckpt_dir)
    with torch.no_grad():
        logits = eng.model(input_ids=torch.tensor([PROMPT])).logits
    torch.testing.assert_close(logits, ref["logits"], rtol=1e-5, atol=1e-5)
    r = eng.generate(PROMPT, max_new_tokens=5, rid="t")
    assert r["output_ids"] == ref["greedy"]
    pre = r["trace"]["records"][0]["topk_ids_of_layer"]
    for layer, ids in ref["routes"].items():
        assert torch.equal(
            pre[layer].long().sort(-1).values, ids.sort(-1).values
        )


def test_trace_layout(ckpt_dir, tmp_path):
    eng = _engine(ckpt_dir)
    r = eng.generate(PROMPT, max_new_tokens=4, rid="req")
    tr = r["trace"]
    recs = tr["records"]
    assert [x["forward_mode"] for x in recs] == ["extend"] + ["decode"] * 3
    assert recs[0]["topk_ids_of_layer"].shape == (3, len(PROMPT), 2)
    assert recs[0]["num_tokens"] == len(PROMPT)  # 6 chunks of 7 merged
    assert all(x["topk_ids_of_layer"].shape == (3, 1, 2) for x in recs[1:])
    assert bool((recs[0]["topk_ids_of_layer"][0] == -1).all())
    assert bool((recs[0]["topk_ids_of_layer"][1:] >= 0).all())
    assert tr["meta"]["moe_layers"] == [1, 2]
    assert tr["meta"]["decode_steps"] == 3
    p = save_trace(tr, tmp_path / "x.pt")
    assert load_trace(p)["meta"]["rid"] == "req"


def test_fp8_compute_close(ckpt_dir, ref):
    eng = _engine(ckpt_dir, expert_compute="fp8")
    with torch.no_grad():
        logits = eng.model(input_ids=torch.tensor([PROMPT])).logits
    rel = (logits - ref["logits"]).norm() / ref["logits"].norm()
    assert rel < 0.15, rel


@pytest.mark.parametrize("compute", ["bf16", "fp8"])
def test_bfloat16_engine(ckpt_dir, ref, compute):
    eng = _engine(ckpt_dir, dtype="bfloat16", expert_compute=compute)
    assert {p.dtype for p in eng.model.parameters()} == {torch.bfloat16}
    assert eng.moe_modules[0].e_score_correction_bias.dtype == torch.float32
    with torch.no_grad():
        logits = eng.model(input_ids=torch.tensor([PROMPT])).logits.float()
    rel = (logits - ref["logits"]).norm() / ref["logits"].norm()
    assert rel < 0.2, rel
    assert eng.generate(PROMPT, max_new_tokens=3)["stats"]["completion_tokens"]


def test_pinned_store_evicts_and_stays_exact(ckpt_dir, ref):
    eng = _engine(ckpt_dir)
    assert eng.store.n_slots == 9  # < 2 layers x 8 experts
    r = eng.generate(PROMPT, max_new_tokens=5)
    assert r["output_ids"] == ref["greedy"]
    assert r["stats"]["prefill_expert_misses"] > 0
    assert r["stats"]["h2d_bytes"] > 9 * eng.store.nbytes


def test_mi_store_with_fake(ckpt_dir, ref, tmp_path):
    fake_mi.CALLS.clear()
    eng = _engine(ckpt_dir, store="mi", offload_dir=str(tmp_path / "store"))
    kinds = [c[0] for c in fake_mi.CALLS]
    assert kinds.count("offload") == 2 * 8 * 3 + 2
    assert "set_topology_v2" in kinds and "resize_expert_cache" in kinds
    with torch.no_grad():
        logits = eng.model(input_ids=torch.tensor([PROMPT])).logits
    torch.testing.assert_close(logits, ref["logits"], rtol=1e-5, atol=1e-5)
    r = eng.generate(PROMPT, max_new_tokens=5)
    assert r["output_ids"] == ref["greedy"]
    assert "begin" in [c[0] for c in fake_mi.CALLS]
    manifest = json.loads(
        (tmp_path / "store" / "hy3_store_manifest.json").read_text()
    )
    assert manifest["moe_layers"] == [1, 2]


def test_mi_store_refuses_foreign_dir(ckpt_dir, tmp_path):
    d = tmp_path / "other"
    d.mkdir()
    (d / "junk").write_text("x")
    with pytest.raises(RuntimeError, match="not a Hy3 expert store"):
        _engine(ckpt_dir, store="mi", offload_dir=str(d))


def test_sglang_dump_split():
    conv = importlib.import_module("sglang_dump_to_traces")

    def rec(mode, T, pos0):
        ids = torch.randint(0, 8, (3, T, 2), dtype=torch.int32)
        ids[0] = -1
        return {
            "forward_mode": mode,
            "topk_ids_of_layer": ids,
            "positions": list(range(pos0, pos0 + T)),
        }

    recs = [
        rec(1, 5, 0),
        rec(1, 3, 5),
        rec(2, 1, 8),
        rec(2, 1, 9),
        rec(4, 0, 0),
        rec(1, 4, 0),
        rec(2, 1, 4),
    ]
    reqs = conv.split_requests(recs)
    assert [len(r) for r in reqs] == [3, 2]
    assert reqs[0][0]["num_tokens"] == 8 and reqs[1][0]["num_tokens"] == 4
    with pytest.raises(ValueError, match="MIXED"):
        conv.split_requests([rec(3, 2, 0)])


def test_simulators_builtin(tmp_path, ckpt_dir):
    sim = importlib.import_module("simulate_traces")
    acc = [(1, 0), (1, 1), (1, 2), (1, 0), (1, 1), (1, 3), (1, 0)]
    assert sim._belady(acc, 2)[0] == 2
    assert sim.lru(acc, 2)[0] == 0
    assert sim.lru(acc, 4)[0] == 3
    eng = _engine(ckpt_dir)
    for i in range(2):
        r = eng.generate(PROMPT[i:], max_new_tokens=6, rid=f"r{i}")
        save_trace(r["trace"], tmp_path / "tr" / f"{i}.pt")
    sim.main(
        [
            "--traces",
            str(tmp_path / "tr"),
            "--capacities",
            "4",
            "16",
            "--out",
            str(tmp_path / "sim"),
        ]
    )
    rows = json.loads((tmp_path / "sim" / "results.json").read_text())
    for scope in ("global", "per_request"):
        by = {
            (r["capacity"], r["policy"]): r["hit_rate"]
            for r in rows
            if r["scope"] == scope
        }
        # 16 slots hold every routed expert: only compulsory misses remain.
        assert by[(16, "belady")] == by[(16, "lru")] == by[(16, "lfu")]
        assert by[(4, "belady")] >= max(by[(4, "lru")], by[(4, "lfu")])
    assert (tmp_path / "sim" / "summary.md").exists()


def test_server_roundtrip(ckpt_dir, tmp_path):
    from http.server import ThreadingHTTPServer

    from moe_infinity.hy3 import server

    server._State.engine = _engine(ckpt_dir)
    server._State.trace_dir = tmp_path / "srv"
    server._State.trace_dir.mkdir()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/v1/chat/completions"
        body = json.dumps(
            {
                "model": "/model",
                "messages": [{"role": "user", "content": "你好"}],
                "max_tokens": 4,
                "chat_template_kwargs": {"reasoning_effort": "no_think"},
            }
        ).encode()
        req = urllib.request.Request(
            url, body, {"Content-Type": "application/json"}
        )
        out = json.loads(urllib.request.urlopen(req, timeout=60).read())
        assert out["usage"]["completion_tokens"] == 4
        assert out["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
        req = urllib.request.Request(
            url,
            json.dumps(
                {
                    **json.loads(body),
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
            ).encode(),
            {"Content-Type": "application/json"},
        )
        text = urllib.request.urlopen(req, timeout=60).read().decode()
        assert text.rstrip().endswith("data: [DONE]")
        assert '"usage"' in text
    finally:
        srv.shutdown()
    assert len(list((tmp_path / "srv").glob("*.pt"))) == 2


def test_preflight(tmp_path):
    from moe_infinity.hy3.cli import preflight

    proc, sysfs = tmp_path / "proc", tmp_path / "node"
    (proc / "self").mkdir(parents=True)
    (proc / "meminfo").write_text(f"MemAvailable:   {600 * 2**20} kB\n")
    (proc / "self" / "status").write_text("Mems_allowed_list:\t0\n")
    (sysfs / "node0").mkdir(parents=True)
    (sysfs / "node0" / "meminfo").write_text(
        f"Node 0 MemFree:   {300 * 2**20} kB\n"
        f"Node 0 Inactive(file):  {100 * 2**20} kB\n"
    )
    ok, lines = preflight(
        str(tmp_path / "store"),
        min_ram_gib=500,
        min_disk_gib=0,
        min_gpu_free_gib=0,
        proc=str(proc),
        sysfs=str(sysfs),
    )
    assert not ok  # node 0 has only 400 GiB free + reclaimable
    assert lines[0].startswith("OK") and "600 GiB" in lines[0]
    assert lines[1].startswith("FAIL") and "node0 400" in lines[1]
    ok, lines = preflight(
        str(tmp_path / "store"), 350, 0, 0, proc=str(proc), sysfs=str(sysfs)
    )
    assert all(not ln.startswith("FAIL") for ln in lines[:3])


def test_memory_plan(ckpt_dir):
    from moe_infinity.hy3.cli import memory_plan

    plan = memory_plan(str(ckpt_dir), 0.001, 1024)
    assert plan["routed_experts"] == 16
    assert plan["kv_bytes_per_token"] == 2 * 3 * 2 * 16 * 2
    assert plan["mi_pinned_host_gib"] >= plan["experts_host_gib"]
