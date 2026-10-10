# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""benchmarks/expert_placement: checkpoint detection, kernels, CLI (CPU-only)."""

from __future__ import annotations

import csv
import importlib
import json
import sys
import textwrap
from pathlib import Path

import pytest
import torch

BENCH = Path(__file__).resolve().parents[3] / "benchmarks" / "expert_placement"
sys.path.insert(0, str(BENCH))

ck = importlib.import_module("checkpoint_expert")
runners = importlib.import_module("expert_runners")
fake = importlib.import_module("make_fake_checkpoint")
bench = importlib.import_module("expert_placement_bench")
summ = importlib.import_module("summarize_placement")


def _cpu_moe_available() -> bool:
    try:
        from moe_infinity.kernel.cpu import is_available

        return is_available()
    except Exception:
        return False


CPU_MOE = _cpu_moe_available()
needs_cpu_moe = pytest.mark.skipif(
    not CPU_MOE, reason="CPU MoE kernels unavailable"
)


@pytest.fixture(scope="module")
def ckpts(tmp_path_factory):
    root = tmp_path_factory.mktemp("ckpt")
    out = {}
    for fmt in ("hy3_fp8", "block_fp8", "bf16"):
        out[fmt] = fake.make(root / fmt, fmt)
    out["stacked_bf16"] = fake.make(root / "stacked", "stacked_bf16", inter=96)
    return out


def _load(path, layer=1, expert=0):
    ix = ck.SafetensorsIndex(path)
    return ck.load_expert(ix, ck.load_config(path), layer, expert)


# -- checkpoint detection ----------------------------------------------------


def test_hy3_layout_detected(ckpts):
    ix = ck.SafetensorsIndex(ckpts["hy3_fp8"])
    layers = ck.discover_experts(ix)
    assert sorted(layers) == [1, 2]  # layer 0 is dense (first_k_dense_replace)
    le = layers[1]
    assert le.kind == "per_expert" and le.expert_ids == [0, 1, 2, 3]
    assert le.base == "model.layers.1.mlp.experts."
    ew = ck.load_expert(ix, ck.load_config(ckpts["hy3_fp8"]), 2, 3, layers)
    assert ew.quant == "fp8_per_tensor"
    assert (ew.hidden_size, ew.intermediate_size) == (256, 128)
    assert ew.gate.scale.shape == () and ew.gate.input_scale.shape == (1,)
    a1, a2 = ew.static_input_scales()
    assert a1 == pytest.approx(0.0021) and a2 == pytest.approx(0.0021)
    d = ew.describe()
    assert d["projs"]["down"]["shape"] == [256, 128]
    assert (
        "model.layers.2.mlp.experts.3.up_proj.input_scale"
        in d["projs"]["up"]["tensors"]
    )


def test_reader_matches_safetensors_library(ckpts):
    st = pytest.importorskip("safetensors.torch")
    for fmt in ("hy3_fp8", "block_fp8", "stacked_bf16"):
        ix = ck.SafetensorsIndex(ckpts[fmt])
        for shard in sorted(Path(ckpts[fmt]).glob("*.safetensors")):
            ref = st.load_file(str(shard))
            for name, t in ref.items():
                got = ix.load(name)
                assert got.dtype == t.dtype and got.shape == t.shape
                assert torch.equal(
                    got.reshape(-1).view(torch.uint8),
                    t.reshape(-1).view(torch.uint8),
                ), name


def test_block_fp8_and_dequant(ckpts):
    ew = _load(ckpts["block_fp8"], 1, 2)
    assert ew.quant == "fp8_block" and ew.gate.block == (128, 128)
    assert tuple(ew.down.scale.shape) == (2, 1)
    w = ew.down.dequant()
    manual = ew.down.weight.float().clone()
    manual[:128] *= ew.down.scale[0, 0]
    manual[128:] *= ew.down.scale[1, 0]
    assert torch.allclose(w, manual)
    grid = ew.gate.scale_grid((128, 128))
    assert torch.equal(grid, ew.gate.scale)
    with pytest.raises(ck.CheckpointError):
        ew.gate.scale_grid((64, 128))


def test_stacked_layout_reads_one_expert(ckpts):
    path = ckpts["stacked_bf16"]
    ix = ck.SafetensorsIndex(path)
    layers = ck.discover_experts(ix)
    assert layers[1].kind == "stacked" and layers[1].num_experts == 4
    ew = ck.load_expert(ix, ck.load_config(path), 1, 3, layers)
    full = ix.load("model.layers.1.mlp.experts.gate_up_proj")
    assert torch.equal(ew.gate.weight, full[3, :96])
    assert torch.equal(ew.up.weight, full[3, 96:])
    assert torch.equal(
        ew.down.weight, ix.load("model.layers.1.mlp.experts.down_proj")[3]
    )
    with pytest.raises(ck.CheckpointError, match="out of range"):
        ck.load_expert(ix, ck.load_config(path), 1, 4, layers)


def test_clear_errors(ckpts, tmp_path):
    with pytest.raises(
        ck.CheckpointError, match="no routed experts; MoE layers: 1-2"
    ):
        _load(ckpts["hy3_fp8"], 0, 0)
    with pytest.raises(ck.CheckpointError, match="expert 7 not found"):
        _load(ckpts["hy3_fp8"], 1, 7)
    with pytest.raises(ck.CheckpointError, match="does not exist"):
        ck.SafetensorsIndex(tmp_path / "nope")
    w = torch.zeros(128, 256).to(torch.float8_e4m3fn)
    with pytest.raises(ck.CheckpointError, match="cannot classify"):
        ck._infer_scheme("x", w, torch.ones(3, 5), None)
    with pytest.raises(ck.CheckpointError, match="without a weight_scale"):
        ck._infer_scheme("x", w, None, None)
    with pytest.raises(ck.CheckpointError, match="integer-packed"):
        ck._infer_scheme(
            "x", torch.zeros(4, 4, dtype=torch.int32), torch.ones(4), None
        )
    assert (
        ck._infer_scheme("x", w, torch.ones(128, 1), None)[0] == "per_channel"
    )
    assert ck._infer_scheme("x", w, torch.ones(2, 4), None)[2] == (64, 64)


# -- kernels -------------------------------------------------------------------


def _x(M, K, std):
    g = torch.Generator().manual_seed(M)
    return (torch.randn(M, K, generator=g) * std).to(torch.bfloat16)


@pytest.mark.parametrize("fmt", ["hy3_fp8", "block_fp8", "bf16"])
def test_gpu_runners_numerics_on_cpu(ckpts, fmt):
    ew = _load(ckpts[fmt])
    std = 0.155 if fmt == "hy3_fp8" else 1.0
    x = _x(33, ew.hidden_size, std)
    ref = runners.reference_expert(x, ew)
    r = runners.build_gpu_runner(ew, "torch_bf16", "cpu")
    fn, reset = r.bind(x)
    assert reset is None and runners.rel_err(fn(), ref) < 0.02
    if fmt == "hy3_fp8":
        r = runners.build_gpu_runner(ew, "torch_scaled_mm", "cpu")
        assert r.weights["w13"].dtype == torch.float8_e4m3fn
        assert r.weight_bytes == 3 * 128 * 256 + 4 * 4
        fn, _ = r.bind(x)
        assert runners.rel_err(fn(), ref) < 0.12
    else:
        with pytest.raises(runners.KernelUnavailable):
            runners.build_gpu_runner(ew, "torch_scaled_mm", "cpu")


def test_merged_w13_follows_sglang(ckpts):
    ew = _load(ckpts["hy3_fp8"])
    w13, s = runners.merged_per_tensor_w13(ew)
    assert s == max(float(ew.gate.scale), float(ew.up.scale))
    deq = w13.float() * s
    ref = torch.cat([ew.gate.dequant(), ew.up.dequant()])
    assert float((deq - ref).norm() / ref.norm()) < 0.05


@needs_cpu_moe
@pytest.mark.parametrize(
    "fmt,kernel,tol",
    [
        ("hy3_fp8", "sglang_fp8_w8a16", 0.02),
        ("hy3_fp8", "sglang_bf16", 0.02),
        ("hy3_fp8", "sglang_int8_w8a8", 0.08),
        ("hy3_fp8", "torch_bf16", 0.02),
        ("block_fp8", "sglang_fp8_w8a16", 0.02),
        ("bf16", "sglang_bf16", 0.02),
    ],
)
def test_cpu_runners_numerics(ckpts, fmt, kernel, tol):
    ew = _load(ckpts[fmt])
    r = runners.build_cpu_runner(ew, kernel)
    for M in (1, 7, 64):
        x = _x(M, ew.hidden_size, 0.5)
        ref = runners.reference_expert(x, ew)
        xh = x.clone()
        fn, _ = r.bind(xh)
        out = fn()
        assert out.data_ptr() == xh.data_ptr() or kernel == "torch_bf16"
        assert runners.rel_err(xh, ref) < tol, (kernel, M)


@needs_cpu_moe
def test_cpu_fp8_rejects_bf16_checkpoint(ckpts):
    with pytest.raises(runners.KernelUnavailable, match="not FP8"):
        runners.build_cpu_runner(_load(ckpts["bf16"]), "sglang_fp8_w8a16")


# -- SGLang adapter against a stand-in with the v0.5.19 signatures -------------

FAKE_SGLANG = {
    "sglang/__init__.py": "__version__ = '0.5.19-fake'\n",
    "sglang/srt/__init__.py": "",
    "sglang/srt/layers/__init__.py": "",
    "sglang/srt/layers/moe/__init__.py": "",
    "sglang/srt/layers/moe/topk.py": """
        from typing import NamedTuple
        import torch
        class StandardTopKOutput(NamedTuple):
            topk_weights: torch.Tensor
            topk_ids: torch.Tensor
            router_logits: torch.Tensor
    """,
    # Field list copied from sglang v0.5.19 moe_runner/base.py.
    "sglang/srt/layers/moe/moe_runner/__init__.py": """
        from dataclasses import dataclass
        from typing import Optional, Any
        @dataclass
        class MoeRunnerConfig:
            num_experts: Optional[int] = None
            num_local_experts: Optional[int] = None
            hidden_size: Optional[int] = None
            intermediate_size_per_partition: Optional[int] = None
            layer_id: Optional[int] = None
            top_k: Optional[int] = None
            num_fused_shared_experts: Optional[int] = None
            params_dtype: Any = None
            routing_method_type: Any = None
            activation: str = "silu"
            is_gated: bool = True
            apply_router_weight_on_input: bool = False
            inplace: bool = True
            no_combine: bool = False
            routed_scaling_factor: Optional[float] = None
            gemm1_alpha: Optional[float] = None
            gemm1_beta: Optional[float] = None
            gemm1_clamp_limit: Optional[float] = None
            swiglu_limit: Optional[float] = None
            gate_up_interleaved: bool = True
            layer: Any = None
            use_tp_all_gather_activation: bool = False
    """,
    "sglang/srt/runtime_context.py": """
        PUBLISHED = []
        def get_exec():
            if not PUBLISHED:
                raise ValueError("config namespace 'exec' not published")
            return PUBLISHED[-1]
    """,
    "sglang/srt/server_args.py": """
        from dataclasses import dataclass
        from sglang.srt import runtime_context
        @dataclass
        class ServerArgs:
            model_path: str
        def set_global_server_args_for_scheduler(server_args):
            runtime_context.PUBLISHED.append(server_args)
    """,
    # Signature copied from sglang v0.5.19 triton_utils/fused_moe.py.
    "sglang/srt/layers/moe/fused_moe_triton/__init__.py": """
        import torch
        import torch.nn.functional as F
        from sglang.srt.runtime_context import get_exec
        CALLS = []
        def fused_experts(hidden_states, w1, w2, topk_output, moe_runner_config, b1=None,
                          b2=None, use_fp8_w8a8=False, use_int8_w8a8=False,
                          use_int8_w8a16=False, use_int4_w4a16=False,
                          per_channel_quant=False, w1_scale=None, w2_scale=None,
                          w1_zp=None, w2_zp=None, a1_scale=None, a2_scale=None,
                          block_shape=None, a1_q=None, fuse_swiglu_interleaved=False):
            get_exec()
            cfg = moe_runner_config
            assert cfg.inplace and not cfg.no_combine
            tw, ids, _ = topk_output
            assert int(ids.max()) == 0 and w1.shape[0] == 1
            CALLS.append(dict(use_fp8_w8a8=use_fp8_w8a8, block_shape=block_shape,
                              a1=a1_scale, w1_scale=w1_scale, per_channel=per_channel_quant))
            x = hidden_states.float()
            if use_fp8_w8a8:
                assert block_shape is None and not per_channel_quant
                x = (x / a1_scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * a1_scale
                g1 = w1[0].float() * w1_scale[0]
                g2 = w2[0].float() * w2_scale[0]
            else:
                g1, g2 = w1[0].float(), w2[0].float()
            h = x @ g1.t()
            n = h.shape[-1] // 2
            a = F.silu(h[:, :n]) * h[:, n:]
            if use_fp8_w8a8:
                a = (a / a2_scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * a2_scale
            out = (a @ g2.t()) * tw
            hidden_states.copy_(out.to(hidden_states.dtype))
            return hidden_states
    """,
}


@pytest.fixture
def fake_sglang(tmp_path, monkeypatch):
    for rel, src in FAKE_SGLANG.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(src))
    for mod in [
        m for m in sys.modules if m == "sglang" or m.startswith("sglang.")
    ]:
        monkeypatch.delitem(sys.modules, mod)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(runners, "_SGLANG", {})
    yield
    for mod in [
        m for m in sys.modules if m == "sglang" or m.startswith("sglang.")
    ]:
        sys.modules.pop(mod, None)


def test_sglang_triton_adapter_wiring(ckpts, fake_sglang):
    ew = _load(ckpts["hy3_fp8"])
    r = runners.build_gpu_runner(ew, "sglang_triton", "cpu")
    assert runners._SGLANG["published"] == "dummy ServerArgs"
    assert (
        runners._SGLANG["new_api"]
        and runners._SGLANG["version"] == "0.5.19-fake"
    )
    assert r.meta["mode"] == "per_tensor" and "static" in r.meta["act_quant"]
    assert r.weights["w13"].shape == (1, 256, 256) and r.weights[
        "w2"
    ].shape == (1, 256, 128)
    x = _x(17, 256, 0.155)
    fn, reset = r.bind(x)
    reset()
    out = fn()
    assert not torch.equal(out, x)  # input untouched, work buffer overwritten
    assert runners.rel_err(out, runners.reference_expert(x, ew)) < 0.12
    sm = runners.build_gpu_runner(ew, "torch_scaled_mm", "cpu")
    assert runners.rel_err(out, sm.bind(x)[0]()) < 0.03
    from sglang.srt.layers.moe.fused_moe_triton import CALLS

    call = CALLS[-1]
    assert (
        call["use_fp8_w8a8"]
        and call["block_shape"] is None
        and not call["per_channel"]
    )
    assert float(call["a1"]) == pytest.approx(0.0021)
    assert "sglang_triton" in runners.resolve_gpu_kernels(["auto"], ew)


def test_sweep_through_fake_sglang(ckpts, fake_sglang, tmp_path):
    out = tmp_path / "sg"
    rc = bench.main(
        [
            "--model-dir", str(ckpts["hy3_fp8"]), "--device", "cpu",
            "--tokens", "1,8", "--scenarios", "gpu,fetch",
            "--gpu-kernels", "sglang_triton", "--warmup", "1", "--repeats", "2",
            "--min-repeats", "1", "--out-dir", str(out),
        ]
    )  # fmt: skip
    assert rc == 0
    res = json.loads((out / "results.json").read_text())
    assert not res["errors"], res["errors"]
    assert {r["variant"] for r in res["rows"]} == {
        "sglang_triton",
        "sglang_triton+pinned",
        "sglang_triton+pageable",
    }
    assert all(r["rel_err"] < 0.15 for r in res["rows"])


def test_sglang_missing_is_reported(ckpts, monkeypatch):
    monkeypatch.setattr(runners, "_SGLANG", {})
    monkeypatch.setitem(sys.modules, "sglang", None)
    ew = _load(ckpts["hy3_fp8"])
    with pytest.raises(runners.KernelUnavailable, match="cannot import SGLang"):
        runners.build_gpu_runner(ew, "sglang_triton", "cpu")
    assert runners.resolve_gpu_kernels(["auto"], ew) == [
        "torch_scaled_mm",
        "torch_bf16",
    ]


# -- CLI -----------------------------------------------------------------------


def test_list_experts_cli(ckpts, capsys):
    assert (
        bench.main(
            [
                "--model-dir",
                str(ckpts["hy3_fp8"]),
                "--list-experts",
                "--layer",
                "2",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "MoE layers     : 1-2" in out and "quant=fp8_per_tensor" in out
    assert "model.layers.2.mlp.experts.0.gate_proj.weight_scale" in out
    assert (
        bench.main(
            [
                "--model-dir",
                str(ckpts["hy3_fp8"]),
                "--list-experts",
                "--layer",
                "0",
            ]
        )
        == 2
    )


def test_cuda_required_without_device_flag(ckpts, capsys):
    if torch.cuda.is_available():
        pytest.skip("CUDA present")
    rc = bench.main(
        [
            "--model-dir",
            str(ckpts["hy3_fp8"]),
            "--dry-run",
            "--out-dir",
            "/tmp/unused",
        ]
    )
    assert rc == 2 and "CUDA not available" in capsys.readouterr().err


def test_sweep_end_to_end_on_cpu_device(ckpts, tmp_path):
    out = tmp_path / "res"
    cpu_k = "torch_bf16" + (",sglang_fp8_w8a16" if CPU_MOE else "")
    rc = bench.main(
        [
            "--model-dir",
            str(ckpts["hy3_fp8"]),
            "--layer",
            "1",
            "--expert",
            "2",
            "--device",
            "cpu",
            "--tokens",
            "1,4,32",
            "--gpu-kernels",
            "torch_scaled_mm,torch_bf16",
            "--cpu-kernels",
            cpu_k,
            "--warmup",
            "1",
            "--repeats",
            "3",
            "--min-repeats",
            "1",
            "--cpu-llc-flush-mb",
            "4",
            "--skip-amx-check",
            "--out-dir",
            str(out),
        ]
    )
    assert rc == 0
    res = json.loads((out / "results.json").read_text())
    assert not res["errors"], res["errors"]
    n_cpu = len(cpu_k.split(","))
    assert len(res["rows"]) == 3 * (2 + n_cpu + 2 * 2)
    assert res["expert"]["quant"] == "fp8_per_tensor"
    assert set(res["definitions"]) >= {
        "gpu_resident",
        "cpu_compute",
        "cpu_store_gpu_compute",
    }
    assert (
        res["setup"]["gpu"]["torch_scaled_mm"]["weight_bytes"]
        == 3 * 128 * 256 + 16
    )
    rows = list(csv.DictReader(open(out / "results.csv")))
    for r in rows:
        total, comp, over = (
            float(r[k]) for k in ("total_ms", "compute_ms", "overhead_ms")
        )
        assert (
            total > 0
            and comp > 0
            and over == pytest.approx(total - comp, abs=1e-6)
        )
        assert float(r["rel_err"]) < 0.15
        if r["scenario"] == "cpu_compute":
            assert float(r["d2h_ms"]) + float(r["h2d_ms"]) <= total + 1e-6
        if r["scenario"] == "cpu_store_gpu_compute":
            assert int(r["xfer_bytes"]) > 0 and r["host_mem"] in (
                "pinned",
                "pageable",
            )
    assert {
        r["variant"] for r in rows if r["scenario"] == "cpu_store_gpu_compute"
    } == {
        "torch_scaled_mm+pinned",
        "torch_scaled_mm+pageable",
        "torch_bf16+pinned",
        "torch_bf16+pageable",
    }
    assert (out / "env" / "lscpu.txt").exists()
    assert summ.main([str(out), "--no-plots"]) == 0
    md = (out / "summary.md").read_text()
    assert (
        "## Total latency" in md
        and "S3 CPU-stored->GPU" in md
        and "## Crossovers" in md
    )


def test_dry_run(ckpts, tmp_path):
    out = tmp_path / "dry"
    rc = bench.main(
        [
            "--model-dir",
            str(ckpts["block_fp8"]),
            "--device",
            "cpu",
            "--dry-run",
            "--scenarios",
            "gpu",
            "--gpu-kernels",
            "auto",
            "--out-dir",
            str(out),
        ]
    )
    assert rc == 0
    res = json.loads((out / "results.json").read_text())
    assert res["dry_run"] and {r["tokens"] for r in res["rows"]} == {1, 16}
    assert {r["variant"] for r in res["rows"]} == {"torch_bf16"}
