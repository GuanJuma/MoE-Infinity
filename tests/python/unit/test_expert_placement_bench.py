# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""benchmarks/expert_placement: checkpoint detection, kernels, CLI (CPU-only)."""

from __future__ import annotations

import csv
import importlib
import json
import os
import re
import subprocess
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
mi_store = importlib.import_module("mi_expert_store")
sys.path.insert(0, str(Path(__file__).resolve().parent))
fake_mi = importlib.import_module("fake_moe_infinity_store")
REPO = Path(__file__).resolve().parents[3]
FAKE_MI = "fake_moe_infinity_store"


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


def _main_ok(args, out):
    rc = bench.main([*args, "--out-dir", str(out)])
    assert rc == 0
    return json.loads((out / "results.json").read_text())


COMMON = [
    "--device", "cpu", "--warmup", "1", "--repeats", "2", "--min-repeats", "1",
    "--cpu-llc-flush-mb", "4", "--skip-amx-check", "--mi-store-lib", FAKE_MI,
]  # fmt: skip


def test_sweep_through_fake_sglang(ckpts, fake_sglang, tmp_path):
    res = _main_ok(
        ["--model-dir", str(ckpts["hy3_fp8"]), "--tokens", "1,8",
         "--scenarios", "gpu,fetch", "--gpu-kernels", "sglang_triton", *COMMON],
        tmp_path / "sg",
    )  # fmt: skip
    assert not res["errors"], res["errors"]
    assert {r["variant"] for r in res["rows"]} == {
        "sglang_triton",
        *(f"sglang_triton+{m}" for m in bench.FETCH_MODES),
    }
    assert all(
        r["rel_err"] < 0.15 and r["rel_err_w8a8"] < 0.05 for r in res["rows"]
    )


def test_sglang_missing_is_reported(ckpts, monkeypatch):
    monkeypatch.setattr(runners, "_SGLANG", {})
    monkeypatch.setitem(sys.modules, "sglang", None)
    ew = _load(ckpts["hy3_fp8"])
    with pytest.raises(runners.KernelUnavailable, match="cannot import SGLang"):
        runners.build_gpu_runner(ew, "sglang_triton", "cpu")
    assert runners.resolve_gpu_kernels(["auto"], ew) == ["torch_scaled_mm"]
    with pytest.raises(runners.KernelUnavailable, match="no FP8 GPU kernel"):
        runners.resolve_gpu_kernels(["auto"], _load(ckpts["block_fp8"]))


def test_fp8_is_the_only_default():
    gpu, cpu = runners.default_kernels("fp8")
    assert gpu == ("sglang_triton", "torch_scaled_mm")
    assert cpu == ("sglang_fp8_w8a16", "sglang_fp8_w8a8_emu")
    gpu, cpu = runners.default_kernels("bf16")
    assert gpu[0] == "batchgen_triton" and cpu == ("sglang_bf16",)
    args = bench.parse_args(["--model-dir", "x"])
    assert (
        args.precision == "fp8"
        and args.gpu_kernels == args.cpu_kernels == "default"
    )


# -- BF16 mode -------------------------------------------------------------------


def test_bf16_dequant_keeps_per_projection_scales(ckpts):
    ew = _load(ckpts["hy3_fp8"])
    b = ew.to_bf16()
    assert b.quant == "bfloat16" and "dequantized" in b.source
    for p, q in zip(ew.projs, b.projs):
        want = (p.weight.float() * float(p.scale)).to(torch.bfloat16)
        assert torch.equal(q.weight, want), p.role
    assert float(ew.gate.scale) != float(ew.up.scale)
    assert b.checkpoint_nbytes == 2 * (ew.checkpoint_nbytes - 3 * 4)


def test_batchgen_runner_wiring(ckpts, monkeypatch):
    seen = {}

    def fake_ffn(x, g, u, d):
        seen["packed"] = (
            u.data_ptr() == g.data_ptr() + g.numel() * g.element_size()
        )
        h = torch.cat(
            [x.float() @ g.float().t(), x.float() @ u.float().t()], -1
        )
        return (runners._act_mul(h, "silu") @ d.float().t()).to(x.dtype)

    monkeypatch.setattr(runners, "_BATCHGEN", {"expert_ffn": fake_ffn})
    ew = _load(ckpts["hy3_fp8"])
    with pytest.raises(runners.KernelUnavailable, match="BF16-only"):
        runners.build_gpu_runner(ew, "batchgen_triton", "cpu")
    b = ew.to_bf16()
    r = runners.build_gpu_runner(b, "batchgen_triton", "cpu")
    x = _x(9, 256, 0.5)
    out = r.bind(x)[0]()
    assert (
        seen["packed"]
        and runners.rel_err(out, runners.reference_expert(x, b)) < 0.02
    )


def test_batchgen_triton_interpreter(ckpts, tmp_path):
    pytest.importorskip("triton")
    code = textwrap.dedent(
        f"""
        import sys, torch
        sys.path[:0] = [{str(BENCH)!r}, {str(REPO)!r}]
        from checkpoint_expert import SafetensorsIndex, load_config, load_expert
        import expert_runners as R
        p = {str(ckpts["hy3_fp8"])!r}
        ew = load_expert(SafetensorsIndex(p), load_config(p), 1, 0).to_bf16()
        r = R.build_gpu_runner(ew, "batchgen_triton", "cpu")
        r.weights = {{k: v.float() for k, v in r.weights.items()}}
        for M in (1, 17):
            x = torch.randn(M, 256) * 0.5
            err = R.rel_err(r.bind(x)[0](), R.reference_expert(x, ew))
            assert err < 1e-4, (M, err)
        print("ok")
        """
    )
    env = dict(os.environ, TRITON_INTERPRET="1")
    p = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if "No module named 'triton" in p.stderr:
        pytest.skip("triton missing")
    assert p.returncode == 0 and "ok" in p.stdout, p.stderr[-2000:]


# -- CPU W8A8 emulation ------------------------------------------------------------


@needs_cpu_moe
def test_cpu_w8a8_emu_matches_gpu_w8a8_math(ckpts):
    ew = _load(ckpts["hy3_fp8"])
    r = runners.build_cpu_runner(ew, "sglang_fp8_w8a8_emu")
    assert r.fp8 and r.tensors["w13"].dtype == torch.float8_e4m3fn
    gpu = runners.build_gpu_runner(ew, "torch_scaled_mm", "cpu")
    assert torch.equal(
        runners.merged_per_tensor_w13(ew)[0].view(torch.uint8),
        gpu.weights["w13"].view(torch.uint8),
    )
    for M in (1, 7, 64):
        x = _x(M, 256, 0.155)
        xh = x.clone()
        out = r.bind(xh)[0]().clone()
        assert set(r.last) == {"act_quant_ms", "gemm_ms"}
        ref8 = runners.reference_expert_w8a8(x, ew)
        assert runners.rel_err(out, ref8) < 0.03, M
        assert runners.rel_err(out, gpu.bind(x)[0]()) < 0.03, M


# -- MoE-Infinity store harness --------------------------------------------------


def _bindings_in_core():
    names = set()
    for f in (REPO / "core" / "python").glob("*.cpp"):
        names |= set(re.findall(r'\.def\(\s*"(\w+)"', f.read_text()))
    return names


def test_store_harness_only_calls_real_bindings():
    missing = set(mi_store.USED_BINDINGS) - _bindings_in_core()
    assert not missing, missing
    src = (BENCH / "mi_expert_store.py").read_text()
    called = set(re.findall(r"self\.h\.(\w+)\(", src))
    assert called <= set(mi_store.USED_BINDINGS), called - set(
        mi_store.USED_BINDINGS
    )


def test_store_moves_nodes_through_runtime_api(tmp_path):
    a = {
        "w13": torch.randn(64, 32).to(torch.float8_e4m3fn),
        "s": torch.tensor(0.5),
    }
    b = {k: v.clone() for k, v in a.items()}
    st = mi_store.MoEInfinityExpertStore(
        tmp_path / "s", {"a": a, "b": b}, lib=fake_mi
    )
    h = st.h
    names = [c[0] for c in fake_mi.CALLS[-12:]]
    assert "set_topology_v2" in names and names.index(
        "set_topology_v2"
    ) > names.index("register")
    stage = h.topology[1]
    assert (
        stage[1] is True and len(stage[2]) == 2
    )  # one sparse stage, two expert nodes
    w = st.tensors("a")["w13"]
    host_ptr = w.data_ptr()
    assert torch.equal(w.view(torch.uint8), a["w13"].view(torch.uint8))
    st.fetch("a")
    assert (
        st.on_gpu("a")
        and w.data_ptr() != host_ptr
        and st.h2d_bytes_total() == a["w13"].nbytes + 4
    )
    st.release("a")
    assert st.on_gpu(
        "a"
    )  # end() keeps the node cached, like the post-forward hook
    st.evict_all()
    assert not st.on_gpu("a") and w.data_ptr() == host_ptr
    st.set_cache_limit(None)
    st.fetch("b")
    st.release("b")
    st.set_cache_limit(st.nodes["a"].aligned_bytes)
    st.fetch("a")  # cache full: the runtime evicts b first
    assert st.on_gpu("a") and not st.on_gpu("b")
    st.release("a")
    st.evict_all()
    st.prefetch("b")
    assert st.on_gpu("b")
    st.close()
    assert not (tmp_path / "s").exists()


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
    assert '"quant_method": "fp8"' in out and "torch_dtype=" in out
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


def _check_rows(rows):
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


def test_fp8_sweep_through_moe_infinity_store(ckpts, tmp_path):
    out = tmp_path / "fp8"
    cpu_k = "sglang_fp8_w8a16,sglang_fp8_w8a8_emu" if CPU_MOE else "torch_bf16"
    res = _main_ok(
        ["--model-dir", str(ckpts["hy3_fp8"]), "--layer", "1", "--expert", "2",
         "--tokens", "1,4,32", "--gpu-kernels", "default", "--cpu-kernels", cpu_k, *COMMON],
        out,
    )  # fmt: skip
    errs = [e for e in res["errors"] if "sglang_triton" not in e["stage"]]
    assert not errs, errs
    assert res["precision"] == "fp8" and res["setup"]["mi_store"]["nodes"]
    rows = list(csv.DictReader(open(out / "results.csv")))
    _check_rows(rows)
    s3 = [r for r in rows if r["scenario"] == "cpu_store_gpu_compute"]
    assert {r["variant"] for r in s3} == {
        f"torch_scaled_mm+{m}" for m in bench.FETCH_MODES
    }
    fp8_bytes = 3 * 128 * 256 + 16
    for r in s3:
        assert int(r["xfer_bytes"]) == fp8_bytes
        if "+mi_" in r["variant"]:
            assert r["storage"].startswith("MoE-Infinity pinned host pool")
            assert (
                int(r["mi_h2d_bytes"]) == fp8_bytes
                and float(r["release_ms"]) >= 0
            )
            assert "Node::SetDevice" in r["fetch_path"]
        else:
            assert r["fetch_path"].startswith("reference")
        assert float(r["rel_err_w8a8"]) < 0.05
    s2 = [r for r in rows if r["scenario"] == "cpu_compute"]
    assert all(
        r["storage"].startswith("MoE-Infinity pinned host pool") for r in s2
    )
    if CPU_MOE:
        emu = [r for r in s2 if r["kernel"] == "sglang_fp8_w8a8_emu"]
        assert all(
            float(r["act_quant_ms"]) > 0 and float(r["rel_err_w8a8"]) < 0.05
            for r in emu
        )
        par = res["parity"]["pairs"][
            "cpu:sglang_fp8_w8a8_emu vs gpu:torch_scaled_mm"
        ]
        assert max(par.values()) < 0.03
    assert summ.main([str(out), "--no-plots"]) == 0
    md = (out / "summary.md").read_text()
    for part in (
        "## Total latency",
        "## Crossovers",
        "MoE-Infinity load path vs raw",
        "## Kernel parity",
    ):
        assert part in md


def test_bf16_mode_moves_the_dequantized_expert(ckpts, tmp_path):
    out = tmp_path / "bf16"
    cpu_k = "sglang_bf16" if CPU_MOE else "torch_bf16"
    res = _main_ok(
        ["--model-dir", str(ckpts["hy3_fp8"]), "--precision", "bf16", "--tokens", "1,8",
         "--gpu-kernels", "torch_bf16", "--cpu-kernels", cpu_k,
         "--fetch-modes", "mi_fetch,raw_pinned", *COMMON],
        out,
    )  # fmt: skip
    assert not res["errors"], res["errors"]
    assert res["precision"] == "bf16" and res["expert"]["quant"] == "bfloat16"
    assert (
        "dequantized to BF16 from fp8_per_tensor"
        in res["setup"]["expert_source"]
    )
    assert res["checkpoint_expert"]["quant"] == "fp8_per_tensor"
    rows = list(csv.DictReader(open(out / "results.csv")))
    _check_rows(rows)
    bf16_bytes = 2 * 3 * 128 * 256
    assert {
        int(r["xfer_bytes"])
        for r in rows
        if r["scenario"] == "cpu_store_gpu_compute"
    } == {bf16_bytes}
    assert all(r["numerics"] == "BF16" for r in rows)


def test_fp8_mode_rejects_bf16_checkpoint(ckpts, tmp_path, capsys):
    rc = bench.main(
        ["--model-dir", str(ckpts["bf16"]), *COMMON, "--out-dir", str(tmp_path)]
    )
    assert rc == 2 and "needs an FP8 checkpoint" in capsys.readouterr().err


def test_torch_store_fallback_and_missing_store(ckpts, tmp_path):
    res = _main_ok(
        ["--model-dir", str(ckpts["hy3_fp8"]), "--tokens", "1", "--scenarios", "fetch",
         "--gpu-kernels", "torch_scaled_mm", "--expert-store", "torch", *COMMON],
        tmp_path / "t",
    )  # fmt: skip
    assert {r["variant"] for r in res["rows"]} == {
        "torch_scaled_mm+raw_pinned",
        "torch_scaled_mm+raw_pageable",
    }
    res = _main_ok(
        ["--model-dir", str(ckpts["hy3_fp8"]), "--tokens", "1", "--scenarios", "fetch",
         "--gpu-kernels", "torch_scaled_mm", *COMMON[:-2], "--mi-store-lib", "no_such_store"],
        tmp_path / "m",
    )  # fmt: skip
    assert any(e["stage"] == "mi_store" for e in res["errors"])
    assert all("+raw_" in r["variant"] for r in res["rows"]) and res["rows"]


def test_dry_run(ckpts, tmp_path):
    res = _main_ok(
        ["--model-dir", str(ckpts["hy3_fp8"]), "--dry-run", "--scenarios", "gpu",
         "--gpu-kernels", "torch_scaled_mm", *COMMON],
        tmp_path / "dry",
    )  # fmt: skip
    assert res["dry_run"] and {r["tokens"] for r in res["rows"]} == {1, 16}
