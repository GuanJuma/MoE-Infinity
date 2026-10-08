"""BatchGen expert-FFN kernels: numerics, layout handling and AOT contract.

On a GPU the vendored Triton kernels run natively.  Without one, run with
``TRITON_INTERPRET=1`` so the per-expert decomposition used by the native
launcher is still checked numerically on CPU.  The interpreter mishandles
raw BF16 ``tl.dot`` operands, so BF16 cases only run on CUDA.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import re
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("triton") is None, reason="Triton not installed"
)

ON_GPU = torch.cuda.is_available()
DEVICE = "cuda" if ON_GPU else "cpu"
DTYPES = [torch.float32, torch.float16] + ([torch.bfloat16] if ON_GPU else [])
ATOL = {torch.float32: 1e-4, torch.float16: 2e-2, torch.bfloat16: 6e-2}
_PKG = "moe_infinity.kernel.batchgen"

INTERPRETED = os.environ.get("TRITON_INTERPRET") == "1"


@pytest.fixture(scope="module")
def bg():
    """``(expert_ffn module, vendor fused_moe_bf16 module)``."""
    if not ON_GPU and not INTERPRETED:
        # The interpreter must be on before Triton is first imported.
        pytest.skip("needs CUDA, or TRITON_INTERPRET=1 to run on CPU")
    return (
        importlib.import_module(f"{_PKG}.expert_ffn"),
        importlib.import_module(f"{_PKG}.vendor.fused_moe_bf16"),
    )


def _reference(x, gate, up, down):
    x, gate, up, down = (t.double() for t in (x, gate, up, down))
    return (F.silu(x @ gate.T) * (x @ up.T)) @ down.T


def _expert_weights(n, k, h, dtype, packed):
    w13 = (torch.randn(2 * n, k, device=DEVICE) * 0.1).to(dtype)
    gate, up = w13[:n], w13[n:]
    if not packed:
        gate, up = gate.clone(), up.clone()
    down = (torch.randn(h, n, device=DEVICE) * 0.1).to(dtype)
    return gate, up, down


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize(
    "rows,k,n,h",
    [
        (1, 64, 96, 64),  # decode, single token
        (7, 128, 64, 128),
        (33, 80, 48, 96),  # K % BLOCK_K != 0 -> EVEN_K=False variant
        (1100, 64, 32, 64),  # > 1024 rows -> BatchGen's large M-tile
    ],
)
@pytest.mark.parametrize("packed", [True, False], ids=["w13", "split"])
def test_expert_ffn_matches_reference(bg, dtype, rows, k, n, h, packed):
    expert_ffn, _ = bg
    torch.manual_seed(0)
    x = torch.randn(rows, k, device=DEVICE).to(dtype)
    gate, up, down = _expert_weights(n, k, h, dtype, packed)
    assert expert_ffn.gate_up_packed(gate, up) is packed

    out = expert_ffn.expert_ffn(x, gate, up, down)

    assert out.shape == (rows, h) and out.dtype == dtype
    ref = _reference(x, gate, up, down)
    err = (out.double() - ref).abs().max().item()
    assert err <= ATOL[dtype] * max(1.0, ref.abs().max().item())


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_packed_and_split_gate_up_are_identical(bg, dtype):
    expert_ffn, _ = bg
    torch.manual_seed(1)
    x = torch.randn(9, 64, device=DEVICE).to(dtype)
    gate, up, down = _expert_weights(48, 64, 32, dtype, packed=True)

    packed = expert_ffn.expert_ffn(x, gate, up, down)
    split = expert_ffn.expert_ffn(x, gate.clone(), up.clone(), down)

    assert torch.equal(packed, split)


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_per_expert_dispatch_matches_batchgen_grouped_moe(bg, dtype):
    """MoE-Infinity's per-expert loop == BatchGen's grouped fused_moe_bf16.

    The loop mirrors ExpertDispatcher: gather the tokens routed to an expert,
    run its FFN, scale by the router weight and index_add into an FP32
    accumulator.
    """
    expert_ffn, vendor = bg
    torch.manual_seed(2)
    tokens, k, n, num_experts, top_k = 21, 64, 32, 6, 2
    x = torch.randn(tokens, k, device=DEVICE).to(dtype)
    w13 = (torch.randn(num_experts, 2 * n, k, device=DEVICE) * 0.1).to(dtype)
    w2 = (torch.randn(num_experts, k, n, device=DEVICE) * 0.1).to(dtype)
    probs = torch.softmax(torch.randn(tokens, num_experts, device=DEVICE), -1)
    topk_w, topk_ids = torch.topk(probs, top_k, dim=-1)
    topk_w = topk_w / topk_w.sum(-1, keepdim=True)

    grouped = vendor.fused_moe_bf16(x, w13, w2, topk_w, topk_ids.int())

    router_weight = torch.zeros(tokens, num_experts, device=DEVICE)
    router_weight.scatter_(1, topk_ids, topk_w)
    accum = torch.zeros(tokens, k, device=DEVICE, dtype=torch.float32)
    reference = torch.zeros(tokens, k, dtype=torch.float64, device=DEVICE)
    for e in range(num_experts):
        mask = router_weight[:, e] > 0
        if not mask.any():
            continue
        gate, up = w13[e, :n], w13[e, n:]
        out = expert_ffn.expert_ffn(x[mask], gate, up, w2[e])
        idx = torch.nonzero(mask).squeeze(1)
        weight = router_weight[mask, e].unsqueeze(1)
        accum.index_add_(0, idx, out.float() * weight)
        reference.index_add_(
            0, idx, _reference(x[mask], gate, up, w2[e]) * weight.double()
        )

    scale = max(1.0, reference.abs().max().item())
    assert (accum.double() - reference).abs().max() <= ATOL[dtype] * scale
    assert (grouped.double() - reference).abs().max() <= ATOL[dtype] * scale
    assert (accum.double() - grouped.double()).abs().max() <= (
        2 * ATOL[dtype] * scale
    )


def test_expert_ffn_rejects_bad_shapes(bg):
    expert_ffn, _ = bg
    x = torch.randn(2, 64, device=DEVICE)
    gate, up, down = _expert_weights(32, 64, 64, torch.float32, packed=False)
    with pytest.raises(ValueError):
        expert_ffn.expert_ffn(x[:, :32], gate, up, down)
    with pytest.raises(ValueError):
        expert_ffn.expert_ffn(x, gate, up, down.T)
    with pytest.raises(ValueError):
        expert_ffn.expert_ffn(x[:0], gate, up, down)


def test_aot_param_tables_match_kernel_signatures():
    """The native launcher packs parameters in GEMM_PARAMS/SILU_PARAMS order."""
    aot = importlib.import_module(f"{_PKG}.aot")
    vendor = aot.load_vendor_module()
    for variant in aot.all_variants():
        fn = getattr(vendor, variant.kernel)
        params, _ = aot._params_for(variant)
        runtime_args = [a for a in fn.arg_names if a not in variant.constexprs]
        assert runtime_args == [name for name, _ in params], variant.name


def test_aot_variants_follow_batchgen_configs():
    aot = importlib.import_module(f"{_PKG}.aot")
    vendor = aot.load_vendor_module()
    gemms = {v.name: v for v in aot.gemm_variants()}
    assert set(gemms) == {
        "gemm_small_evenk",
        "gemm_small_oddk",
        "gemm_large_evenk",
        "gemm_large_oddk",
    }
    for label, config in (
        ("small", vendor._CONFIG_SMALL),
        ("large", vendor._CONFIG_LARGE),
    ):
        v = gemms[f"gemm_{label}_evenk"]
        assert (v.block_m, v.block_n, v.block_k) == config[:3]
        assert (v.num_warps, v.num_stages) == config[4:]


def test_aot_header_embeds_cubins(tmp_path):
    """Offline cubin build, run the way setup.py runs it (no GPU needed)."""
    aot = importlib.import_module(f"{_PKG}.aot")
    out = tmp_path / "batchgen_moe_cubins.h"
    env = {k: v for k, v in os.environ.items() if k != "TRITON_INTERPRET"}
    cmd = [sys.executable, aot.__file__, "--archs", "90,80", "--out", str(out)]

    first = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if first.returncode != 0:  # pragma: no cover - toolchain specific
        pytest.skip(f"Triton AOT unavailable: {first.stderr[-500:]}")
    second = subprocess.run(cmd, env=env, capture_output=True, text=True)

    assert "wrote" in first.stdout and "unchanged" in second.stdout
    text = out.read_text()
    assert f"kNumGemmParams = {len(aot.GEMM_PARAMS)};" in text
    assert f"kNumSiluParams = {len(aot.SILU_PARAMS)};" in text
    assert "kArchs[] = {80, 90};" in text
    for variant in aot.all_variants():
        for arch in (80, 90):
            assert f'{{"{variant.name}", {arch}, ' in text


_DUMP_CUBINS = """
import sys
sys.path.insert(0, sys.argv[1])
import aot
for v in aot.all_variants():
    cc = aot.compile_variant(v, 80)
    open(f"{sys.argv[2]}/{v.name}.cubin", "wb").write(cc.asm["cubin"])
    print(v.name, aot.scratch_arg_count(cc.metadata))
"""


def test_cubin_param_abi_matches_native_launcher(tmp_path):
    """Parameter sizes in the cubins == what batchgen_moe.cpp passes.

    The launcher packs ``GEMM_PARAMS`` / ``SILU_PARAMS`` (pointers as
    CUdeviceptr, i32 as int, i64 as int64_t) followed by the scratch pointers.
    """
    triton = pytest.importorskip("triton")
    aot = importlib.import_module(f"{_PKG}.aot")
    cuobjdump = os.path.join(
        os.path.dirname(triton.__file__), "backends/nvidia/bin/cuobjdump"
    )
    if not os.path.exists(cuobjdump):
        pytest.skip("Triton does not ship cuobjdump")
    env = {k: v for k, v in os.environ.items() if k != "TRITON_INTERPRET"}
    dump = subprocess.run(
        [
            sys.executable,
            "-c",
            _DUMP_CUBINS,
            os.path.dirname(aot.__file__),
            str(tmp_path),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if dump.returncode != 0:  # pragma: no cover - toolchain specific
        pytest.skip(f"Triton AOT unavailable: {dump.stderr[-500:]}")
    scratch = {
        line.split()[0]: int(line.split()[1])
        for line in dump.stdout.splitlines()
    }
    size_of = {"i32": 4, "i64": 8}

    for variant in aot.all_variants():
        elf = subprocess.run(
            [cuobjdump, "-elf", str(tmp_path / f"{variant.name}.cubin")],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        params = {
            int(o, 16): int(s, 16)
            for o, s in re.findall(
                r"Ordinal\s*:\s*(0x[0-9a-f]+)\s+Offset\s*:\s*0x[0-9a-f]+"
                r"\s+Size\s*:\s*(0x[0-9a-f]+)",
                elf,
            )
        }
        declared, _ = aot._params_for(variant)
        expected = [8 if t.startswith("*") else size_of[t] for _, t in declared]
        expected += [8] * scratch[variant.name]
        assert [params[i] for i in sorted(params)] == expected, variant.name


def test_archer_config_expert_kernel_validation():
    from moe_infinity.utils.config import ArcherConfig

    assert ArcherConfig(offload_path="/tmp").expert_kernel is None
    cfg = ArcherConfig(offload_path="/tmp", expert_kernel="batchgen")
    assert cfg.expert_kernel == "batchgen"
    with pytest.raises(ValueError, match="expert_kernel"):
        ArcherConfig(offload_path="/tmp", expert_kernel="cutlass")
