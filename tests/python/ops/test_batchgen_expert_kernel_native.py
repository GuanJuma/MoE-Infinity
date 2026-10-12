"""Native BatchGen expert kernels vs the default CUTLASS expert FFN.

``batchgen_expert_ffn`` / ``default_expert_ffn`` run exactly the kernels that
``MoEMLP::ForwardHelper`` dispatches to, so this is the GPU parity check for
``MOE_EXPERT_KERNEL=batchgen``.
"""

from __future__ import annotations

import importlib

import pytest
import torch
import torch.nn.functional as F

from tests.python.ops.conftest import requires_cuda


def _store():
    try:
        store = importlib.import_module("moe_infinity._store")
    except Exception as exc:  # pragma: no cover - build specific
        pytest.skip(f"moe_infinity._store not importable: {exc}")
    if not hasattr(store, "batchgen_expert_ffn"):
        pytest.skip("_store built without BatchGen bindings")
    if not store.batchgen_expert_kernel_stats()["compiled_in"]:
        pytest.skip("_store built with MOE_BUILD_BATCHGEN=0")
    return store


def _reference(x, gate, up, down):
    x, gate, up, down = (t.float() for t in (x, gate, up, down))
    return (F.silu(x @ gate.T) * (x @ up.T)) @ down.T


def _weights(n, k, h, packed):
    w13 = torch.randn(2 * n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    gate, up = w13[:n], w13[n:]
    if not packed:
        gate, up = gate.clone(), up.clone()
    down = torch.randn(h, n, device="cuda", dtype=torch.bfloat16) * 0.02
    return gate, up, down


@requires_cuda
@pytest.mark.parametrize(
    "k,n",
    [(2048, 1408), (2048, 768), (4096, 14336)],
    ids=["deepseek-v2-lite", "qwen3-30b-a3b", "mixtral-8x7b"],
)
@pytest.mark.parametrize("rows", [1, 3, 64, 1500])
@pytest.mark.parametrize("packed", [True, False], ids=["w13", "split"])
def test_batchgen_matches_default_expert_ffn(
    seed_everything, k, n, rows, packed
):
    store = _store()
    gate, up, down = _weights(n, k, k, packed)
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    store.reset_batchgen_expert_kernel_stats()

    batchgen = store.batchgen_expert_ffn(x, gate, up, down)
    default = store.default_expert_ffn(x, gate, up, down)
    torch.cuda.synchronize()

    stats = store.batchgen_expert_kernel_stats()
    assert stats["expert_calls"] == 1 and stats["fallback_calls"] == 0
    assert stats["packed_gate_up_calls"] == int(packed)
    ref = _reference(x, gate, up, down)
    scale = ref.abs().max().item()
    assert (batchgen.float() - ref).abs().max().item() <= 2e-2 * scale
    assert (batchgen.float() - default.float()).abs().max().item() <= (
        3e-2 * scale
    )


@requires_cuda
def test_unsupported_shape_reports_fallback(seed_everything):
    store = _store()
    gate, up, down = _weights(40, 72, 72, packed=False)  # 72 % 16 != 0
    x = torch.randn(4, 72, device="cuda", dtype=torch.bfloat16)
    store.reset_batchgen_expert_kernel_stats()

    with pytest.raises(RuntimeError, match="multiples of 16"):
        store.batchgen_expert_ffn(x, gate, up, down)

    stats = store.batchgen_expert_kernel_stats()
    assert stats["fallback_calls"] == 1 and stats["expert_calls"] == 0


@requires_cuda
def test_expert_kernel_switch_round_trip():
    store = _store()
    previous = store.get_expert_kernel()
    try:
        store.set_expert_kernel("batchgen")
        assert store.get_expert_kernel() == "batchgen"
        store.set_expert_kernel("default")
        assert store.get_expert_kernel() == "default"
        with pytest.raises(ValueError):
            store.set_expert_kernel("cutlass")
    finally:
        store.set_expert_kernel(previous)
