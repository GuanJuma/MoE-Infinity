# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Numerics of the vendored SGLang CPU MoE kernels (moe_infinity.kernel.cpu).

Skipped unless the host is x86 with AVX512-BF16 and the extension builds
(prebuilt moe_infinity._cpu_moe or JIT from extensions/kernel/cpu).
"""

import pytest

torch = pytest.importorskip("torch")

from moe_infinity.kernel import cpu as cpu_kernels  # noqa: E402

if not cpu_kernels.is_available():
    pytest.skip(
        "CPU MoE kernels unavailable (need x86 AVX512-BF16 + build)",
        allow_module_level=True,
    )

from moe_infinity.kernel.cpu import (  # noqa: E402
    CpuExpertQuant,
    dequant_fp8_block,
    fused_experts,
    pack_experts,
    reference_experts,
)
from moe_infinity.kernel.cpu.expert_ffn import (  # noqa: E402
    _quant_fp8_block,
    _quant_int8_per_channel,
)


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


def _problem(M, E, K, N, topk, seed=0):
    g = torch.Generator().manual_seed(seed)
    w13 = (torch.randn(E, 2 * N, K, generator=g) / K**0.5).bfloat16()
    w2 = (torch.randn(E, K, N, generator=g) / N**0.5).bfloat16()
    x = torch.randn(M, K, generator=g).bfloat16()
    tw, ids = torch.topk(torch.randn(M, E, generator=g).softmax(-1), topk)
    tw = tw / tw.sum(-1, keepdim=True)
    return x, w13, w2, ids.int(), tw.float()


# M * topk / E > 4 switches the BF16 path from the AVX512 micro-kernel to
# the AMX/oneDNN brgemm path, so both are covered.
@pytest.mark.parametrize("M", [1, 3, 17, 40])
@pytest.mark.parametrize("activation", ["silu", "gelu"])
def test_bf16_matches_reference(M, activation):
    x, w13, w2, ids, tw = _problem(M, E=8, K=256, N=128, topk=2)
    packed = pack_experts(w13, w2, "bf16", activation=activation)
    out = fused_experts(x, packed, ids, tw)
    ref = reference_experts(x, w13, w2, ids, tw, activation=activation)
    assert out.shape == x.shape and out.dtype == torch.bfloat16
    assert _rel(out, ref) < 1e-2


@pytest.mark.parametrize("M", [1, 9, 40])
def test_int8_w8a8_matches_dequantized_reference(M):
    x, w13, w2, ids, tw = _problem(M, E=8, K=256, N=128, topk=2, seed=1)
    packed = pack_experts(w13, w2, CpuExpertQuant.INT8_W8A8)
    q13, s13 = _quant_int8_per_channel(w13)
    q2, s2 = _quant_int8_per_channel(w2)
    ref = reference_experts(
        x, q13.float() * s13[..., None], q2.float() * s2[..., None], ids, tw
    )
    # Remaining gap is the dynamic per-token INT8 activation quantization.
    assert _rel(fused_experts(x, packed, ids, tw), ref) < 4e-2


@pytest.mark.parametrize("M", [1, 9, 40])
def test_fp8_w8a16_matches_dequantized_reference(M):
    x, w13, w2, ids, tw = _problem(M, E=8, K=256, N=128, topk=2, seed=2)
    packed = pack_experts(w13, w2, CpuExpertQuant.FP8_W8A16)
    q13, s13 = _quant_fp8_block(w13, (128, 128))
    q2, s2 = _quant_fp8_block(w2, (128, 128))
    ref = reference_experts(
        x,
        dequant_fp8_block(q13, s13, (128, 128)),
        dequant_fp8_block(q2, s2, (128, 128)),
        ids,
        tw,
    )
    assert _rel(fused_experts(x, packed, ids, tw), ref) < 1e-2


def test_packed_bf16_is_same_size_as_input():
    _, w13, w2, _, _ = _problem(1, E=4, K=256, N=128, topk=1)
    packed = pack_experts(w13, w2, "bf16")
    assert packed.nbytes == (w13.numel() + w2.numel()) * 2


def test_shape_validation():
    _, w13, w2, _, _ = _problem(1, E=4, K=256, N=128, topk=1)
    with pytest.raises(ValueError):
        pack_experts(w13, w2[:, :, :64], "bf16")
    with pytest.raises(ValueError):
        pack_experts(w13, w2, "int8_w8a8", activation="gelu")
