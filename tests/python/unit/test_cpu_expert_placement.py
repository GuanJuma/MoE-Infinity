# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""CPU/GPU expert split: placement parsing, CpuExpertBackend numerics, and
the DistributedExpertExecutor seam (with a fake native dispatcher standing in
for the GPU, so this runs on CPU-only hosts)."""

import pytest

torch = pytest.importorskip("torch")

from moe_infinity.kernel import cpu as cpu_kernels  # noqa: E402
from moe_infinity.runtime.cpu_experts import (  # noqa: E402
    CpuExpertBackend,
    CpuExpertPlacement,
    merge_cpu_result,
)

needs_kernels = pytest.mark.skipif(
    not cpu_kernels.is_available(),
    reason="CPU MoE kernels unavailable (need x86 AVX512-BF16 + build)",
)

E, K, N = 8, 256, 128


def test_placement_specs():
    layers = [1, 2]
    p = CpuExpertPlacement.from_spec("all", layers, E)
    assert p.experts_for(1) == list(range(E)) and len(p) == 2 * E
    p = CpuExpertPlacement.from_spec("0-2,6", layers, E)
    assert p.experts_for(2) == [0, 1, 2, 6]
    p = CpuExpertPlacement.from_spec("ratio:0.25", layers, E)
    assert p.experts_for(1) == [6, 7]
    p = CpuExpertPlacement.from_spec("1:0,3;2:5-7", layers, E)
    assert p.experts_for(1) == [0, 3] and p.experts_for(2) == [5, 6, 7]
    assert p.mask(2).tolist() == [False] * 5 + [True] * 3
    assert p.mask(9) is None
    assert len(CpuExpertPlacement.from_spec("", layers, E)) == 0
    with pytest.raises(ValueError):
        CpuExpertPlacement.from_spec("0-8", layers, E)


def _experts(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        e: (
            (torch.randn(N, K, generator=g) / K**0.5).bfloat16(),
            (torch.randn(N, K, generator=g) / K**0.5).bfloat16(),
            (torch.randn(K, N, generator=g) / N**0.5).bfloat16(),
        )
        for e in range(E)
    }


def _expert_fn(w):
    gate, up, down = (t.float() for t in w)

    def fn(x):
        x = x.float()
        return (
            torch.nn.functional.silu(x @ gate.t()) * (x @ up.t())
        ) @ down.t()

    return fn


def _routing(T, topk=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    probs = torch.randn(T, E, generator=g).softmax(-1)
    tw, ids = torch.topk(probs, topk)
    tw = tw / tw.sum(-1, keepdim=True)
    mask = torch.zeros(T, E, dtype=torch.bool).scatter_(1, ids, True)
    weights = torch.zeros(T, E).scatter_(1, ids, tw)
    hidden = torch.randn(T, K, generator=g).bfloat16()
    return hidden, mask, weights


def _reference(hidden, mask, weights, experts):
    out = torch.zeros(hidden.shape, dtype=torch.float32)
    for e, w in experts.items():
        rows = mask[:, e]
        if rows.any():
            out[rows] += _expert_fn(w)(hidden[rows]) * weights[rows, e, None]
    return out


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


@needs_kernels
@pytest.mark.parametrize("T", [1, 5, 33])
def test_backend_computes_only_cpu_experts(T):
    experts = _experts()
    placement = CpuExpertPlacement(E, {3: [1, 4, 6]})
    backend = CpuExpertBackend(placement)
    backend.register_layer(3, {e: experts[e] for e in (1, 4, 6)})
    hidden, mask, weights = _routing(T)

    gpu_mask, gpu_weights, fut = backend.submit(3, hidden, mask, weights)
    assert not gpu_mask[:, [1, 4, 6]].any()
    assert torch.equal(gpu_mask[:, [0, 2, 3, 5, 7]], mask[:, [0, 2, 3, 5, 7]])
    assert torch.all(gpu_weights[:, [1, 4, 6]] == 0)

    cpu_ref = _reference(
        hidden, mask, weights, {e: experts[e] for e in (1, 4, 6)}
    )
    out = fut.result()
    if not mask[:, [1, 4, 6]].any():
        assert out is None
    else:
        assert _rel(out, cpu_ref) < 1e-2
    backend.close()


@needs_kernels
def test_backend_skips_layers_without_cpu_experts():
    backend = CpuExpertBackend(CpuExpertPlacement(E, {0: [1]}))
    backend.register_layer(0, {1: _experts()[1]})
    hidden, mask, weights = _routing(4)
    m, w, fut = backend.submit(5, hidden, mask, weights)
    assert m is mask and w is weights and fut is None
    assert merge_cpu_result(torch.ones(2), None).tolist() == [1.0, 1.0]


@needs_kernels
def test_register_layer_requires_all_placed_experts():
    backend = CpuExpertBackend(CpuExpertPlacement(E, {0: [1, 2]}))
    with pytest.raises(ValueError, match="missing"):
        backend.register_layer(0, {1: _experts()[1]})


@needs_kernels
def test_load_from_safetensors(tmp_path):
    safetensors = pytest.importorskip("safetensors.torch")
    experts = _experts()
    tensors = {}
    for e, (gate, up, down) in experts.items():
        prefix = f"model.layers.2.mlp.experts.{e}"
        tensors[f"{prefix}.gate_proj.weight"] = gate
        tensors[f"{prefix}.up_proj.weight"] = up
        tensors[f"{prefix}.down_proj.weight"] = down
    tensors["model.layers.2.mlp.gate.weight"] = torch.zeros(E, K)
    path = tmp_path / "model.safetensors"
    safetensors.save_file(tensors, str(path))

    def parse(name):
        parts = name.split(".")
        if "experts" in parts:
            return int(parts[2]), int(parts[parts.index("experts") + 1])
        return None, None

    backend = CpuExpertBackend(CpuExpertPlacement(E, {2: [0, 7]}))
    assert backend.load_from_safetensors([str(path)], parse) == 2
    hidden, mask, weights = _routing(16)
    out = backend.compute(2, hidden, mask, weights)
    ref = _reference(hidden, mask, weights, {e: experts[e] for e in (0, 7)})
    assert _rel(out, ref) < 1e-2


class _FakeGpuDispatcher:
    """Native-dispatcher stand-in: runs enqueued experts in fp32 on CPU."""

    def __init__(self, experts):
        self.fns = {e: _expert_fn(w) for e, w in experts.items()}
        self.enqueued = []

    def set_inputs(self, hidden, mask, weights):
        self.hidden, self.mask, self.weights = hidden, mask, weights
        self.enqueued = []

    def set_expected_queue(self, n):
        self.expected = n

    def enqueue_expert(self, layer_id, expert_id, gpu_id, remote, phase=2):
        self.enqueued.append(int(expert_id))

    def notify_fetch_start(self):
        pass

    def wait_expert(self):
        out = torch.zeros(self.hidden.shape, dtype=torch.float32)
        for e in self.enqueued:
            rows = self.mask[:, e]
            out[rows] += (
                self.fns[e](self.hidden[rows]) * self.weights[rows, e, None]
            )
        return out


@needs_kernels
@pytest.mark.parametrize(
    "cpu_ids", [[], [0, 3, 5], list(range(E))], ids=["none", "some", "all"]
)
def test_executor_splits_experts_between_cpu_and_gpu(monkeypatch, cpu_ids):
    pytest.importorskip("transformers")
    from moe_infinity.distributed.expert_executor import (
        DistributedExpertExecutor,
    )
    from moe_infinity.utils import ArcherConfig

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    experts = _experts(seed=3)
    executor = DistributedExpertExecutor(ArcherConfig(offload_path="unused"))
    fake = _FakeGpuDispatcher(experts)
    executor.set_expert_dispatcher(fake)
    backend = CpuExpertBackend(CpuExpertPlacement(E, {7: cpu_ids}))
    if cpu_ids:
        backend.register_layer(7, {e: experts[e] for e in cpu_ids})
    executor.set_cpu_expert_backend(backend)

    hidden, mask, weights = _routing(24, seed=4)
    executor.dispatch_local(7, hidden, mask, weights)
    out = executor.wait_dispatch_local()

    routed = set(torch.nonzero(mask.any(0)).flatten().tolist())
    assert set(fake.enqueued) == routed - set(cpu_ids)
    assert _rel(out, _reference(hidden, mask, weights, experts)) < 1e-2
    if cpu_ids:
        assert backend.stats["calls"] == 1
    backend.close()
