# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end CPU/GPU expert split through a real MoE block wrapper.

moe-store's SyncOlmoeMoEBlock drives DistributedExpertExecutor exactly as in
serving; CPU experts are loaded from a safetensors shard with HF per-expert
names via moe_infinity.utils.parse_expert_id, the remaining experts run on a
fake GPU dispatcher (CPU tensors), and the block output is compared with a
plain loop over the block's own experts.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("moe_store")
safetensors = pytest.importorskip("safetensors.torch")

from moe_infinity.kernel import cpu as cpu_kernels  # noqa: E402

if not cpu_kernels.is_available():
    pytest.skip(
        "CPU MoE kernels unavailable (need x86 AVX512-BF16 + build)",
        allow_module_level=True,
    )

import torch.nn.functional as F  # noqa: E402
from moe_store.wrappers import SyncOlmoeMoEBlock  # noqa: E402
from transformers import OlmoeConfig  # noqa: E402

from moe_infinity.distributed.expert_executor import (  # noqa: E402
    DistributedExpertExecutor,
)
from moe_infinity.runtime.cpu_experts import (  # noqa: E402
    CpuExpertBackend,
    CpuExpertPlacement,
)
from moe_infinity.utils import ArcherConfig, parse_expert_id  # noqa: E402

LAYER = 1


class _FakeGpuDispatcher:
    def __init__(self, experts):
        self.experts = experts
        self.enqueued = []

    def set_inputs(self, hidden, mask, weights):
        self.hidden, self.mask, self.weights = hidden, mask, weights
        self.enqueued = []

    def set_expected_queue(self, n):
        pass

    def enqueue_expert(self, layer_id, expert_id, gpu_id, remote, phase=2):
        self.enqueued.append(int(expert_id))

    def notify_fetch_start(self):
        pass

    def wait_expert(self):
        out = torch.zeros(self.hidden.shape, dtype=torch.float32)
        for e in self.enqueued:
            rows = self.mask[:, e]
            y = self.experts[e](self.hidden[rows]).float()
            out[rows] += y * self.weights[rows, e, None].float()
        return out


def _reference(block, hidden):
    x = hidden.view(-1, hidden.shape[-1])
    probs = F.softmax(block.gate(x), dim=1, dtype=torch.float)
    w, ids = torch.topk(probs, block.top_k, dim=-1)
    if block.norm_topk_prob:
        w = w / w.sum(-1, keepdim=True)
    out = torch.zeros(x.shape, dtype=torch.float32)
    for t in range(x.shape[0]):
        for k in range(block.top_k):
            e = int(ids[t, k])
            out[t] += (
                w[t, k].float() * block.experts[e](x[t : t + 1])[0].float()
            )
    return out.view(hidden.shape)


@pytest.mark.parametrize("spec", ["0-5", "all"])
def test_olmoe_block_with_cpu_experts(tmp_path, monkeypatch, spec):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    torch.manual_seed(0)
    config = OlmoeConfig(
        hidden_size=256,
        intermediate_size=128,
        num_experts=16,
        num_experts_per_tok=4,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        architectures=["OlmoeForCausalLM"],
    )
    block = SyncOlmoeMoEBlock(config).to(torch.bfloat16).eval()
    block.layer_id = LAYER
    for p in block.parameters():
        torch.nn.init.normal_(p, std=0.05)

    shard = {}
    for e, mlp in enumerate(block.experts):
        prefix = f"model.layers.{LAYER}.mlp.experts.{e}"
        shard[f"{prefix}.gate_proj.weight"] = mlp.gate_proj.weight.detach()
        shard[f"{prefix}.up_proj.weight"] = mlp.up_proj.weight.detach()
        shard[f"{prefix}.down_proj.weight"] = mlp.down_proj.weight.detach()
    shard[f"model.layers.{LAYER}.mlp.gate.weight"] = block.gate.weight.detach()
    path = tmp_path / "model-00001-of-00001.safetensors"
    safetensors.save_file(shard, str(path))

    placement = CpuExpertPlacement.from_spec(spec, [LAYER], config.num_experts)
    backend = CpuExpertBackend(placement)
    loaded = backend.load_from_safetensors(
        [str(path)], lambda name: parse_expert_id(name, config)
    )
    assert loaded == len(placement)

    executor = DistributedExpertExecutor(ArcherConfig(offload_path="unused"))
    fake = _FakeGpuDispatcher(block.experts)
    executor.set_expert_dispatcher(fake)
    executor.set_cpu_expert_backend(backend)
    block.expert_executor = executor

    hidden = torch.randn(2, 5, config.hidden_size).to(torch.bfloat16)
    with torch.no_grad():
        out = block(hidden)
        ref = _reference(block, hidden)

    assert set(fake.enqueued).isdisjoint(placement.experts_for(LAYER))
    rel = (out.float() - ref).norm() / ref.norm()
    assert rel < 2e-2
    backend.close()
