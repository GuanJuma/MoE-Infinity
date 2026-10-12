# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Per-expert CPU placement: compute selected experts on the host.

``CpuExpertPlacement`` says which ``(layer, expert)`` pairs run on CPU;
``CpuExpertBackend`` keeps those experts' weights prepacked in host memory and
runs them with the SGLang CPU kernels (``moe_infinity.kernel.cpu``).

``DistributedExpertExecutor.dispatch_local`` calls ``CpuExpertBackend.submit``
before handing the batch to the native GPU dispatcher: CPU-placed experts are
removed from ``router_mask`` / ``router_weights`` (so they are never fetched,
cached or run on GPU) and computed on a background thread, and
``wait_dispatch_local`` adds the CPU partial sum to the GPU result.  Both
paths consume the dense ``router_mask [T, E]`` / ``router_weights [T, E]``
contract the moe-store MoE blocks already use, so no model wrapper changes.
"""

from __future__ import annotations

import concurrent.futures
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence

import torch

from moe_infinity.kernel.cpu import (
    CpuExpertQuant,
    PackedCpuExperts,
    fused_experts,
    pack_experts,
)

_GATE_KEYS = ("gate_proj", "w1")
_UP_KEYS = ("up_proj", "w3")
_DOWN_KEYS = ("down_proj", "w2")


def _parse_ids(text: str, limit: int) -> list[int]:
    ids: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.update(range(int(lo), int(hi) + 1))
        else:
            ids.add(int(part))
    bad = [i for i in ids if i < 0 or i >= limit]
    if bad:
        raise ValueError(f"ids {sorted(bad)} out of range [0, {limit})")
    return sorted(ids)


@dataclass
class CpuExpertPlacement:
    """``layer_id -> sorted expert ids`` computed on CPU."""

    num_experts: int
    cpu_experts: Dict[int, list] = field(default_factory=dict)

    def __post_init__(self):
        self.cpu_experts = {
            int(layer): sorted({int(e) for e in experts})
            for layer, experts in self.cpu_experts.items()
            if experts
        }
        self._masks: Dict[int, torch.Tensor] = {}
        for layer, experts in self.cpu_experts.items():
            if experts and (experts[0] < 0 or experts[-1] >= self.num_experts):
                raise ValueError(
                    f"layer {layer}: expert ids must be in [0, "
                    f"{self.num_experts})"
                )
            mask = torch.zeros(self.num_experts, dtype=torch.bool)
            mask[experts] = True
            self._masks[layer] = mask

    @classmethod
    def from_spec(
        cls, spec: str, layer_ids: Sequence[int], num_experts: int
    ) -> "CpuExpertPlacement":
        """Parse a placement spec.

        ``"all"``                    every expert of every MoE layer
        ``"0-7,12"``                 these expert ids in every MoE layer
        ``"ratio:0.25"``             the highest 25% expert ids per layer
        ``"3:0-7;5:1,2"``            per-layer lists (``layer:ids`` by ``;``)
        """
        spec = (spec or "").strip()
        layers = [int(layer) for layer in layer_ids]
        if not spec:
            return cls(num_experts, {})
        if spec == "all":
            ids = list(range(num_experts))
            return cls(num_experts, {layer: ids for layer in layers})
        if spec.startswith("ratio:"):
            ratio = float(spec.split(":", 1)[1])
            if not 0.0 <= ratio <= 1.0:
                raise ValueError("ratio must be in [0, 1]")
            n = int(round(ratio * num_experts))
            ids = list(range(num_experts - n, num_experts))
            return cls(num_experts, {layer: ids for layer in layers})
        if re.fullmatch(r"[\d,\-\s]+", spec):
            ids = _parse_ids(spec, num_experts)
            return cls(num_experts, {layer: ids for layer in layers})
        table: Dict[int, list] = {}
        for chunk in spec.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            layer, ids = chunk.split(":", 1)
            table[int(layer)] = _parse_ids(ids, num_experts)
        return cls(num_experts, table)

    def experts_for(self, layer_id: int) -> list:
        return self.cpu_experts.get(int(layer_id), [])

    def mask(self, layer_id: int) -> Optional[torch.Tensor]:
        return self._masks.get(int(layer_id))

    def items(self) -> Iterable[tuple[int, int]]:
        for layer, experts in self.cpu_experts.items():
            for e in experts:
                yield layer, e

    def __len__(self) -> int:
        return sum(len(v) for v in self.cpu_experts.values())


@dataclass
class _LayerBank:
    expert_ids: torch.Tensor  # [E_cpu] global ids, int64
    local_of_global: torch.Tensor  # [E] -> local id or -1
    packed: PackedCpuExperts


class CpuExpertBackend:
    """Prepacked CPU experts + a single worker thread that runs them."""

    def __init__(
        self,
        placement: CpuExpertPlacement,
        quant: CpuExpertQuant | str = CpuExpertQuant.BF16,
        num_threads: int = 0,
        activation: str = "silu",
    ):
        self.placement = placement
        self.quant = CpuExpertQuant(quant)
        self.activation = activation
        self.num_threads = int(num_threads) or torch.get_num_threads()
        self._banks: Dict[int, _LayerBank] = {}
        self._pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._pool_lock = threading.Lock()
        self.stats = {"calls": 0, "rows": 0, "skipped_empty": 0}

    # -- weights ---------------------------------------------------------
    def register_layer(
        self,
        layer_id: int,
        weights: Mapping[int, tuple],
    ) -> None:
        """``weights[expert] = (gate [N,K], up [N,K], down [K,N])``.

        Must cover exactly the experts the placement puts on CPU for
        ``layer_id``; tensors may live anywhere and are copied to host.
        """
        experts = self.placement.experts_for(layer_id)
        missing = [e for e in experts if e not in weights]
        if missing:
            raise ValueError(
                f"layer {layer_id}: missing CPU expert weights for {missing}"
            )
        w13 = torch.stack(
            [
                torch.cat([weights[e][0], weights[e][1]], dim=0)
                .detach()
                .to("cpu", torch.bfloat16)
                for e in experts
            ]
        )
        w2 = torch.stack(
            [weights[e][2].detach().to("cpu", torch.bfloat16) for e in experts]
        )
        packed = pack_experts(w13, w2, self.quant, activation=self.activation)
        local = torch.full((self.placement.num_experts,), -1, dtype=torch.int64)
        ids = torch.tensor(experts, dtype=torch.int64)
        local[ids] = torch.arange(len(experts))
        self._banks[int(layer_id)] = _LayerBank(ids, local, packed)

    def load_from_safetensors(
        self,
        ckpt_files: Sequence[str],
        parse_expert_id: Callable[[str], tuple],
        name_filter: Optional[Callable[[str], bool]] = None,
    ) -> int:
        """Read CPU-placed experts from HF safetensors shards.

        ``parse_expert_id(name) -> (layer_id, expert_id)`` (``expert_id``
        ``None`` for non-expert tensors).  Supports BF16/FP16/FP32 experts
        named ``{gate_proj,up_proj,down_proj}`` or Mixtral's ``{w1,w3,w2}``.
        Returns the number of experts loaded.
        """
        from safetensors import safe_open

        wanted = set(self.placement.items())
        found: Dict[int, Dict[int, dict]] = {}
        for path in ckpt_files:
            if not str(path).endswith(".safetensors"):
                continue
            with safe_open(path, framework="pt", device="cpu") as f:
                for name in f.keys():
                    if not name.endswith(".weight"):
                        continue
                    if name_filter is not None and not name_filter(name):
                        continue
                    layer_id, expert_id = parse_expert_id(name)
                    if expert_id is None or (layer_id, expert_id) not in wanted:
                        continue
                    proj = name[: -len(".weight")].rsplit(".", 1)[-1]
                    role = (
                        "gate"
                        if proj in _GATE_KEYS
                        else "up"
                        if proj in _UP_KEYS
                        else "down"
                        if proj in _DOWN_KEYS
                        else None
                    )
                    if role is None:
                        continue
                    t = f.get_tensor(name)
                    if not t.is_floating_point() or t.element_size() == 1:
                        raise NotImplementedError(
                            f"{name}: dtype {t.dtype} is not supported for "
                            "CPU experts (dense BF16/FP16/FP32 only)"
                        )
                    found.setdefault(layer_id, {}).setdefault(expert_id, {})[
                        role
                    ] = t
        count = 0
        for layer_id, experts in found.items():
            weights = {}
            for e, roles in experts.items():
                if set(roles) != {"gate", "up", "down"}:
                    raise ValueError(
                        f"layer {layer_id} expert {e}: incomplete projections "
                        f"{sorted(roles)}"
                    )
                weights[e] = (roles["gate"], roles["up"], roles["down"])
            self.register_layer(layer_id, weights)
            count += len(weights)
        absent = sorted(
            {layer for layer, _ in wanted} - set(self._banks.keys())
        )
        if absent:
            raise ValueError(f"no CPU expert weights found for layers {absent}")
        return count

    def has_layer(self, layer_id: int) -> bool:
        return int(layer_id) in self._banks

    @property
    def host_bytes(self) -> int:
        return sum(b.packed.nbytes for b in self._banks.values())

    # -- compute ---------------------------------------------------------
    def compute(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        router_mask: torch.Tensor,
        router_weights: torch.Tensor,
    ) -> torch.Tensor:
        """CPU experts' weighted contribution, ``[T, H]`` float32 on CPU.

        Inputs are host tensors; ``router_mask``/``router_weights`` are the
        dense ``[T, E]`` routing (entries for GPU experts are ignored).
        """
        bank = self._banks[int(layer_id)]
        T, H = hidden_states.shape
        sub_mask = router_mask[:, bank.expert_ids].bool()
        token_idx, local_idx = sub_mask.nonzero(as_tuple=True)
        out = torch.zeros(T, H, dtype=torch.float32)
        self.stats["calls"] += 1
        if token_idx.numel() == 0:
            self.stats["skipped_empty"] += 1
            return out
        self.stats["rows"] += int(token_idx.numel())
        weights = router_weights[token_idx, bank.expert_ids[local_idx]]
        rows = fused_experts(
            hidden_states.index_select(0, token_idx),
            bank.packed,
            local_idx.view(-1, 1),
            weights.view(-1, 1),
        )
        out.index_add_(0, token_idx, rows.float())
        return out

    def _executor(self) -> concurrent.futures.ThreadPoolExecutor:
        with self._pool_lock:
            if self._pool is None:
                n = self.num_threads
                self._pool = concurrent.futures.ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="moe-cpu-experts",
                    initializer=lambda: torch.set_num_threads(n),
                )
            return self._pool

    def submit(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        router_mask: torch.Tensor,
        router_weights: torch.Tensor,
    ):
        """Split one layer's routing between CPU and GPU.

        Returns ``(gpu_router_mask, gpu_router_weights, future)``; the GPU
        tensors have CPU experts zeroed and ``future.result()`` is the CPU
        contribution as ``[T, H]`` float32 on the host, or ``None`` when no
        token routes to a CPU expert.
        """
        cpu_mask = self.placement.mask(layer_id)
        if cpu_mask is None or not self.has_layer(layer_id):
            return router_mask, router_weights, None
        dev = router_mask.device
        cpu_cols = cpu_mask.to(dev, non_blocking=True)
        routed = router_mask.bool()
        gpu_mask = routed & ~cpu_cols
        gpu_weights = router_weights * gpu_mask.to(router_weights.dtype)
        cpu_part = routed & cpu_cols

        if dev.type == "cuda":
            # Stage on pinned host buffers without blocking the issuing
            # stream; the worker waits on the event before reading them.
            h_host = torch.empty(
                hidden_states.shape,
                dtype=hidden_states.dtype,
                pin_memory=True,
            )
            m_host = torch.empty(
                cpu_part.shape, dtype=torch.bool, pin_memory=True
            )
            w_host = torch.empty(
                router_weights.shape,
                dtype=router_weights.dtype,
                pin_memory=True,
            )
            h_host.copy_(hidden_states, non_blocking=True)
            m_host.copy_(cpu_part, non_blocking=True)
            w_host.copy_(router_weights, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record()
        else:
            h_host, m_host, w_host, ready = (
                hidden_states,
                cpu_part,
                router_weights,
                None,
            )

        def work():
            if ready is not None:
                ready.synchronize()
            if not bool(m_host.any()):
                return None
            return self.compute(layer_id, h_host, m_host, w_host)

        if os.environ.get("MOE_CPU_EXPERTS_SYNC") == "1":
            fut = concurrent.futures.Future()
            fut.set_result(work())
        else:
            fut = self._executor().submit(work)
        return gpu_mask, gpu_weights, fut

    def close(self) -> None:
        with self._pool_lock:
            if self._pool is not None:
                self._pool.shutdown(wait=True)
                self._pool = None


def merge_cpu_result(gpu_result: torch.Tensor, cpu_future) -> torch.Tensor:
    """``gpu_result + cpu contribution`` (no-op when nothing ran on CPU)."""
    if cpu_future is None:
        return gpu_result
    cpu_out = cpu_future.result()
    if cpu_out is None:
        return gpu_result
    return gpu_result + cpu_out.to(gpu_result.device, gpu_result.dtype)
