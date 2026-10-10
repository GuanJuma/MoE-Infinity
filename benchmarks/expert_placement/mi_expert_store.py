# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""The pinned expert inside MoE-Infinity's own store, cache and fetch path.

Builds a one-MoE-layer MoE-Infinity engine (``moe_infinity._store``) the
same way ``OffloadEngine`` does for a model: tensors are written to the
offload store (``prefetch_handle.offload``), placeholder tensors are
registered (``register``), and ``set_topology_v2`` lays out one dense stage,
one sparse (expert) stage and a closing dense stage.  Topology init reads
every sparse node into MoE-Infinity's pinned host pool (``kHostMemoryPool``,
one contiguous buffer per node) and re-points the registered tensors at it,
which is exactly where offloaded experts live at runtime.

Expert nodes:

* ``gpu``: the expert in the GPU kernel's layout (FP8 ``w13 = [gate; up]``,
  FP8 ``w2``, float32 scales); moved CPU->GPU by MoE-Infinity;
* ``other``: a second expert of the same layer, used to fill the expert
  cache for the "miss with eviction" case;
* ``cpu`` (optional): the same FP8 bytes in the CPU kernel's VNNI-packed
  layout.  It stays in the pinned host pool; the CPU kernel reads it there.

Loads go through the runtime's code:

* ``fetch``  -- ``prefetch_handle.begin`` (``AcquireTensor``): sparse-cache
  bookkeeping and eviction (``RemoveCachedSparseNode``), an on-demand task
  on ``ArcherTaskPool``'s GPU thread, ``Node::SetDevice`` (device pool
  allocation, ``cudaMemcpyAsync`` from the pinned host buffer on the H2D
  stream, event sync, tensor re-pointing).  Returns once the weights are on
  the GPU; ``end`` releases the node like the post-forward hook.
* ``prefetch`` -- ``prefetch_tensors`` (``EnqueuePrefetchTensors``, the
  expert prefetch API), then wait until the node is resident.
* eviction -- ``resize_expert_cache(device, 0)`` (``ReserveSparseCacheVictims``
  + ``CommitSparseCacheReservation`` -> ``Node::SetDevice(host)``), done
  outside the timed region.

The store-registered tensors are re-pointed in place (``set_data``) on every
move, so callers keep using the same Python tensor objects; views derived
from them must not be cached across moves.
"""

from __future__ import annotations

import importlib
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch

# Methods of moe_infinity._store.prefetch_handle this module calls; checked
# against core/python/*.cpp by the unit tests.
USED_BINDINGS = (
    "offload",
    "register",
    "is_tensor_offloaded",
    "set_topology_v2",
    "begin",
    "end",
    "prefetch_tensors",
    "is_tensor_on_device",
    "resize_expert_cache",
    "get_expert_cache_limit",
    "get_expert_h2d_bytes_total",
    "clean_up_resources",
)


class StoreUnavailable(RuntimeError):
    pass


class StoreError(RuntimeError):
    """A call would trip a native DLOG_FATAL (abort()); raised instead."""


def load_store_lib(name: str = "moe_infinity._store"):
    try:
        return importlib.import_module(name)
    except Exception as e:  # noqa: BLE001
        raise StoreUnavailable(
            f"moe_infinity._store is not importable ({e!r}); build MoE-Infinity's "
            f"CUDA extension (see docs/expert-placement-benchmark.md)"
        )


@dataclass
class NodeHandle:
    name: str
    tensor_ids: List[int]
    tensors: Dict[str, torch.Tensor]  # store-registered, track node memory
    params: Dict[str, torch.Tensor]  # module-parameter stand-ins for begin/end
    nbytes: int
    order: List[str] = field(default_factory=list)
    aligned_bytes: int = (
        0  # node byte_size: tensors 4 KiB-aligned (kAioAlignment)
    )


class MoEInfinityExpertStore:
    def __init__(
        self,
        offload_dir: str | Path,
        nodes: Dict[str, Dict[str, torch.Tensor]],
        device_id: int = 0,
        device_memory_ratio: float = 0.5,
        lib=None,
        keep_dir: bool = False,
    ):
        self.lib = lib or load_store_lib()
        self.dir = Path(offload_dir)
        self.keep_dir = keep_dir
        if self.dir.exists():
            shutil.rmtree(self.dir)
        self.dir.mkdir(parents=True)
        self.device_id = device_id
        t0 = time.perf_counter()
        self.h = self.lib.prefetch_handle(str(self.dir), device_memory_ratio)
        self.nodes: Dict[str, NodeHandle] = {}
        # Python mirror of ArcherTensorHandle::tensor_to_id_ (data_ptr -> id).
        self._ptr_ids: Dict[int, int] = {}
        self.broken: Optional[str] = None
        tid = 0

        def put(t: torch.Tensor):
            nonlocal tid
            t = t.detach().contiguous().cpu()
            if not self.h.is_tensor_offloaded(tid):
                self.h.offload(t, tid)
            # Exactly what OffloadEngine does: register ``param.data`` (an
            # alias that shares param's data_ptr and that the store re-points)
            # and pass ``param`` itself to begin/end.  begin/end look the
            # buffer up by data_ptr in tensor_to_id_ (UpdateTensorMap) and
            # abort() if it is unknown, so param must start at the registered
            # pointer.
            param = torch.zeros(1, dtype=t.dtype)
            reg = param.data
            self.h.register(reg, tid)
            self._ptr_ids.setdefault(param.data_ptr(), tid)
            tid += 1
            return tid - 1, reg, param

        dense_in = put(torch.zeros(1024, dtype=torch.bfloat16))[0]
        groups = []
        for name, tensors in nodes.items():
            ids, regs, params, nbytes, aligned = [], {}, {}, 0, 0
            for k, t in tensors.items():
                i, reg, param = put(t)
                ids.append(i)
                regs[k] = reg
                params[k] = param
                b = t.numel() * t.element_size()
                nbytes += b
                aligned += (b + 4095) & ~4095
            self.nodes[name] = NodeHandle(
                name, ids, regs, params, nbytes, list(tensors), aligned
            )
            groups.append(ids)
        if len(groups) < 2:
            raise ValueError("a sparse stage needs at least two expert nodes")
        dense_out = put(torch.zeros(1024, dtype=torch.bfloat16))[0]
        self.offload_ms = (time.perf_counter() - t0) * 1e3

        build_topology_specs = _runtime_topology_specs()

        topo = [
            ("model.embed_tokens", [[dense_in]]),
            ("model.layers.0.mlp.experts", groups),
            ("lm_head", [[dense_out]]),
        ]
        t0 = time.perf_counter()
        self.h.set_topology_v2(build_topology_specs(topo))
        self.topology_ms = (time.perf_counter() - t0) * 1e3
        self.request_id = 0
        self.default_limit = int(self.h.get_expert_cache_limit(device_id))
        for n in self.nodes.values():
            for t in n.tensors.values():
                if t.device.type != "cpu":
                    raise RuntimeError(
                        f"node {n.name} not in host memory after init"
                    )

    # -- state --------------------------------------------------------------

    def tensors(self, name: str) -> Dict[str, torch.Tensor]:
        return self.nodes[name].tensors

    def on_gpu(self, name: str) -> bool:
        return bool(self.h.is_tensor_on_device(self.nodes[name].tensor_ids[0]))

    def is_pinned(self, name: str) -> bool:
        t = next(iter(self.nodes[name].tensors.values()))
        return t.device.type == "cpu" and t.is_pinned()

    def h2d_bytes_total(self) -> int:
        return int(self.h.get_expert_h2d_bytes_total())

    # -- moves (the runtime's own entry points) ----------------------------------

    def _hook(self, fn, param: torch.Tensor, tid: int) -> None:
        """Call begin/end the way the forward hooks do, keeping the
        tensor_to_id_ mirror; refuse calls the engine would abort() on."""
        if self.broken:
            raise StoreError(f"MoE-Infinity store unusable: {self.broken}")
        old = param.data_ptr()
        if self._ptr_ids.get(old) != tid:
            self.broken = (
                f"tensor {tid}: buffer data_ptr {old:#x} is not registered for it in "
                f"tensor_to_id_ (would hit DLOG_FATAL in UpdateTensorMap)"
            )
            raise StoreError(self.broken)
        fn(self.request_id, param, tid)
        self._ptr_ids.pop(old, None)
        self._ptr_ids[param.data_ptr()] = tid

    def fetch(self, name: str) -> None:
        """Demand fetch via the pre-forward hook path (``begin``)."""
        n = self.nodes[name]
        for k, tid in zip(n.order, n.tensor_ids):
            self._hook(self.h.begin, n.params[k], tid)

    def release(self, name: str) -> None:
        """Post-forward hook (``end``); the node stays cached on the GPU."""
        n = self.nodes[name]
        for k, tid in zip(n.order, n.tensor_ids):
            self._hook(self.h.end, n.params[k], tid)
        self.request_id += 1

    def prefetch(self, name: str, timeout_s: float = 10.0) -> None:
        n = self.nodes[name]
        self.h.prefetch_tensors(list(n.tensor_ids))
        deadline = time.perf_counter() + timeout_s
        while not self.h.is_tensor_on_device(n.tensor_ids[0]):
            if time.perf_counter() > deadline:
                raise TimeoutError(f"prefetch of {name} did not complete")

    def evict_all(self) -> None:
        r = self.h.resize_expert_cache(self.device_id, 0)
        if "committed" not in r:
            raise RuntimeError(f"expert cache eviction rejected: {dict(r)}")

    def set_cache_limit(self, nbytes: Optional[int]) -> None:
        r = self.h.resize_expert_cache(
            self.device_id,
            self.default_limit if nbytes is None else int(nbytes),
        )
        if "committed" not in r:
            raise RuntimeError(f"expert cache resize rejected: {r}")

    def close(self):
        try:
            self.h.clean_up_resources()
        finally:
            if not self.keep_dir:
                shutil.rmtree(self.dir, ignore_errors=True)


def _runtime_topology_specs():
    """``moe_infinity.utils.topology.build_topology_specs`` (the function
    OffloadEngine feeds to ``set_topology_v2``), loaded from its file so the
    rest of ``moe_infinity.utils`` and its imports are not needed."""
    import importlib.util

    path = (
        Path(__file__).resolve().parents[2]
        / "moe_infinity"
        / "utils"
        / "topology.py"
    )
    spec = importlib.util.spec_from_file_location("_mi_topology", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_topology_specs


def offload_dir_default(out_dir: Path) -> Path:
    return Path(os.environ.get("MOE_EP_STORE_DIR", str(out_dir / "mi_store")))
