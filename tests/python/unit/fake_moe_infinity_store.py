# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""CPU stand-in for ``moe_infinity._store.prefetch_handle`` (tests only).

Mirrors the semantics benchmarks/expert_placement/mi_expert_store.py relies
on, as implemented in core/: ``set_topology_v2`` puts every sparse node in a
host buffer and re-points the registered tensors at it
(SetModuleMemoryFromDisk_Views); ``begin`` fetches the node (evicting other
cached experts above the cache limit, RemoveCachedSparseNode) and points the
caller's buffer at the store tensor (SetTensor); ``end`` points the caller's
buffer at a 1-element placeholder (ReleaseTensor); ``prefetch_tensors``
moves the node; ``resize_expert_cache`` evicts down to the target and keeps
it as the cache limit.  "Device" memory is a fresh CPU clone so that tensor
addresses change on every move, like the real device pool.
"""

from __future__ import annotations

import torch

CALLS = []
INSTANCES = []


class _Node:
    def __init__(self, ids, sparse):
        self.ids = ids
        self.sparse = sparse
        self.on_gpu = False
        self.host = {}
        self.nbytes = 0


class prefetch_handle:  # noqa: N801 - mirrors the pybind class name
    def __init__(self, prefix, device_memory_ratio):
        if INSTANCES and not INSTANCES[-1].closed:
            raise RuntimeError("MoE-Infinity supports one model per process")
        self.prefix = prefix
        self.data = {}
        self.reg = {}
        self.nodes = []
        self.node_of = {}
        self.limit = 1 << 40
        self.h2d = 0
        self.closed = False
        self.topology = None
        INSTANCES.append(self)

    def _log(self, *a):
        CALLS.append(a)

    def is_tensor_offloaded(self, tid):
        return tid in self.data

    def offload(self, t, tid):
        self._log("offload", tid)
        assert t.device.type == "cpu" and t.is_contiguous()
        self.data[tid] = t.clone()

    def register(self, t, tid):
        self._log("register", tid)
        assert tid in self.data, "register before offload"
        self.reg[tid] = t

    def set_topology_v2(self, specs):
        self._log("set_topology_v2", len(specs))
        self.topology = specs
        for name, is_sparse, groups, corr in specs:
            assert len(corr) == len(groups)
            assert is_sparse == (len(groups) > 1)
            for ids in groups:
                n = _Node(list(ids), is_sparse)
                for tid in ids:
                    n.host[tid] = self.data[tid].clone()
                    n.nbytes += (self.data[tid].nbytes + 4095) & ~4095
                    self.reg[tid].data = n.host[tid]
                    self.node_of[tid] = n
                self.nodes.append(n)

    def get_expert_cache_limit(self, device_id):
        return self.limit

    def get_expert_h2d_bytes_total(self):
        return self.h2d

    def is_tensor_on_device(self, tid):
        return self.node_of[tid].on_gpu

    def _evict(self, n):
        for tid in n.ids:
            self.reg[tid].data = n.host[tid]
        n.on_gpu = False

    def _fetch(self, n):
        if n.on_gpu:
            return
        resident = [m for m in self.nodes if m.sparse and m.on_gpu]
        size = sum(m.nbytes for m in resident)
        for m in resident:
            if size <= self.limit - n.nbytes:
                break
            self._evict(m)
            size -= m.nbytes
        for tid in n.ids:
            self.reg[tid].data = n.host[tid].clone()
            self.h2d += n.host[tid].nbytes
        n.on_gpu = True

    def begin(self, request_id, buffer, tid):
        self._log("begin", tid)
        n = self.node_of[tid]
        self._fetch(n)
        buffer.data = self.reg[tid]

    def end(self, request_id, buffer, tid):
        self._log("end", tid)
        buffer.data = torch.zeros(1, dtype=buffer.dtype)

    def prefetch_tensors(self, tensor_ids, priority=1, phase=2):
        self._log("prefetch_tensors", tuple(tensor_ids))
        self._fetch(self.node_of[tensor_ids[0]])

    def resize_expert_cache(self, device_id, target_bytes):
        self._log("resize_expert_cache", target_bytes)
        resident = [m for m in self.nodes if m.sparse and m.on_gpu]
        size = sum(m.nbytes for m in resident)
        for m in resident:
            if size <= target_bytes:
                break
            self._evict(m)
            size -= m.nbytes
        self.limit = target_bytes
        return {
            "committed": 1,
            "device_id": device_id,
            "target_bytes": target_bytes,
        }

    def clean_up_resources(self):
        self._log("clean_up_resources")
        self.closed = True
