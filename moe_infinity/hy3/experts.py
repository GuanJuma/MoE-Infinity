# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Routed-expert storage, GPU expert cache and expert FFN kernels for Hy3.

Each routed expert is one store node holding the checkpoint's exact
``gate_proj``/``up_proj``/``down_proj`` weights (FP8 e4m3 for Hy3-FP8,
18 MiB per expert).  Scales stay in small device tables.

Stores (same interface: ``prefetch`` -> ``acquire`` -> ``release``):

* :class:`MiExpertStore` -- ``moe_infinity._store``: the tensors are written
  to MoE-Infinity's offload files once (reused on later runs; an interrupted
  build starts over), topology init reads every expert node into the pinned
  host pool (one exact-size ``cudaHostAlloc`` per node), ``begin`` fetches a
  node to the GPU through ``AcquireTensor``/``ArcherTaskPool``/
  ``Node::SetDevice`` (evicting other experts when the cache is full),
  ``prefetch_tensors`` enqueues async fetches, ``end`` releases.
* :class:`PinnedExpertStore` -- pinned host buffers per MoE layer, a fixed
  pool of GPU slots with LRU eviction and copies on a side stream.

Kernels (:class:`ExpertFFN`):

* ``fp8`` -- ``torch._scaled_mm`` (cuBLASLt FP8) with the checkpoint's static
  per-tensor input scales, gate and up as separate GEMMs with their own
  weight scales (no requantization);
* ``bf16`` -- dequantize the expert on the device, BF16 ``F.linear``.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import shutil
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .checkpoint import PROJS, ExpertScales, Hy3Checkpoint

FP8_MAX = 448.0
MANIFEST = "hy3_store_manifest.json"
LAYOUT_VERSION = 1


# --------------------------------------------------------------------------
# expert FFN
# --------------------------------------------------------------------------


def _quant_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)


def _dyn_scale(x: torch.Tensor) -> torch.Tensor:
    return (x.abs().amax().float() / FP8_MAX).clamp(min=1e-12).reshape(())


def _scaled_mm(a, b_t, sa, sb, out_dtype):
    if a.is_cuda:
        return torch._scaled_mm(
            a, b_t, scale_a=sa, scale_b=sb, out_dtype=out_dtype
        )
    # CPU (tests/debug): oneDNN lacks most FP8 GEMM shapes; same math.
    return ((a.float() * sa) @ (b_t.float() * sb)).to(out_dtype)


class ExpertFFN:
    """``down(silu(gate(x)) * up(x))`` for one expert."""

    def __init__(self, mode: str, scales: ExpertScales, device, dtype):
        if mode not in ("fp8", "bf16"):
            raise ValueError(f"unknown expert compute mode {mode!r}")
        self.mode = mode
        self.dtype = dtype
        self.wscale = scales.weight.to(device)
        self.iscale = scales.input.to(device)
        self.static = bool((scales.input > 0).all())
        # Host-side so the hot path never syncs on a device comparison.
        self.same_gu = (scales.input[..., 0] == scales.input[..., 1]).tolist()

    def __call__(self, li: int, e: int, x: torch.Tensor, w) -> torch.Tensor:
        g, u, d = w
        ws = self.wscale[li, e]
        if self.mode == "bf16" or g.dtype != torch.float8_e4m3fn:
            if g.dtype == torch.float8_e4m3fn:
                g = (g.float() * ws[0]).to(self.dtype)
                u = (u.float() * ws[1]).to(self.dtype)
                d = (d.float() * ws[2]).to(self.dtype)
            x = x.to(g.dtype)
            return F.linear(F.silu(F.linear(x, g)) * F.linear(x, u), d)
        ins = self.iscale[li, e]
        sg = ins[0] if self.static else _dyn_scale(x)
        su = ins[1] if self.static else sg
        xg = _quant_fp8(x, sg)
        reuse = self.static and self.same_gu[li][e]
        xu = xg if reuse else _quant_fp8(x, su)
        hg = _scaled_mm(xg, g.t(), sg, ws[0], self.dtype)
        hu = _scaled_mm(xu, u.t(), su, ws[1], self.dtype)
        h = F.silu(hg) * hu
        sd = ins[2] if self.static else _dyn_scale(h)
        return _scaled_mm(_quant_fp8(h, sd), d.t(), sd, ws[2], self.dtype)


# --------------------------------------------------------------------------
# stores
# --------------------------------------------------------------------------


class ExpertStoreBase:
    kind = "base"

    def __init__(self):
        self.phase = "prefill"
        self.counts = {"prefill": [0, 0], "decode": [0, 0]}  # hits, misses

    def _count(self, hit: bool) -> None:
        c = self.counts.setdefault(self.phase, [0, 0])
        c[0 if hit else 1] += 1

    def stats(self) -> dict:
        out = {"store": self.kind}
        for ph, (h, m) in self.counts.items():
            out[f"{ph}_hits"] = h
            out[f"{ph}_misses"] = m
            out[f"{ph}_hit_rate"] = h / (h + m) if h + m else None
        out["h2d_bytes"] = self.h2d_bytes()
        return out

    def reset_stats(self) -> None:
        self.counts = {"prefill": [0, 0], "decode": [0, 0]}

    def h2d_bytes(self) -> int:
        return 0

    def prefetch(self, li: int, experts: Sequence[int]) -> None:
        raise NotImplementedError

    def acquire(self, li: int, e: int):
        raise NotImplementedError

    def release(self, li: int, e: int) -> None:
        raise NotImplementedError


class PinnedExpertStore(ExpertStoreBase):
    kind = "pinned"

    def __init__(
        self,
        ckpt: Hy3Checkpoint,
        device: torch.device,
        cache_bytes: int,
        log=print,
    ):
        super().__init__()
        self.device = torch.device(device)
        cuda = self.device.type == "cuda"
        self.L, self.E = len(ckpt.moe_layers), ckpt.num_experts
        self.dtype = ckpt.expert_dtype()
        self.shapes = ckpt.expert_shapes()
        es = torch.tensor([], dtype=self.dtype).element_size()
        self.sizes = [r * c * es for r, c in self.shapes]
        self.offsets = [0, self.sizes[0], self.sizes[0] + self.sizes[1]]
        self.nbytes = sum(self.sizes)
        self.n_slots = max(1, int(cache_bytes // self.nbytes))
        if self.n_slots < self.E:
            raise ValueError(
                f"expert cache {cache_bytes / 2**30:.1f} GiB holds "
                f"{self.n_slots} experts; one MoE layer needs up to {self.E}"
            )
        t0 = time.perf_counter()
        self.host = [
            torch.empty(
                (self.E, self.nbytes), dtype=torch.uint8, pin_memory=cuda
            )
            for _ in range(self.L)
        ]
        names = ckpt.expert_weight_names()
        for n, (li, e, p, name) in enumerate(names):
            t = ckpt.get(name)
            o = self.offsets[p]
            self.host[li][e, o : o + self.sizes[p]].copy_(
                t.contiguous().view(torch.uint8).reshape(-1)
            )
            if n % 3000 == 0:
                log(
                    f"[hy3] pinned store: {n}/{len(names)} expert tensors "
                    f"({time.perf_counter() - t0:.0f}s)"
                )
        self.slots = torch.empty(
            (self.n_slots, self.nbytes), dtype=torch.uint8, device=self.device
        )
        self.lru: "OrderedDict[Tuple[int, int], int]" = OrderedDict()
        self.free = list(range(self.n_slots))
        self.inflight: Dict[Tuple[int, int], object] = {}
        self.slot_free_ev: Dict[int, object] = {}
        self.protected: set = set()
        self.copy_stream = torch.cuda.Stream(self.device) if cuda else None
        self._h2d = 0
        log(
            f"[hy3] pinned store ready: {self.L}x{self.E} experts, "
            f"{self.L * self.E * self.nbytes / 2**30:.1f} GiB host, "
            f"{self.n_slots} GPU slots ({self.n_slots * self.nbytes / 2**30:.1f}"
            f" GiB), {time.perf_counter() - t0:.0f}s"
        )

    def h2d_bytes(self) -> int:
        return self._h2d

    def resident(self, li: int, e: int) -> bool:
        return (li, e) in self.lru

    def _victim(self) -> int:
        for key, slot in self.lru.items():
            if key not in self.protected:
                del self.lru[key]
                self.inflight.pop(key, None)
                return slot
        raise RuntimeError("expert cache: every slot is in use")

    def _ensure(self, key) -> int:
        slot = self.lru.get(key)
        if slot is not None:
            self.lru.move_to_end(key)
            return slot
        slot = self.free.pop() if self.free else self._victim()
        li, e = key
        if self.copy_stream is not None:
            ev_free = self.slot_free_ev.pop(slot, None)
            with torch.cuda.stream(self.copy_stream):
                if ev_free is not None:
                    self.copy_stream.wait_event(ev_free)
                self.slots[slot].copy_(self.host[li][e], non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(self.copy_stream)
            self.inflight[key] = ev
        else:
            self.slots[slot].copy_(self.host[li][e])
        self._h2d += self.nbytes
        self.lru[key] = slot
        return slot

    def prefetch(self, li: int, experts: Sequence[int]) -> None:
        self.protected = {(li, e) for e in experts}
        for e in experts:
            self._count((li, e) in self.lru)
            self._ensure((li, e))

    def acquire(self, li: int, e: int):
        key = (li, e)
        slot = self._ensure(key)
        ev = self.inflight.pop(key, None)
        if ev is not None:
            torch.cuda.current_stream(self.device).wait_event(ev)
        buf = self.slots[slot]
        return tuple(
            buf[o : o + n].view(self.dtype).view(shape)
            for o, n, shape in zip(self.offsets, self.sizes, self.shapes)
        )

    def release(self, li: int, e: int) -> None:
        if self.copy_stream is None:
            return
        slot = self.lru.get((li, e))
        if slot is not None:
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream(self.device))
            self.slot_free_ev[slot] = ev


def _load_topology_specs():
    """``moe_infinity/utils/topology.py`` by path (``moe_infinity.utils``
    imports moe-store, which this module does not need)."""
    path = Path(__file__).resolve().parents[1] / "utils" / "topology.py"
    spec = importlib.util.spec_from_file_location("_hy3_topology", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_topology_specs


class MiExpertStore(ExpertStoreBase):
    """Routed experts in MoE-Infinity's native store (``_store``).

    Only one native engine may exist per process and its teardown can
    segfault (see benchmarks/expert_placement/mi_expert_store.py); callers
    leave with ``os._exit`` after writing their outputs.
    """

    kind = "mi"

    def __init__(
        self,
        ckpt: Hy3Checkpoint,
        offload_dir: str | Path,
        device: torch.device,
        cache_bytes: int,
        device_memory_ratio: float = 0.6,
        rebuild: bool = False,
        lib=None,
        log=print,
    ):
        super().__init__()
        self.dir = Path(offload_dir)
        self.device = torch.device(device)
        self.device_id = self.device.index or 0
        self.L, self.E = len(ckpt.moe_layers), ckpt.num_experts
        self.nbytes = ckpt.expert_nbytes()
        self.dtype = ckpt.expert_dtype()
        manifest = {
            "layout_version": LAYOUT_VERSION,
            "fingerprint": ckpt.fingerprint(),
            "model_dir": str(ckpt.model_dir),
            "moe_layers": ckpt.moe_layers,
            "num_experts": self.E,
            "expert_dtype": str(self.dtype),
        }
        mpath = self.dir / MANIFEST
        old = json.loads(mpath.read_text()) if mpath.exists() else None
        has_files = self.dir.exists() and any(self.dir.iterdir())
        if rebuild and self.dir.exists():
            shutil.rmtree(self.dir)
            old, has_files = None, False
        if has_files and old is not None and old != manifest:
            raise RuntimeError(
                f"{self.dir} holds an expert store for another checkpoint "
                f"({old.get('model_dir')}, {old.get('fingerprint')}); "
                "use another --offload-dir or pass --rebuild-store"
            )
        if (
            has_files
            and old is None
            and not (self.dir / ".hy3_building").exists()
        ):
            raise RuntimeError(
                f"{self.dir} is not empty and is not a Hy3 expert store; "
                "use an empty --offload-dir"
            )
        building = self.dir / ".hy3_building"
        complete = old == manifest and not building.exists()
        if has_files and not complete:
            # Interrupted build.  The native store restarts its write cursor
            # at partition 0 offset 0 when it reopens an index, so appending
            # the missing tensors would overwrite stored ones: start over.
            log(f"[hy3] mi store: {self.dir} is an unfinished build; wiping")
            shutil.rmtree(self.dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        if lib is None:
            try:
                lib = importlib.import_module("moe_infinity._store")
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    f"moe_infinity._store is not importable ({e!r}): build it "
                    "(benchmarks/expert_placement/offline_setup.sh) and put "
                    "the repo on PYTHONPATH, or use --store pinned"
                )
        self.lib = lib
        self.h = self.lib.prefetch_handle(str(self.dir), device_memory_ratio)

        self.tid_in = 0
        self.tid_out = 1 + self.L * self.E * 3
        t0 = time.perf_counter()
        if complete and not (
            self.h.is_tensor_offloaded(self.tid_out)
            and self.h.is_tensor_offloaded(self.tid(self.L - 1, self.E - 1, 2))
        ):
            raise RuntimeError(
                f"{self.dir}: manifest present but the store index is "
                "incomplete; pass --rebuild-store"
            )
        if not complete:
            building.touch()
            self._build(ckpt, log)
            mpath.write_text(json.dumps(manifest, indent=1))
            building.unlink(missing_ok=True)
            log(
                f"[hy3] mi store: offload done ({time.perf_counter() - t0:.0f}s)"
            )
        else:
            log(f"[hy3] mi store: reusing {self.dir}")

        self.params: Dict[int, torch.Tensor] = {}
        self.regs: Dict[int, torch.Tensor] = {}
        self._ptr_ids: Dict[int, int] = {}
        for tid in range(self.tid_out + 1):
            dtype = (
                torch.bfloat16
                if tid in (self.tid_in, self.tid_out)
                else (self.dtype)
            )
            param = torch.zeros(1, dtype=dtype)
            reg = param.data
            self.h.register(reg, tid)
            self._ptr_ids.setdefault(param.data_ptr(), tid)
            self.params[tid], self.regs[tid] = param, reg
        topo = [("model.embed_tokens", [[self.tid_in]])]
        for li, layer in enumerate(ckpt.moe_layers):
            groups = [
                [self.tid(li, e, p) for p in range(3)] for e in range(self.E)
            ]
            topo.append((f"model.layers.{layer}.mlp.experts", groups))
        topo.append(("lm_head", [[self.tid_out]]))
        t1 = time.perf_counter()
        log(
            f"[hy3] mi store: topology init, reading {self.L * self.E} expert "
            "nodes into the pinned host pool ..."
        )
        self.h.set_topology_v2(_load_topology_specs()(topo))
        log(f"[hy3] mi store: topology ready ({time.perf_counter() - t1:.0f}s)")
        r = self.h.resize_expert_cache(self.device_id, int(cache_bytes))
        if "committed" not in r:
            raise RuntimeError(f"expert cache resize rejected: {dict(r)}")
        self.cache_bytes = int(cache_bytes)
        self.request_id = 0

    def tid(self, li: int, e: int, p: int) -> int:
        return 1 + (li * self.E + e) * 3 + p

    def _build(self, ckpt: Hy3Checkpoint, log) -> None:
        if not self.h.is_tensor_offloaded(self.tid_in):
            self.h.offload(torch.zeros(1024, dtype=torch.bfloat16), self.tid_in)
        names = ckpt.expert_weight_names()
        t0 = time.perf_counter()
        for n, (li, e, p, name) in enumerate(names):
            tid = self.tid(li, e, p)
            if not self.h.is_tensor_offloaded(tid):
                self.h.offload(ckpt.get(name).contiguous(), tid)
            if n % 3000 == 0:
                log(
                    f"[hy3] mi store: offloaded {n}/{len(names)} expert "
                    f"tensors ({time.perf_counter() - t0:.0f}s)"
                )
        if not self.h.is_tensor_offloaded(self.tid_out):
            self.h.offload(
                torch.zeros(1024, dtype=torch.bfloat16), self.tid_out
            )

    def h2d_bytes(self) -> int:
        return int(self.h.get_expert_h2d_bytes_total())

    def resident(self, li: int, e: int) -> bool:
        return bool(self.h.is_tensor_on_device(self.tid(li, e, 0)))

    def _hook(self, fn, tid: int) -> None:
        param = self.params[tid]
        old = param.data_ptr()
        if self._ptr_ids.get(old) != tid:
            raise RuntimeError(
                f"tensor {tid}: data_ptr {old:#x} not registered "
                "(the native engine would abort in UpdateTensorMap)"
            )
        fn(self.request_id, param, tid)
        self._ptr_ids.pop(old, None)
        self._ptr_ids[param.data_ptr()] = tid

    def prefetch(self, li: int, experts: Sequence[int]) -> None:
        missing = []
        for e in experts:
            hit = self.resident(li, e)
            self._count(hit)
            if not hit:
                missing.append(self.tid(li, e, 0))
        if missing:
            self.h.prefetch_tensors(missing)

    def acquire(self, li: int, e: int):
        tids = [self.tid(li, e, p) for p in range(3)]
        for tid in tids:
            self._hook(self.h.begin, tid)
        return tuple(self.regs[t] for t in tids)

    def release(self, li: int, e: int) -> None:
        for p in range(3):
            self._hook(self.h.end, self.tid(li, e, p))
        self.request_id += 1


def build_store(
    kind: str,
    ckpt: Hy3Checkpoint,
    device: torch.device,
    cache_bytes: int,
    offload_dir: Optional[str] = None,
    device_memory_ratio: float = 0.6,
    rebuild: bool = False,
    log=print,
    lib=None,
) -> ExpertStoreBase:
    if kind == "pinned":
        return PinnedExpertStore(ckpt, device, cache_bytes, log=log)
    if kind == "mi":
        if not offload_dir:
            raise ValueError("--store mi needs --offload-dir")
        return MiExpertStore(
            ckpt,
            offload_dir,
            device,
            cache_bytes,
            device_memory_ratio=device_memory_ratio,
            rebuild=rebuild,
            lib=lib,
            log=log,
        )
    raise ValueError(f"unknown expert store {kind!r}")


def exit_process(code: int = 0) -> None:
    """Leave without native destructors (MoE-Infinity engine teardown)."""
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def expert_counts(topk_ids: torch.Tensor, num_experts: int) -> List[int]:
    return torch.bincount(topk_ids.reshape(-1), minlength=num_experts).tolist()


__all__ = [
    "PROJS",
    "ExpertFFN",
    "MiExpertStore",
    "PinnedExpertStore",
    "build_store",
    "exit_process",
]
