# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Host / GPU / software facts recorded next to every benchmark result."""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch

ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "GOMP_CPU_AFFINITY",
    "KMP_AFFINITY",
    "ONEDNN_MAX_CPU_ISA",
    "DNNL_MAX_CPU_ISA",
    "TORCH_CUDA_ARCH_LIST",
    "TORCH_EXTENSIONS_DIR",
    "MOE_CPU_MOE_JIT",
    "TRITON_CACHE_DIR",
    "SGLANG_MOE_PADDING",
)

GPU_QUERY = (
    "index,name,uuid,pci.bus_id,driver_version,memory.used,memory.total,"
    "utilization.gpu,temperature.gpu,clocks.sm,clocks.max.sm,clocks.mem,"
    "power.draw,power.limit,pcie.link.gen.current,pcie.link.gen.max,"
    "pcie.link.width.current,pcie.link.width.max,compute_mode"
)

COMMANDS = {
    "lscpu": ["lscpu"],
    "numactl_hardware": ["numactl", "--hardware"],
    "numactl_show": ["numactl", "--show"],
    "nvidia_smi": ["nvidia-smi"],
    "nvidia_smi_query": [
        "nvidia-smi",
        f"--query-gpu={GPU_QUERY}",
        "--format=csv",
    ],
    "nvidia_smi_topo": ["nvidia-smi", "topo", "-m"],
    "nvidia_smi_apps": [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv",
    ],
    "free": ["free", "-g"],
    "uname": ["uname", "-a"],
}


def run_cmd(cmd, timeout=20) -> str:
    if shutil.which(cmd[0]) is None:
        return f"<{cmd[0]} not found>"
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.stdout + (
            f"\n<stderr>\n{p.stderr}" if p.stderr.strip() else ""
        )
    except Exception as e:  # noqa: BLE001
        return f"<{' '.join(cmd)} failed: {e!r}>"


def _proc_status() -> Dict[str, str]:
    out = {}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            k, _, v = line.partition(":")
            if k in ("Cpus_allowed_list", "Mems_allowed_list"):
                out[k] = v.strip()
    except OSError:
        pass
    return out


def _versions() -> dict:
    v = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version()
        if torch.backends.cudnn.is_available()
        else None,
    }
    for mod in (
        "triton",
        "sglang",
        "sgl_kernel",
        "flashinfer",
        "transformers",
        "numpy",
    ):
        try:
            m = __import__(mod)
            v[mod] = getattr(m, "__version__", "?")
        except Exception:  # noqa: BLE001
            v[mod] = None
    return v


def gpu_facts(device: torch.device) -> dict:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {"available": False}
    p = torch.cuda.get_device_properties(device)
    free, total = torch.cuda.mem_get_info(device)
    d = {
        "available": True,
        "visible_devices": torch.cuda.device_count(),
        "name": p.name,
        "capability": f"sm_{p.major}{p.minor}",
        "multi_processor_count": p.multi_processor_count,
        "total_memory_mib": total // 2**20,
        "used_memory_mib_at_start": (total - free) // 2**20,
        "l2_cache_bytes": getattr(p, "L2_cache_size", None),
        "uuid": str(getattr(p, "uuid", "")) or None,
        "arch_list": torch.cuda.get_arch_list(),
    }
    return d


def cpu_facts(threads: int) -> dict:
    from_aff = None
    try:
        from_aff = len(os.sched_getaffinity(0))
    except AttributeError:
        pass
    d = {
        "machine": platform.machine(),
        "os_cpu_count": os.cpu_count(),
        "affinity_cpus": from_aff,
        "torch_threads": torch.get_num_threads(),
        "requested_threads": threads,
        "torch_parallel_info": torch.__config__.parallel_info(),
        **_proc_status(),
    }
    try:
        from moe_infinity.kernel.cpu import cpu_isa_flags

        d["isa"] = cpu_isa_flags()
    except Exception as e:  # noqa: BLE001
        d["isa"] = f"<unavailable: {e!r}>"
    l3 = Path("/sys/devices/system/cpu/cpu0/cache/index3/size")
    d["l3_cpu0"] = l3.read_text().strip() if l3.exists() else None
    return d


def l3_bytes() -> Optional[int]:
    p = Path("/sys/devices/system/cpu/cpu0/cache/index3/size")
    if not p.exists():
        return None
    s = p.read_text().strip().upper()
    mult = {"K": 2**10, "M": 2**20, "G": 2**30}.get(s[-1], 1)
    try:
        return int(s.rstrip("KMG")) * mult
    except ValueError:
        return None


def collect(out_dir: Path, device: torch.device, threads: int) -> dict:
    raw_dir = out_dir / "env"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw = {}
    for key, cmd in COMMANDS.items():
        if key.startswith("nvidia") and device.type != "cuda":
            continue
        txt = run_cmd(cmd)
        raw[key] = txt
        (raw_dir / f"{key}.txt").write_text(txt)
    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": platform.node(),
        "argv": sys.argv,
        "env": {
            k: os.environ.get(k)
            for k in ENV_KEYS
            if os.environ.get(k) is not None
        },
        "versions": _versions(),
        "cpu": cpu_facts(threads),
        "gpu": gpu_facts(device),
        "nvidia_smi_query": raw.get("nvidia_smi_query"),
        "numactl_show": raw.get("numactl_show"),
        "in_container": Path("/.dockerenv").exists(),
    }


def other_gpu_processes(device: torch.device) -> Optional[str]:
    """Compute apps on our GPU other than this process (None if unknown)."""
    if device.type != "cuda" or shutil.which("nvidia-smi") is None:
        return None
    txt = run_cmd(COMMANDS["nvidia_smi_apps"])
    me = str(os.getpid())
    uuid = gpu_facts(device).get("uuid") or ""
    lines = []
    for line in txt.splitlines()[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or parts[1] == me:
            continue
        if uuid and uuid.replace("GPU-", "") not in parts[0]:
            continue
        lines.append(line.strip())
    return "\n".join(lines)


def gpu_used_mib(device: torch.device) -> Optional[int]:
    if device.type != "cuda":
        return None
    free, total = torch.cuda.mem_get_info(device)
    return (total - free) // 2**20


def pcie_probe(device: torch.device, sizes, repeats: int = 10) -> list:
    """Raw copy bandwidth (GB/s, 1e9) for pinned/pageable H2D and pinned D2H."""
    if device.type != "cuda":
        return []
    out = []
    for nbytes in sizes:
        nbytes = int(nbytes)
        dev = torch.empty(nbytes, dtype=torch.uint8, device=device)
        for kind in ("h2d_pinned", "h2d_pageable", "d2h_pinned"):
            host = torch.empty(
                nbytes, dtype=torch.uint8, pin_memory=(kind != "h2d_pageable")
            )
            host.fill_(1)
            ms = []
            for i in range(repeats + 2):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                if kind.startswith("h2d"):
                    dev.copy_(host, non_blocking=True)
                else:
                    host.copy_(dev, non_blocking=True)
                torch.cuda.synchronize()
                if i >= 2:
                    ms.append((time.perf_counter() - t0) * 1e3)
            med = statistics.median(ms)
            out.append(
                {
                    "kind": kind,
                    "bytes": nbytes,
                    "median_ms": med,
                    "GBps": nbytes / med / 1e6,
                }
            )
            del host
        del dev
    return out


# x86_64 syscall numbers; numactl --membind uses the same call.
_SYS_SET_MEMPOLICY = 238
_SYS_GET_MEMPOLICY = 239
_MPOL_BIND = 2
_MAX_NODES = 1024


def parse_cpulist(spec: str) -> set:
    cpus = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition("-")
        cpus.update(range(int(a), int(b or a) + 1))
    return cpus


def bind_numa(cpus: Optional[str], node: Optional[int]) -> dict:
    """In-process replacement for ``numactl --physcpubind --membind``.

    Must run before torch starts its thread pools and before large
    allocations (pinned host memory included): affinity and memory policy
    are inherited by threads created afterwards.  Inside Docker,
    ``set_mempolicy`` needs ``--cap-add SYS_NICE``.
    """
    out = {}
    if cpus:
        os.sched_setaffinity(0, parse_cpulist(cpus))
        out["cpus"] = sorted(os.sched_getaffinity(0))
    if node is not None:
        if platform.machine() != "x86_64":
            raise OSError("--membind-node is implemented for x86_64 only")
        libc = ctypes.CDLL(None, use_errno=True)
        words = _MAX_NODES // (8 * ctypes.sizeof(ctypes.c_ulong))
        mask = (ctypes.c_ulong * words)()
        bits = 8 * ctypes.sizeof(ctypes.c_ulong)
        mask[node // bits] |= 1 << (node % bits)
        if (
            libc.syscall(_SYS_SET_MEMPOLICY, _MPOL_BIND, mask, _MAX_NODES + 1)
            != 0
        ):
            err = ctypes.get_errno()
            raise OSError(
                err,
                f"set_mempolicy(MPOL_BIND, node {node}) failed: {os.strerror(err)}",
            )
        mode = ctypes.c_int(-1)
        got = (ctypes.c_ulong * words)()
        if (
            libc.syscall(
                _SYS_GET_MEMPOLICY,
                ctypes.byref(mode),
                got,
                _MAX_NODES + 1,
                None,
                0,
            )
            == 0
        ):
            out["mempolicy"] = {
                "mode": "MPOL_BIND" if mode.value == _MPOL_BIND else mode.value,
                "nodes": [
                    i
                    for i in range(_MAX_NODES)
                    if got[i // bits] >> (i % bits) & 1
                ],
            }
    return out
