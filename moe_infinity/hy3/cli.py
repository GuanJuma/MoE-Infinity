# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Shared engine flags and the ``python -m moe_infinity.hy3`` commands.

    python -m moe_infinity.hy3 inspect  --model-dir /model
    python -m moe_infinity.hy3 plan     --model-dir /model --expert-cache-gib 32
    python -m moe_infinity.hy3 preflight --offload-dir D --model-dir /model
    python -m moe_infinity.hy3 build-store --model-dir /model --offload-dir D
    python -m moe_infinity.hy3 generate --model-dir /model --offload-dir D \
        --prompt "你好" --max-new-tokens 16
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from .engine import Hy3Engine, Hy3EngineConfig

KV_BYTES_PER_TOKEN_NOTE = "2 (K,V) x layers x kv_heads x head_dim x 2 B"


def add_engine_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("Hy3 engine")
    g.add_argument("--model-dir", required=True)
    g.add_argument("--device", default="cuda:0")
    g.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float32"]
    )
    g.add_argument(
        "--store",
        default="mi",
        choices=["mi", "pinned"],
        help="mi = MoE-Infinity native store (default); pinned = PyTorch",
    )
    g.add_argument(
        "--offload-dir",
        default=None,
        help="MoE-Infinity offload store dir (--store mi); built once "
        "(~267 GiB for Hy3-FP8), reused afterwards",
    )
    g.add_argument("--expert-cache-gib", type=float, default=32.0)
    g.add_argument("--device-memory-ratio", type=float, default=0.6)
    g.add_argument("--expert-compute", default="fp8", choices=["fp8", "bf16"])
    g.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    g.add_argument("--prefill-chunk", type=int, default=4096)
    g.add_argument("--max-context", type=int, default=20480)
    g.add_argument("--rebuild-store", action="store_true")
    g.add_argument("--record-weights", action="store_true")


def config_from_args(a) -> Hy3EngineConfig:
    return Hy3EngineConfig(
        model_dir=a.model_dir,
        device=a.device,
        dtype=a.dtype,
        store=a.store,
        offload_dir=a.offload_dir,
        expert_cache_gib=a.expert_cache_gib,
        device_memory_ratio=a.device_memory_ratio,
        expert_compute=a.expert_compute,
        attn_implementation=a.attn,
        prefill_chunk=a.prefill_chunk,
        max_context=a.max_context,
        rebuild_store=a.rebuild_store,
        record_weights=a.record_weights,
    )


AIO_ALIGN = 4096  # kAioAlignment: each tensor in a store node is 4 KiB aligned
PARTITION_GIB = 10.0  # kPartitionSize: topology init reads one file at a time
HOST_HEADROOM_GIB = 16.0  # process, aio staging pool, checkpoint reads


def host_requirements(ck) -> dict:
    """Host RAM / disk from the checkpoint layout.

    ``--store mi``: topology init gives every expert node one exact-size
    ``cudaHostAlloc`` (MoE-Infinity's own HostCachingAllocator, no rounding)
    of its 4 KiB-aligned tensors, plus a transient read buffer of one 10 GiB
    offload partition (ordinary memory).  ``--store pinned``: one exact
    pinned buffer per MoE layer.
    """
    es = torch_element_size(ck.expert_dtype())
    sizes = [r * c * es for r, c in ck.expert_shapes()]
    node = sum((s + AIO_ALIGN - 1) // AIO_ALIGN * AIO_ALIGN for s in sizes)
    n_exp = len(ck.moe_layers) * ck.num_experts
    mi_pinned = n_exp * node / 2**30
    pinned = n_exp * sum(sizes) / 2**30
    return {
        "mi_node_bytes": node,
        "mi_pinned_host_gib": mi_pinned,
        "mi_offload_disk_gib": mi_pinned,
        "mi_host_ram_required_gib": mi_pinned
        + PARTITION_GIB
        + HOST_HEADROOM_GIB,
        "mi_disk_required_gib": mi_pinned * 1.05 + 5,
        "pinned_store_host_gib": pinned,
        "pinned_store_ram_required_gib": pinned + HOST_HEADROOM_GIB,
    }


def torch_element_size(dtype) -> int:
    import torch

    return torch.tensor([], dtype=dtype).element_size()


def memory_plan(
    model_dir: str,
    expert_cache_gib: float,
    max_context: int,
    device_gib: float = 71.5,
) -> dict:
    """Byte budget from config.json + safetensors headers (no weights read)."""
    from .checkpoint import Hy3Checkpoint

    ck = Hy3Checkpoint(model_dir)
    c = ck.config_dict
    H, V = c["hidden_size"], c["vocab_size"]
    nh, kvh = c["num_attention_heads"], c["num_key_value_heads"]
    hd = c.get("head_dim") or H // nh
    L = ck.num_hidden_layers
    n_moe = len(ck.moe_layers)
    n_dense = L - n_moe
    moe_i = c["moe_intermediate_size"]
    shared = c.get("num_shared_experts", 1) * moe_i
    attn = H * (nh * hd) + 2 * H * (kvh * hd) + (nh * hd) * H
    params = {
        "embed+lm_head": 2 * V * H,
        "attention": L * attn,
        "dense_mlp": n_dense * 3 * H * c["intermediate_size"],
        "shared_experts": n_moe * 3 * H * shared,
        "routers": n_moe * ck.num_experts * H,
    }
    resident = sum(params.values()) * 2
    kv_tok = 2 * L * kvh * hd * 2
    e_b = ck.expert_nbytes()
    n_exp = n_moe * ck.num_experts
    cache = expert_cache_gib * 2**30
    host = host_requirements(ck)
    out = {
        "resident_bf16_gib": resident / 2**30,
        "resident_params_b": {k: v / 1e9 for k, v in params.items()},
        "kv_bytes_per_token": kv_tok,
        "kv_gib_at_max_context": kv_tok * max_context / 2**30,
        "expert_mib": e_b / 2**20,
        "routed_experts": n_exp,
        "experts_host_gib": n_exp * e_b / 2**30,
        **host,
        "expert_cache_gib": expert_cache_gib,
        "experts_in_gpu_cache": int(cache // e_b),
        "cache_fraction": int(cache // e_b) / n_exp,
        "workspace_gib_estimate": 4.0,
        "device_gib": device_gib,
    }
    out["device_total_gib_estimate"] = (
        out["resident_bf16_gib"]
        + out["kv_gib_at_max_context"]
        + expert_cache_gib
        + out["workspace_gib_estimate"]
        + 1.5
    )
    out["fits"] = out["device_total_gib_estimate"] <= device_gib
    return out


def _cmd_inspect(a):
    from .checkpoint import Hy3Checkpoint, describe

    ck = Hy3Checkpoint(a.model_dir)
    info = describe(ck, ck.expert_scales() if a.scales else None)
    info["resident_tensors"] = len(ck.resident_names())
    print(json.dumps(info, indent=1, default=str))


def _cmd_plan(a):
    print(
        json.dumps(
            memory_plan(
                a.model_dir, a.expert_cache_gib, a.max_context, a.device_gib
            ),
            indent=1,
        )
    )


def _parse_list(spec: str) -> list:
    out = []
    for part in spec.strip().split(","):
        if not part:
            continue
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def _meminfo_kib(path: str, prefix: str = "") -> dict:
    vals = {}
    with open(path) as f:
        for line in f:
            line = line[len(prefix) :] if line.startswith(prefix) else line
            key, _, rest = line.partition(":")
            parts = rest.split()
            if parts and parts[0].isdigit():
                vals[key.strip()] = int(parts[0])
    return vals


def _gpu_numa_nodes() -> dict:
    """{cuda index: NUMA node} from the PCI address (best effort)."""
    out = {}
    try:
        import torch

        for i in range(torch.cuda.device_count()):
            pr = torch.cuda.get_device_properties(i)
            addr = (
                f"{getattr(pr, 'pci_domain_id', 0):04x}:"
                f"{pr.pci_bus_id:02x}:{pr.pci_device_id:02x}.0"
            )
            with open(f"/sys/bus/pci/devices/{addr}/numa_node") as f:
                out[i] = int(f.read().strip())
    except Exception:  # noqa: BLE001
        pass
    return out


def preflight(
    offload_dir: str,
    model_dir: str | None = None,
    store: str = "mi",
    min_ram_gib: float | None = None,
    min_disk_gib: float | None = None,
    min_gpu_free_gib: float = 62.0,
    proc: str = "/proc",
    sysfs: str = "/sys/devices/system/node",
    gpu_numa: dict | None = None,
) -> tuple:
    """(ok, lines): host RAM -- in total and on the NUMA nodes this process
    may allocate pinned memory from -- free disk for the offload store and
    free GPU memory.  Requirements come from the checkpoint layout
    (:func:`host_requirements`) when ``model_dir`` is given."""
    import os
    import shutil
    from pathlib import Path

    from .experts import MANIFEST

    lines, ok = [], True

    def check(cond: bool, msg: str):
        nonlocal ok
        ok &= cond
        lines.append(("OK   " if cond else "FAIL ") + msg)

    ram_req, disk_req = 300.0, 300.0 if store == "mi" else 0.0
    if model_dir:
        from .checkpoint import Hy3Checkpoint

        req = host_requirements(Hy3Checkpoint(model_dir))
        if store == "mi":
            ram_req = req["mi_host_ram_required_gib"]
            disk_req = req["mi_disk_required_gib"]
            lines.append(
                f"INFO --store mi: pinned host pool {req['mi_pinned_host_gib']:.1f}"
                f" GiB (exact per-node cudaHostAlloc) + {PARTITION_GIB:.0f} GiB "
                f"transient partition read + {HOST_HEADROOM_GIB:.0f} GiB "
                f"headroom = {ram_req:.0f} GiB; offload files "
                f"{req['mi_offload_disk_gib']:.1f} GiB"
            )
        else:
            ram_req = req["pinned_store_ram_required_gib"]
            lines.append(
                f"INFO --store pinned: {req['pinned_store_host_gib']:.1f} GiB "
                f"pinned + {HOST_HEADROOM_GIB:.0f} GiB headroom; no offload files"
            )
    if min_ram_gib is not None:
        ram_req = min_ram_gib
    if min_disk_gib is not None:
        disk_req = min_disk_gib

    mem = _meminfo_kib(f"{proc}/meminfo")
    avail = mem.get("MemAvailable", 0) / 2**20
    check(
        avail >= ram_req,
        f"host MemAvailable {avail:.0f} GiB (need >= {ram_req:.0f})",
    )
    nodes = []
    try:
        for line in open(f"{proc}/self/status"):
            if line.startswith("Mems_allowed_list:"):
                nodes = _parse_list(line.split(":", 1)[1])
    except OSError:
        pass
    free_by_node = {}
    for n in nodes:
        p = f"{sysfs}/node{n}/meminfo"
        if not os.path.exists(p):
            continue
        v = _meminfo_kib(p, prefix=f"Node {n} ")
        free_by_node[n] = (
            v.get("MemFree", 0) + v.get("Inactive(file)", 0)
        ) / 2**20
    if free_by_node:
        total = sum(free_by_node.values())
        per = ", ".join(f"node{n} {g:.0f}" for n, g in free_by_node.items())
        check(
            total >= ram_req,
            f"allowed NUMA nodes free+reclaimable {total:.0f} GiB ({per}; "
            f"need >= {ram_req:.0f}; pinned memory comes from these nodes)",
        )
        gpu_numa = _gpu_numa_nodes() if gpu_numa is None else gpu_numa
        gpu_nodes = sorted(set(gpu_numa.values()))
        if len(free_by_node) > 1:
            local = gpu_nodes[0] if len(gpu_nodes) == 1 else None
            msg = (
                f"cpuset allows NUMA nodes {sorted(free_by_node)}: pinned "
                "expert memory is first-touched on the node the process runs "
                "on and spills to the others when it fills up; experts held "
                "on a node other than the GPU's are copied over the "
                "inter-socket link (lower and noisier H2D bandwidth)"
            )
            if local is not None and local in free_by_node:
                lf = free_by_node[local]
                if lf >= ram_req:
                    msg += (
                        f". GPU is on node {local}, which alone has {lf:.0f} "
                        f"GiB: --cpuset-mems={local} would keep it local"
                    )
                else:
                    msg += (
                        f". GPU is on node {local} ({lf:.0f} GiB free): about "
                        f"{ram_req - lf:.0f} GiB will live on a remote node"
                    )
            lines.append("WARN " + msg)
    d = Path(offload_dir)
    built = (d / MANIFEST).exists() and not (d / ".hy3_building").exists()
    probe = d
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free_disk = shutil.disk_usage(probe).free / 2**30
    if store != "mi":
        lines.append(f"OK   --store pinned writes no offload files")
    elif built:
        lines.append(
            f"OK   offload store already built in {d} "
            f"({free_disk:.0f} GiB free on that disk)"
        )
    else:
        check(
            free_disk >= disk_req,
            f"free disk at {probe}: {free_disk:.0f} GiB "
            f"(need >= {disk_req:.0f} for the offload store)",
        )
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                check(
                    free / 2**30 >= min_gpu_free_gib,
                    f"cuda:{i} {torch.cuda.get_device_name(i)} free "
                    f"{free / 2**30:.1f} / {total / 2**30:.1f} GiB "
                    f"(need >= {min_gpu_free_gib:.0f})",
                )
        elif min_gpu_free_gib > 0:
            check(False, "no CUDA device visible")
    except Exception as e:  # noqa: BLE001
        check(False, f"CUDA probe failed: {e!r}")
    return ok, lines


def _cmd_preflight(a):
    ok, lines = preflight(
        a.offload_dir,
        model_dir=a.model_dir,
        store=a.store,
        min_ram_gib=a.min_ram_gib,
        min_disk_gib=a.min_disk_gib,
        min_gpu_free_gib=a.min_gpu_free_gib,
    )
    for line in lines:
        print(line)
    if a.store == "mi":
        print(
            "\nNOTE: --store mi pins the whole expert set (~267 GiB for "
            "Hy3-FP8) in host RAM for the run (released when the process "
            "exits) and keeps ~267 GiB of offload files under the offload "
            "dir. Tell the colleagues sharing this server before you start, "
            "and stop the process when done."
        )
    print("\nPREFLIGHT " + ("PASSED" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


def _cmd_build(a):
    import torch

    from .checkpoint import Hy3Checkpoint
    from .experts import MiExpertStore, exit_process

    ck = Hy3Checkpoint(a.model_dir)
    t0 = time.perf_counter()
    MiExpertStore(
        ck,
        a.offload_dir,
        torch.device(a.device),
        int(a.expert_cache_gib * 2**30),
        device_memory_ratio=a.device_memory_ratio,
        rebuild=a.rebuild_store,
    )
    print(f"[hy3] store ready in {time.perf_counter() - t0:.0f}s")
    exit_process(0)


def _cmd_generate(a):
    from .experts import exit_process

    eng = Hy3Engine(config_from_args(a))
    print(json.dumps(eng.memory_report(), indent=1))
    msgs = [{"role": "user", "content": a.prompt}]
    kw = json.loads(a.chat_template_kwargs) if a.chat_template_kwargs else {}
    ids = eng.encode_chat(msgs, kw)
    for i in range(a.repeat):
        r = eng.generate(ids, max_new_tokens=a.max_new_tokens, trace=False)
        text = eng.tokenizer.decode(r["output_ids"], skip_special_tokens=True)
        print(f"--- run {i}: {text!r}")
        print(json.dumps(r["stats"], indent=1))
    exit_process(0)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m moe_infinity.hy3")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("inspect", help="checkpoint summary (headers only)")
    s.add_argument("--model-dir", required=True)
    s.add_argument(
        "--scales",
        action="store_true",
        help="also read all expert scales (~1 min)",
    )
    s.set_defaults(fn=_cmd_inspect)
    s = sub.add_parser("plan", help="GPU / host / disk memory plan")
    s.add_argument("--model-dir", required=True)
    s.add_argument("--expert-cache-gib", type=float, default=32.0)
    s.add_argument("--max-context", type=int, default=20480)
    s.add_argument("--device-gib", type=float, default=71.5)
    s.set_defaults(fn=_cmd_plan)
    s = sub.add_parser(
        "preflight", help="host RAM / disk / GPU checks before the store"
    )
    s.add_argument("--offload-dir", required=True)
    s.add_argument(
        "--model-dir",
        default=None,
        help="derive RAM/disk needs from this checkpoint",
    )
    s.add_argument("--store", default="mi", choices=["mi", "pinned"])
    s.add_argument("--min-ram-gib", type=float, default=None)
    s.add_argument("--min-disk-gib", type=float, default=None)
    s.add_argument("--min-gpu-free-gib", type=float, default=62.0)
    s.set_defaults(fn=_cmd_preflight)
    s = sub.add_parser("build-store", help="write the MoE-Infinity store")
    add_engine_args(s)
    s.set_defaults(fn=_cmd_build)
    s = sub.add_parser("generate", help="one chat prompt (smoke test)")
    add_engine_args(s)
    s.add_argument("--prompt", default="用一句话介绍你自己。")
    s.add_argument("--max-new-tokens", type=int, default=32)
    s.add_argument("--repeat", type=int, default=2)
    s.add_argument(
        "--chat-template-kwargs",
        default='{"reasoning_effort": "no_think"}',
    )
    s.set_defaults(fn=_cmd_generate)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
