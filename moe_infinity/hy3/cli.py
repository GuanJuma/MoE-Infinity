# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Shared engine flags and the ``python -m moe_infinity.hy3`` commands.

    python -m moe_infinity.hy3 inspect  --model-dir /model
    python -m moe_infinity.hy3 plan     --model-dir /model --expert-cache-gib 32
    python -m moe_infinity.hy3 preflight --offload-dir D
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
    pow2 = 1 << (e_b - 1).bit_length()
    cache = expert_cache_gib * 2**30
    out = {
        "resident_bf16_gib": resident / 2**30,
        "resident_params_b": {k: v / 1e9 for k, v in params.items()},
        "kv_bytes_per_token": kv_tok,
        "kv_gib_at_max_context": kv_tok * max_context / 2**30,
        "expert_mib": e_b / 2**20,
        "routed_experts": n_exp,
        "experts_host_gib": n_exp * e_b / 2**30,
        "mi_pinned_host_gib": n_exp * pow2 / 2**30,
        "mi_offload_disk_gib": n_exp * ((e_b + 4095) // 4096 * 4096) / 2**30,
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


def preflight(
    offload_dir: str,
    min_ram_gib: float = 500.0,
    min_disk_gib: float = 300.0,
    min_gpu_free_gib: float = 62.0,
    proc: str = "/proc",
    sysfs: str = "/sys/devices/system/node",
) -> tuple:
    """(ok, lines): host RAM (and the NUMA nodes this process may allocate
    from), free disk for the offload store, free GPU memory."""
    import os
    import shutil
    from pathlib import Path

    from .experts import MANIFEST

    lines, ok = [], True

    def check(cond: bool, msg: str):
        nonlocal ok
        ok &= cond
        lines.append(("OK   " if cond else "FAIL ") + msg)

    mem = _meminfo_kib(f"{proc}/meminfo")
    avail = mem.get("MemAvailable", 0) / 2**20
    check(
        avail >= min_ram_gib,
        f"host MemAvailable {avail:.0f} GiB (need >= {min_ram_gib:.0f})",
    )
    nodes = []
    try:
        for line in open(f"{proc}/self/status"):
            if line.startswith("Mems_allowed_list:"):
                nodes = _parse_list(line.split(":", 1)[1])
    except OSError:
        pass
    node_free = 0.0
    seen = []
    for n in nodes:
        p = f"{sysfs}/node{n}/meminfo"
        if not os.path.exists(p):
            continue
        v = _meminfo_kib(p, prefix=f"Node {n} ")
        free = (v.get("MemFree", 0) + v.get("Inactive(file)", 0)) / 2**20
        node_free += free
        seen.append(f"node{n} {free:.0f}")
    if seen:
        check(
            node_free >= min_ram_gib,
            f"allowed NUMA nodes free+reclaimable {node_free:.0f} GiB "
            f"({', '.join(seen)}; need >= {min_ram_gib:.0f}; pinned memory "
            "comes from these nodes)",
        )
    d = Path(offload_dir)
    built = (d / MANIFEST).exists() and not (d / ".hy3_building").exists()
    probe = d
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free_disk = shutil.disk_usage(probe).free / 2**30
    if built:
        lines.append(
            f"OK   offload store already built in {d} "
            f"({free_disk:.0f} GiB free on that disk)"
        )
    else:
        check(
            free_disk >= min_disk_gib,
            f"free disk at {probe}: {free_disk:.0f} GiB "
            f"(need >= {min_disk_gib:.0f} for the offload store)",
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
        else:
            check(False, "no CUDA device visible")
    except Exception as e:  # noqa: BLE001
        check(False, f"CUDA probe failed: {e!r}")
    return ok, lines


def _cmd_preflight(a):
    ok, lines = preflight(
        a.offload_dir, a.min_ram_gib, a.min_disk_gib, a.min_gpu_free_gib
    )
    for line in lines:
        print(line)
    print(
        "\nNOTE: --store mi pins ~474 GiB of host RAM for the whole run "
        "(released when the process exits) and writes ~267 GiB under the "
        "offload dir. Tell the colleagues sharing this server before you "
        "start, and stop the process when done."
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
    s.add_argument("--min-ram-gib", type=float, default=500.0)
    s.add_argument("--min-disk-gib", type=float, default=300.0)
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
