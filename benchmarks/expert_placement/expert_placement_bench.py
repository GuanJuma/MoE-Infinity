# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""One routed expert, three placements, a token sweep.

Pins one ``(layer, expert)`` of a HuggingFace MoE checkpoint (built for
Hy3-FP8; any gate/up/down expert layout works) and measures, for every token
count M (= rows routed to this expert):

  gpu_resident           weights live in GPU memory, computed on the GPU
  cpu_compute            weights live in host memory, computed on the CPU
  cpu_store_gpu_compute  weights live in host memory, copied to the GPU
                         (pinned and pageable sources), computed on the GPU

Per-scenario split (all medians over repeats, milliseconds):

  gpu_resident
    compute  = GPU kernel time: CUDA events around a CUDA-graph replay of the
               expert (gate/up GEMM, activation, FP8 casts, down GEMM) --
               pure device time without launch gaps
    total    = host wall time of one eager call, inputs already on the GPU,
               from the call to torch.cuda.synchronize() returning
    overhead = total - compute  (Python dispatch, kernel launches, sync)
  cpu_compute   (hidden states start on the GPU, result must end there)
    d2h      = activations GPU -> pinned host buffer + stream sync
    compute  = CPU kernel wall time (result written in place into the buffer)
    h2d      = result pinned buffer -> GPU + stream sync
    total    = d2h + compute + h2d (one wall-clock span)
    overhead = total - compute
  cpu_store_gpu_compute
    xfer     = H2D copy of the expert's GPU-ready weight tensors into
               preallocated GPU buffers (CUDA events around the copies)
    compute  = GPU kernel time right after the copy (graph replay, events)
    total    = host wall time of copy + eager call + sync
    overhead = total - compute  (= xfer + launch/dispatch)

One-time costs (checkpoint read, weight layout prep, CPU packing, GPU upload,
pinned allocation) are reported separately under ``setup``.

Caches: before every timed repeat the GPU L2 and the CPU LLC are flushed by
writing a buffer of 2x their size (outside the timed region), so weights are
streamed from HBM / DRAM as in a real forward pass where other layers ran in
between.  Disable with ``--gpu-l2-flush-mb 0 --cpu-llc-flush-mb 0``.

Run ``--list-experts`` first to see what the checkpoint holds, ``--dry-run``
to build every kernel and check numerics without timing.  See
docs/expert-placement-benchmark.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parents[1]))

import bench_env  # noqa: E402
import torch  # noqa: E402
from checkpoint_expert import (  # noqa: E402
    CheckpointError,
    SafetensorsIndex,
    config_summary,
    discover_experts,
    inspect_checkpoint,
    load_config,
    load_expert,
)
from expert_runners import (  # noqa: E402
    CPU_KERNELS,
    GPU_KERNELS,
    KernelUnavailable,
    build_cpu_runner,
    build_gpu_runner,
    expert_flops,
    fmt_bytes,
    reference_expert,
    rel_err,
    resolve_gpu_kernels,
)

DEFAULT_TOKENS = "1,2,3,4,8,16,32,64,128,256,512,1024,2048,4096,8192"
SCENARIO_ALIASES = {
    "gpu": "gpu_resident",
    "gpu_resident": "gpu_resident",
    "cpu": "cpu_compute",
    "cpu_compute": "cpu_compute",
    "fetch": "cpu_store_gpu_compute",
    "cpu_store_gpu_compute": "cpu_store_gpu_compute",
}
CSV_FIELDS = [
    "scenario",
    "variant",
    "kernel",
    "host_mem",
    "tokens",
    "total_ms",
    "compute_ms",
    "overhead_ms",
    "xfer_ms",
    "d2h_ms",
    "h2d_ms",
    "launch_ms",
    "total_p10_ms",
    "total_p90_ms",
    "compute_p10_ms",
    "compute_p90_ms",
    "xfer_p10_ms",
    "xfer_p90_ms",
    "eager_gpu_span_ms",
    "xfer_bytes",
    "xfer_GBps",
    "compute_TFLOPS",
    "pipelined_ms",
    "repeats",
    "kernel_timing",
    "rel_err",
    "note",
]


# --------------------------------------------------------------------------
# timing helpers
# --------------------------------------------------------------------------


class HostEvent:
    """CPU stand-in for torch.cuda.Event (debug runs with --device cpu)."""

    def record(self, stream=None):
        self.t = time.perf_counter()

    def elapsed_time(self, other) -> float:
        return (other.t - self.t) * 1e3


class Clock:
    def __init__(self, device: torch.device):
        self.cuda = device.type == "cuda"

    def sync(self):
        if self.cuda:
            torch.cuda.synchronize()

    def stream_sync(self):
        if self.cuda:
            torch.cuda.current_stream().synchronize()

    def events(self, n):
        if self.cuda:
            return [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        return [HostEvent() for _ in range(n)]


class Flusher:
    def __init__(self, device, gpu_mb: float, cpu_mb: float):
        self.gpu = None
        self.cpu = None
        if device.type == "cuda" and gpu_mb > 0:
            self.gpu = torch.empty(
                int(gpu_mb * 2**20) // 4, dtype=torch.float32, device=device
            )
        if cpu_mb > 0:
            self.cpu = torch.zeros(
                int(cpu_mb * 2**20) // 4, dtype=torch.float32
            )

    def flush_gpu(self):
        if self.gpu is not None:
            self.gpu.zero_()

    def flush_cpu(self):
        if self.cpu is not None:
            self.cpu.add_(1.0)


def pct(xs, q):
    s = sorted(xs)
    if not s:
        return float("nan")
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(samples, key):
    xs = [s[key] for s in samples if key in s]
    if not xs:
        return None, None, None
    return statistics.median(xs), pct(xs, 0.10), pct(xs, 0.90)


def run_repeats(one, args, budget_s=None):
    budget_s = args.max_seconds_per_point if budget_s is None else budget_s
    for i in range(max(1, args.warmup)):
        t = time.perf_counter()
        one()
        if i >= 1 and time.perf_counter() - t > budget_s / 4:
            break
    samples, start = [], time.perf_counter()
    while len(samples) < args.repeats:
        samples.append(one())
        if (
            len(samples) >= args.min_repeats
            and time.perf_counter() - start > budget_s
        ):
            break
    return samples


def try_graph(fn, reset, clock: Clock, mode: str):
    """Capture ``fn`` into a CUDA graph (None if not possible / disabled)."""
    if not clock.cuda or mode != "graph":
        return None, "eager_events"
    try:
        if reset:
            reset()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                if reset:
                    reset()
                fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        if reset:
            reset()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        return g, "cuda_graph"
    except Exception as e:  # noqa: BLE001
        torch.cuda.synchronize()
        return None, f"eager_events (graph capture failed: {type(e).__name__})"


# --------------------------------------------------------------------------
# scenarios
# --------------------------------------------------------------------------


class Bench:
    def __init__(self, args, ew, device):
        self.args = args
        self.ew = ew
        self.device = device
        self.clock = Clock(device)
        self.K = ew.hidden_size
        self.N = ew.intermediate_size
        self.dtype = ew.act_dtype
        l2 = None
        if device.type == "cuda":
            l2 = getattr(
                torch.cuda.get_device_properties(device), "L2_cache_size", None
            )
        gpu_mb = args.gpu_l2_flush_mb
        if gpu_mb < 0:
            gpu_mb = max(256.0, 2 * (l2 or 128 * 2**20) / 2**20)
        cpu_mb = args.cpu_llc_flush_mb
        if cpu_mb < 0:
            cpu_mb = max(
                512.0, 2 * (bench_env.l3_bytes() or 256 * 2**20) / 2**20
            )
        self.flush_sizes = {
            "gpu_l2_flush_mb": gpu_mb,
            "cpu_llc_flush_mb": cpu_mb,
        }
        self.flusher = Flusher(device, gpu_mb, cpu_mb)
        a1, _ = ew.static_input_scales()
        if args.x_std == "auto":
            # Keep |x| inside the static FP8 activation range like real
            # (normalized) hidden states: max of ~3e7 N(0,1) samples < 6.
            self.x_std = a1 * 448.0 / 6.0 if a1 else 1.0
        else:
            self.x_std = float(args.x_std)
        self._ref_cache = {}

    # -- inputs and numerics ------------------------------------------------

    def make_x(self, M, device):
        g = torch.Generator().manual_seed(self.args.seed + M)
        x = torch.randn(M, self.K, generator=g) * self.x_std
        return x.to(self.dtype).to(device).contiguous()

    def check(self, out, x, M):
        if M > self.args.check_max_tokens:
            return None
        if M not in self._ref_cache:
            dev = (
                self.device
                if self.device.type == "cuda"
                else torch.device("cpu")
            )
            self._ref_cache[M] = reference_expert(x.to(dev), self.ew)
        ref = self._ref_cache[M]
        return rel_err(out.to(ref.device), ref)

    def _row(self, scenario, variant, kernel, host_mem, M, **kw):
        row = {k: None for k in CSV_FIELDS}
        row.update(
            scenario=scenario,
            variant=variant,
            kernel=kernel,
            host_mem=host_mem,
            tokens=M,
        )
        row.update(kw)
        if row.get("compute_ms"):
            row["compute_TFLOPS"] = (
                expert_flops(M, self.K, self.N)
                / (row["compute_ms"] * 1e-3)
                / 1e12
            )
        if row.get("xfer_ms") and row.get("xfer_bytes"):
            row["xfer_GBps"] = row["xfer_bytes"] / (row["xfer_ms"] * 1e-3) / 1e9
        return row

    def _log(self, row):
        parts = [
            f"[{row['scenario']}/{row['variant']}] M={row['tokens']:>5}",
            f"total {row['total_ms']:.4f} ms = compute {row['compute_ms']:.4f}"
            f" + overhead {row['overhead_ms']:.4f}",
        ]
        if row.get("xfer_ms") is not None:
            parts.append(
                f"(xfer {row['xfer_ms']:.4f}, {row['xfer_GBps'] or 0:.1f} GB/s)"
            )
        if row.get("d2h_ms") is not None:
            parts.append(f"(d2h {row['d2h_ms']:.4f} + h2d {row['h2d_ms']:.4f})")
        if row.get("rel_err") is not None:
            parts.append(f"err {row['rel_err']:.2e}")
        parts.append(f"n={row['repeats']}")
        print("  ".join(parts), flush=True)

    # -- scenario 1 -----------------------------------------------------------

    def gpu_resident(self, runner, M):
        c, fl = self.clock, self.flusher
        x = self.make_x(M, self.device)
        fn, reset = runner.bind(x)
        if reset:
            reset()
        out = fn()
        c.sync()
        err = self.check(out, x, M)
        graph, timing = try_graph(fn, reset, c, self.args.gpu_timing)

        def it_kernel():
            e0, e1 = c.events(2)
            if reset:
                reset()
            fl.flush_gpu()
            e0.record()
            graph.replay() if graph is not None else fn()
            e1.record()
            c.sync()
            return {"kernel": e0.elapsed_time(e1)}

        def it_eager():
            e0, e1 = c.events(2)
            if reset:
                reset()
            fl.flush_gpu()
            c.sync()
            t0 = time.perf_counter()
            e0.record()
            fn()
            e1.record()
            c.sync()
            return {
                "total": (time.perf_counter() - t0) * 1e3,
                "span": e0.elapsed_time(e1),
            }

        sk = run_repeats(it_kernel, self.args)
        se = run_repeats(it_eager, self.args)
        k50, k10, k90 = summarize(sk, "kernel")
        t50, t10, t90 = summarize(se, "total")
        row = self._row(
            "gpu_resident",
            runner.kernel,
            runner.kernel,
            "gpu",
            M,
            total_ms=t50,
            compute_ms=k50,
            overhead_ms=t50 - k50,
            launch_ms=t50 - k50,
            total_p10_ms=t10,
            total_p90_ms=t90,
            compute_p10_ms=k10,
            compute_p90_ms=k90,
            eager_gpu_span_ms=summarize(se, "span")[0],
            repeats=min(len(sk), len(se)),
            kernel_timing=timing,
            rel_err=err,
        )
        del graph
        return row

    # -- scenario 2 -----------------------------------------------------------

    def cpu_compute(self, runner, M):
        c, fl = self.clock, self.flusher
        x_dev = self.make_x(M, self.device)
        pinned = self.args.act_host_mem == "pinned" and c.cuda
        x_host = torch.empty(M, self.K, dtype=self.dtype, pin_memory=pinned)
        out_dev = torch.empty_like(x_dev)
        fn, _ = runner.bind(x_host)
        x_host.copy_(x_dev)
        out = fn()
        err = self.check(out, x_dev, M)

        def it():
            fl.flush_cpu()
            c.sync()
            t0 = time.perf_counter()
            x_host.copy_(x_dev, non_blocking=pinned)
            c.stream_sync()
            t1 = time.perf_counter()
            fn()
            t2 = time.perf_counter()
            out_dev.copy_(x_host, non_blocking=pinned)
            c.stream_sync()
            t3 = time.perf_counter()
            return {
                "d2h": (t1 - t0) * 1e3,
                "compute": (t2 - t1) * 1e3,
                "h2d": (t3 - t2) * 1e3,
                "total": (t3 - t0) * 1e3,
            }

        s = run_repeats(it, self.args)
        t50, t10, t90 = summarize(s, "total")
        k50, k10, k90 = summarize(s, "compute")
        d2h = summarize(s, "d2h")[0]
        h2d = summarize(s, "h2d")[0]
        act_bytes = 2 * M * self.K * x_host.element_size()
        return self._row(
            "cpu_compute",
            runner.kernel,
            runner.kernel,
            self.args.act_host_mem,
            M,
            total_ms=t50,
            compute_ms=k50,
            overhead_ms=t50 - k50,
            xfer_ms=d2h + h2d,
            d2h_ms=d2h,
            h2d_ms=h2d,
            total_p10_ms=t10,
            total_p90_ms=t90,
            compute_p10_ms=k10,
            compute_p90_ms=k90,
            xfer_bytes=act_bytes,
            repeats=len(s),
            kernel_timing="host_wall",
            rel_err=err,
        )

    # -- scenario 3 -----------------------------------------------------------

    def cpu_store_gpu_compute(self, runner, host, host_mem, M):
        c, fl = self.clock, self.flusher
        x = self.make_x(M, self.device)
        fn, reset = runner.bind(x)
        dst = runner.weights
        nbytes = sum(t.numel() * t.element_size() for t in host.values())
        nb = host_mem == "pinned" and c.cuda

        def fetch():
            for k, h in host.items():
                dst[k].copy_(h, non_blocking=nb)

        # Prove the kernel consumes the fetched bytes: wipe, fetch, run, check.
        for t in dst.values():
            t.view(torch.uint8).zero_()
        fetch()
        if reset:
            reset()
        out = fn()
        c.sync()
        err = self.check(out, x, M)
        graph, timing = try_graph(fn, reset, c, self.args.gpu_timing)

        def it_kernel():
            e0, e1, e2 = c.events(3)
            if reset:
                reset()
            fl.flush_gpu()
            c.sync()
            e0.record()
            fetch()
            e1.record()
            graph.replay() if graph is not None else fn()
            e2.record()
            c.sync()
            return {"xfer": e0.elapsed_time(e1), "kernel": e1.elapsed_time(e2)}

        def it_eager():
            e0, e1, e2 = c.events(3)
            if reset:
                reset()
            fl.flush_gpu()
            c.sync()
            t0 = time.perf_counter()
            e0.record()
            fetch()
            e1.record()
            fn()
            e2.record()
            c.sync()
            return {
                "total": (time.perf_counter() - t0) * 1e3,
                "xfer": e0.elapsed_time(e1),
                "span": e1.elapsed_time(e2),
            }

        sk = run_repeats(it_kernel, self.args)
        se = run_repeats(it_eager, self.args)
        k50, k10, k90 = summarize(sk, "kernel")
        t50, t10, t90 = summarize(se, "total")
        x50, x10, x90 = summarize(se, "xfer")
        row = self._row(
            "cpu_store_gpu_compute",
            f"{runner.kernel}+{host_mem}",
            runner.kernel,
            host_mem,
            M,
            total_ms=t50,
            compute_ms=k50,
            overhead_ms=t50 - k50,
            xfer_ms=x50,
            launch_ms=t50 - k50 - x50,
            total_p10_ms=t10,
            total_p90_ms=t90,
            compute_p10_ms=k10,
            compute_p90_ms=k90,
            xfer_p10_ms=x10,
            xfer_p90_ms=x90,
            eager_gpu_span_ms=summarize(se, "span")[0],
            xfer_bytes=nbytes,
            pipelined_ms=max(x50, k50),
            repeats=min(len(sk), len(se)),
            kernel_timing=timing,
            rel_err=err,
        )
        del graph
        return row


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\n", 1)[1],
    )
    p.add_argument(
        "--model-dir",
        required=True,
        help="HF checkpoint dir (config.json + *.safetensors)",
    )
    p.add_argument(
        "--layer",
        type=int,
        default=None,
        help="MoE layer id (default: first MoE layer)",
    )
    p.add_argument("--expert", type=int, default=0, help="routed expert id")
    p.add_argument(
        "--list-experts",
        action="store_true",
        help="print the expert layout and exit",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="build all kernels, check numerics, no sweep",
    )
    p.add_argument(
        "--tokens", default=DEFAULT_TOKENS, help="comma-separated token counts"
    )
    p.add_argument(
        "--scenarios", default="gpu,cpu,fetch", help="subset of gpu,cpu,fetch"
    )
    p.add_argument(
        "--gpu-kernels",
        default="auto",
        help=f"comma list from {GPU_KERNELS} or 'auto' (= every one that applies)",
    )
    p.add_argument(
        "--cpu-kernels",
        default="sglang_fp8_w8a16,sglang_bf16",
        help=f"comma list from {CPU_KERNELS}",
    )
    p.add_argument(
        "--fetch-host-mem",
        default="pinned,pageable",
        help="host memory for scenario 3",
    )
    p.add_argument(
        "--act-host-mem", default="pinned", choices=("pinned", "pageable")
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--min-repeats", type=int, default=5)
    p.add_argument("--max-seconds-per-point", type=float, default=20.0)
    p.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="torch/OpenMP threads (0 = keep)",
    )
    p.add_argument(
        "--gpu-l2-flush-mb", type=float, default=-1, help="-1 = 2x L2; 0 = off"
    )
    p.add_argument(
        "--cpu-llc-flush-mb", type=float, default=-1, help="-1 = 2x L3; 0 = off"
    )
    p.add_argument("--gpu-timing", default="graph", choices=("graph", "events"))
    p.add_argument(
        "--x-std",
        default="auto",
        help="hidden-state std ('auto' fits static FP8 scale)",
    )
    p.add_argument(
        "--check-max-tokens",
        type=int,
        default=256,
        help="numerics check up to this M",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=None)
    p.add_argument(
        "--require-idle-gpu",
        action="store_true",
        help="abort if the GPU is busy",
    )
    p.add_argument(
        "--idle-mib",
        type=int,
        default=2000,
        help="'busy' threshold for memory.used",
    )
    p.add_argument("--skip-amx-check", action="store_true")
    p.add_argument("--skip-pcie-probe", action="store_true")
    p.add_argument(
        "--device",
        default="cuda",
        choices=("cuda", "cpu"),
        help="'cpu' runs the GPU-scenario code on CPU tensors (debug only; timings meaningless)",
    )
    return p.parse_args(argv)


def write_outputs(out_dir: Path, result: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in result["rows"]:
            w.writerow({k: r.get(k) for k in CSV_FIELDS})


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.list_experts:
            info = inspect_checkpoint(args.model_dir, args.layer, args.expert)
            print_inspection(info)
            return 0
        return run(args)
    except (CheckpointError, KernelUnavailable) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


def print_inspection(info):
    cfg = info["config"]
    print(
        f"model dir      : {info['model_dir']}  ({info['num_tensors']} tensors)"
    )
    print(
        f"config         : {cfg['model_type']} {cfg['architectures']} hidden={cfg['hidden_size']} "
        f"moe_inter={cfg['moe_intermediate_size']} experts={cfg['num_experts']} "
        f"top_k={cfg['num_experts_per_tok']} layers={cfg['num_hidden_layers']} act={cfg['hidden_act']}"
    )
    print(
        f"quantization   : {cfg['quant_method']} activation_scheme={cfg['activation_scheme']} "
        f"weight_block_size={cfg['weight_block_size']}"
    )
    print(f"MoE layers     : {info['moe_layers']}")
    seen = {}
    for L, le in info["layers"].items():
        key = (le["kind"], le["num_experts"], tuple(le["projs"]))
        seen.setdefault(key, []).append(L)
    for (kind, n, projs), Ls in seen.items():
        print(
            f"  layers {Ls[0]}..{Ls[-1]} ({len(Ls)}): {kind}, {n} experts, projs={list(projs)}"
        )
    if "selected" in info:
        s = info["selected"]
        print(
            f"selected       : layer {s['layer']} expert {s['expert']}  K(hidden)={s['hidden_size']} "
            f"N(inter)={s['intermediate_size']}  quant={s['quant']}  act={s['activation']}"
        )
        print(
            f"checkpoint size: {fmt_bytes(s['checkpoint_bytes'])} for this expert"
        )
        print(
            f"static act scales: gate/up {s['static_input_scale_gate_up']}  down "
            f"{s['static_input_scale_down']}"
        )
        for role, p in s["projs"].items():
            print(
                f"  {role:<5} {p['shape']} {p['dtype']} scheme={p['scheme']} scale_shape="
                f"{p['scale_shape']} scale={p['scale_value']} block={p['block']} "
                f"input_scale={p['input_scale']}"
            )
            for t in p["tensors"]:
                print(f"        {t}")


def run(args) -> int:
    t_start = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise KernelUnavailable(
            "CUDA not available (no GPU visible). Check `docker run --gpus` and "
            "CUDA_VISIBLE_DEVICES; --device cpu runs a debug-only CPU emulation."
        )
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
    scenarios = []
    for s in args.scenarios.split(","):
        s = s.strip()
        if s not in SCENARIO_ALIASES:
            raise KernelUnavailable(f"unknown scenario {s!r}")
        scenarios.append(SCENARIO_ALIASES[s])
    tokens = [int(t) for t in args.tokens.split(",") if t.strip()]
    tag = time.strftime("%Y%m%d-%H%M%S")

    t0 = time.perf_counter()
    index = SafetensorsIndex(args.model_dir)
    cfg = load_config(args.model_dir)
    layers = discover_experts(index)
    layer = (
        args.layer
        if args.layer is not None
        else (next(iter(layers)) if layers else 0)
    )
    ew = load_expert(index, cfg, layer, args.expert, layers)
    load_ms = (time.perf_counter() - t0) * 1e3
    desc = ew.describe()
    out_dir = Path(
        args.out_dir
        or f"expert_placement_results/{tag}_L{layer}_E{args.expert}"
    )
    print(
        f"expert L{layer} E{args.expert}: K={ew.hidden_size} N={ew.intermediate_size} {ew.quant} "
        f"({fmt_bytes(ew.checkpoint_nbytes)}), act={ew.activation}, x dtype={ew.act_dtype}"
    )
    print(f"output -> {out_dir}")

    env = bench_env.collect(out_dir, device, args.cpu_threads)
    warnings = []
    if device.type == "cpu":
        warnings.append(
            "--device cpu: GPU scenarios emulated on CPU; timings are NOT meaningful"
        )
    if device.type == "cuda":
        if torch.cuda.device_count() > 1:
            warnings.append(
                f"{torch.cuda.device_count()} GPUs visible; using cuda:0 -- set "
                f"CUDA_VISIBLE_DEVICES to one idle GPU"
            )
        used = env["gpu"]["used_memory_mib_at_start"]
        others = bench_env.other_gpu_processes(device)
        if used > args.idle_mib or others:
            msg = f"GPU not idle: memory.used={used} MiB at start; other processes: {others or '-'}"
            if args.require_idle_gpu:
                raise KernelUnavailable(msg)
            warnings.append(msg)
    aff = env["cpu"].get("affinity_cpus")
    if aff and torch.get_num_threads() > aff:
        warnings.append(
            f"torch threads {torch.get_num_threads()} > allowed CPUs {aff}"
        )

    want_cpu = "cpu_compute" in scenarios
    cpu_names = (
        [k.strip() for k in args.cpu_kernels.split(",") if k.strip()]
        if want_cpu
        else []
    )
    if (
        any(k.startswith("sglang") for k in cpu_names)
        and not args.skip_amx_check
    ):
        try:
            from moe_infinity.kernel.cpu import amx_selfcheck

            env["amx_selfcheck"] = amx_selfcheck()
            if not env["amx_selfcheck"]["ok"]:
                warnings.append(
                    f"amx_selfcheck failed {env['amx_selfcheck']}: AMX results unreliable on this "
                    f"host; rerun with ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16"
                )
        except Exception as e:  # noqa: BLE001
            env["amx_selfcheck"] = f"<failed: {e!r}>"

    bench = Bench(args, ew, device)
    result = {
        "schema": "moe-infinity/expert-placement/v1",
        "expert": desc,
        "config": config_summary(cfg),
        "args": vars(args),
        "tokens": tokens,
        "scenarios": scenarios,
        "x_std": bench.x_std,
        "flush": bench.flush_sizes,
        "env": env,
        "setup": {"checkpoint_read_ms": load_ms, "gpu": {}, "cpu": {}},
        "kernels": {"gpu": {}, "cpu": {}},
        "warnings": warnings,
        "errors": [],
        "rows": [],
        "definitions": DEFINITIONS,
        "dry_run": args.dry_run,
    }
    for w in warnings:
        print(f"WARNING: {w}", flush=True)

    if not args.skip_pcie_probe and device.type == "cuda" and not args.dry_run:
        result["pcie_probe"] = bench_env.pcie_probe(
            device, [desc["checkpoint_bytes"], 256 * 2**20]
        )
        for p in result["pcie_probe"]:
            print(
                f"pcie {p['kind']:<13} {fmt_bytes(p['bytes']):>10}: {p['GBps']:.1f} GB/s"
            )

    gpu_runners = {}
    if {"gpu_resident", "cpu_store_gpu_compute"} & set(scenarios):
        names = resolve_gpu_kernels(
            [k.strip() for k in args.gpu_kernels.split(",")], ew
        )
        for k in names:
            try:
                tt = time.perf_counter()
                r = build_gpu_runner(ew, k, device)
                bench.clock.sync()
                up_ms = (time.perf_counter() - tt) * 1e3 - r.prep_ms
                gpu_runners[k] = r
                result["setup"]["gpu"][k] = {
                    "prep_ms": r.prep_ms,
                    "upload_ms": up_ms,
                    "weight_bytes": r.weight_bytes,
                }
                result["kernels"]["gpu"][k] = dict(r.meta)
                print(
                    f"gpu kernel {k}: {r.meta.get('format')} ({fmt_bytes(r.weight_bytes)})"
                )
            except Exception as e:  # noqa: BLE001
                result["errors"].append(
                    {"stage": f"build gpu {k}", "error": repr(e)}
                )
                print(f"gpu kernel {k}: UNAVAILABLE -- {e}", flush=True)
    cpu_runners = {}
    for k in cpu_names:
        try:
            r = build_cpu_runner(ew, k)
            cpu_runners[k] = r
            result["setup"]["cpu"][k] = {
                "pack_ms": r.pack_ms,
                "packed_bytes": r.nbytes,
            }
            result["kernels"]["cpu"][k] = dict(r.meta)
            print(
                f"cpu kernel {k}: {r.meta.get('format')} (pack {r.pack_ms:.1f} ms)"
            )
        except Exception as e:  # noqa: BLE001
            result["errors"].append(
                {"stage": f"build cpu {k}", "error": repr(e)}
            )
            print(f"cpu kernel {k}: UNAVAILABLE -- {e}", flush=True)

    sweep = tokens
    if args.dry_run:
        sweep = [m for m in (1, 16) if m <= max(tokens)] or [tokens[0]]
        args.warmup, args.repeats, args.min_repeats = 1, 1, 1

    host_copies = {}
    if "cpu_store_gpu_compute" in scenarios:
        for k, r in gpu_runners.items():
            for hm in [
                h.strip() for h in args.fetch_host_mem.split(",") if h.strip()
            ]:
                pinned = hm == "pinned" and device.type == "cuda"
                tt = time.perf_counter()
                host_copies[(k, hm)] = r.host_copy(pinned)
                result["setup"]["gpu"][k][f"host_alloc_{hm}_ms"] = (
                    time.perf_counter() - tt
                ) * 1e3

    def guard(stage, f, *a):
        try:
            row = f(*a)
            result["rows"].append(row)
            bench._log(row)
            if row.get("rel_err") is not None and row["rel_err"] > 0.15:
                print(
                    f"WARNING: {stage} rel_err {row['rel_err']:.3f} > 0.15",
                    flush=True,
                )
        except Exception as e:  # noqa: BLE001
            result["errors"].append(
                {
                    "stage": stage,
                    "error": repr(e),
                    "trace": traceback.format_exc(limit=4),
                }
            )
            print(f"ERROR in {stage}: {e!r}", flush=True)
            if device.type == "cuda":
                torch.cuda.synchronize()

    for M in sweep:
        if "gpu_resident" in scenarios:
            for k, r in gpu_runners.items():
                guard(f"gpu_resident/{k}/M={M}", bench.gpu_resident, r, M)
        if "cpu_compute" in scenarios:
            for k, r in cpu_runners.items():
                guard(f"cpu_compute/{k}/M={M}", bench.cpu_compute, r, M)
        if "cpu_store_gpu_compute" in scenarios:
            for (k, hm), host in host_copies.items():
                guard(
                    f"cpu_store_gpu_compute/{k}+{hm}/M={M}",
                    bench.cpu_store_gpu_compute,
                    gpu_runners[k],
                    host,
                    hm,
                    M,
                )
        write_outputs(out_dir, result)

    if device.type == "cuda":
        others = bench_env.other_gpu_processes(device)
        result["env"]["other_gpu_processes_at_end"] = others
        if others:
            result["warnings"].append(
                f"other processes on the GPU at the end: {others}"
            )
            print(f"WARNING: other processes on the GPU at the end:\n{others}")
    result["wall_s"] = time.perf_counter() - t_start
    write_outputs(out_dir, result)
    print(
        f"done in {result['wall_s']:.1f} s: {len(result['rows'])} rows, "
        f"{len(result['errors'])} errors -> {out_dir}/results.{{csv,json}}"
    )
    return 0 if not result["errors"] or result["rows"] else 1


DEFINITIONS = {
    "gpu_resident": {
        "compute_ms": "GPU kernel time: CUDA events around a CUDA-graph replay of the expert "
        "(falls back to an eager event span if capture fails; see kernel_timing)",
        "total_ms": "host wall time of one eager expert call incl. torch.cuda.synchronize",
        "overhead_ms": "total - compute: Python dispatch + kernel launches + sync",
    },
    "cpu_compute": {
        "d2h_ms": "activations GPU -> host buffer (pinned by default) + stream sync",
        "compute_ms": "CPU kernel wall time, result written in place into the host buffer",
        "h2d_ms": "result host buffer -> GPU + stream sync",
        "total_ms": "d2h + compute + h2d as one wall-clock span",
        "overhead_ms": "total - compute",
    },
    "cpu_store_gpu_compute": {
        "xfer_ms": "H2D copy of all GPU-ready expert weight tensors (CUDA events)",
        "compute_ms": "GPU kernel time right after the copy (graph replay, events)",
        "total_ms": "host wall time of copy + eager expert call + sync",
        "overhead_ms": "total - compute (= xfer + launch_ms)",
        "pipelined_ms": "max(xfer, compute): lower bound if the copy is fully prefetched/overlapped",
    },
    "setup": "one-time costs, not in any per-token number",
    "statistics": "median over repeats; p10/p90 given; overhead = median(total) - median(compute)",
    "caches": "GPU L2 and CPU LLC flushed before every timed repeat (outside the timed region)",
}


if __name__ == "__main__":
    sys.exit(main())
