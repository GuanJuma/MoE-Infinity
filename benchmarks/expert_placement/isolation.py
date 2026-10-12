# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""In-process side of CPU isolation and interference accounting.

* ``thread_stats``: per-thread run-queue delay (schedstat), context switches
  and migrations of this process -- read at row boundaries only.
* ``Isolation.enforce``: under soft isolation, every thread of ours that is not
  an OpenMP compute worker (MoE-Infinity task-pool and AIO threads, CUDA and
  torch helper threads, the main thread when it is not a compute place) is
  pinned to the housekeeping CPUs.  The engine pins its task-pool threads to
  CPU ``(++counter) % nprocs`` itself (core/prefetch/task_thread.cpp); this
  overrides that with ``--engine-cpus``.
* A background ``interference_monitor.py record`` process on the housekeeping
  CPUs samples foreign load on the compute CPUs; rows are joined to its
  samples by CLOCK_MONOTONIC time windows.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import cpu_layout
import interference_monitor as im

HERE = Path(__file__).resolve().parent


def _read(path) -> Optional[str]:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def thread_stats(pid: Optional[int] = None) -> Dict[int, dict]:
    pid = pid or os.getpid()
    out = {}
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return out
    for t in tids:
        tid = int(t)
        base = f"/proc/{pid}/task/{t}"
        d = {}
        s = _read(f"{base}/schedstat")
        if s:
            run, wait, slices = (int(x) for x in s.split()[:3])
            d.update(run_ns=run, wait_ns=wait, slices=slices)
        s = _read(f"{base}/status")
        if s:
            for line in s.splitlines():
                if line.startswith("voluntary_ctxt_switches"):
                    d["vol"] = int(line.split()[1])
                elif line.startswith("nonvoluntary_ctxt_switches"):
                    d["nonvol"] = int(line.split()[1])
        s = _read(f"{base}/sched")
        if s:
            for line in s.splitlines():
                if line.startswith("se.nr_migrations"):
                    d["migrations"] = int(line.split()[-1])
        s = _read(f"{base}/stat")
        if s:
            try:
                d["cpu"] = im.parse_task_stat(s)[1]
            except (ValueError, IndexError):
                pass
        try:
            d["affinity"] = frozenset(os.sched_getaffinity(tid))
        except OSError:
            continue
        out[tid] = d
    return out


def stats_delta(a: dict, b: dict, tids) -> dict:
    """Run-queue delay / switches / migrations of ``tids`` between snapshots."""
    waits, nonvol, migr, moved = [], 0, None, 0
    for t in tids:
        x, y = a.get(t), b.get(t)
        if not x or not y:
            continue
        waits.append(max(0, y.get("wait_ns", 0) - x.get("wait_ns", 0)) / 1e6)
        nonvol += max(0, y.get("nonvol", 0) - x.get("nonvol", 0))
        if "migrations" in x and "migrations" in y:
            migr = (migr or 0) + max(0, y["migrations"] - x["migrations"])
        if x.get("cpu") is not None and x.get("cpu") != y.get("cpu"):
            moved += 1
    return {
        "threads": len(waits),
        "run_delay_max_ms": round(max(waits), 4) if waits else 0.0,
        "run_delay_sum_ms": round(sum(waits), 4),
        "nonvol_switches": nonvol,
        "migrations": migr,
        "cpu_changed": moved,
    }


class Isolation:
    """CPU layout of this run, thread pinning, and per-row interference."""

    def __init__(self, args, out_dir: Path):
        self.args = args
        self.pid = os.getpid()
        self.main_tid = self.pid
        lay = cpu_layout.from_env(os.environ)
        self.soft = lay is not None
        if self.soft:
            self.compute = set(lay["compute"])
            self.housekeeping = set(lay["housekeeping"])
            self.main_mode = lay["main"]
        else:
            allowed = os.environ.get("EP_ALLOWED_CPUS")
            self.compute = (
                cpu_layout.parse_cpulist(allowed)
                if allowed
                else set(os.sched_getaffinity(0))
            )
            self.housekeeping = set()
            self.main_mode = "compute"
        self.engine_cpus = (
            cpu_layout.parse_cpulist(args.engine_cpus)
            if getattr(args, "engine_cpus", None)
            else set(self.housekeeping)
        )
        self.engine_tids = set()
        self._pre_engine = None
        self.siblings = sorted(
            {
                s
                for v in cpu_layout.smt_siblings(self.compute).values()
                for s in v
            }
            - self.compute
        )
        mon = getattr(args, "monitor", "auto")
        self.monitor_on = mon == "on" or (mon == "auto" and self.soft)
        self.samples_path = out_dir / "interference_samples.jsonl"
        self.proc = None
        self.enforced = {}

    # -- layout --------------------------------------------------------------

    def describe(self) -> dict:
        return {
            "mode": "soft" if self.soft else "none",
            "compute_cpus": cpu_layout.format_cpulist(self.compute),
            "housekeeping_cpus": cpu_layout.format_cpulist(self.housekeeping),
            "smt_siblings_left_idle": cpu_layout.format_cpulist(self.siblings),
            "main_thread": self.main_mode,
            "engine_cpus": cpu_layout.format_cpulist(self.engine_cpus),
            "monitor": self.monitor_on,
            "threads": self.enforced,
        }

    def before_engine(self):
        self._pre_engine = set(thread_stats(self.pid))

    def after_engine(self):
        if self._pre_engine is not None:
            self.engine_tids = set(thread_stats(self.pid)) - self._pre_engine
        self.enforce()

    def compute_threads(self, stats=None) -> set:
        stats = stats if stats is not None else thread_stats(self.pid)
        main_aff = stats.get(self.main_tid, {}).get("affinity", frozenset())
        out = set()
        if self.main_mode == "compute" or not self.soft:
            out.add(self.main_tid)
        for tid, d in stats.items():
            aff = d.get("affinity", frozenset())
            if (
                tid != self.main_tid
                and tid not in self.engine_tids
                and len(aff) == 1
                and aff <= self.compute
                and aff != main_aff
            ):
                out.add(tid)
        return out

    def enforce(self) -> dict:
        """Pin every non-compute thread of ours to the housekeeping CPUs."""
        if not self.soft:
            return {}
        stats = thread_stats(self.pid)
        workers = self.compute_threads(stats)
        moved = 0
        for tid, d in stats.items():
            if tid in workers:
                continue
            want = (
                self.engine_cpus
                if tid in self.engine_tids
                else self.housekeeping
            )
            if not want or d.get("affinity") == frozenset(want):
                continue
            try:
                os.sched_setaffinity(tid, want)
                moved += 1
            except OSError:
                pass
        self.enforced = {
            "compute_workers": len(workers) - (self.main_mode == "compute"),
            "engine_threads": len(self.engine_tids),
            "housekeeping_threads": len(stats) - len(workers),
            "repinned_last": moved,
        }
        return self.enforced

    # -- monitor ---------------------------------------------------------------

    def start_monitor(self):
        if not self.monitor_on or self.proc is not None:
            return
        self.samples_path.parent.mkdir(parents=True, exist_ok=True)
        self.samples_path.write_text("")
        cmd = [
            sys.executable,
            str(HERE / "interference_monitor.py"),
            "record",
            "--cpus",
            cpu_layout.format_cpulist(self.compute),
            "--exclude-pid",
            str(self.pid),
            "--interval",
            str(self.args.monitor_interval),
            "--out",
            str(self.samples_path),
        ]
        if self.housekeeping:
            cmd += ["--pin", cpu_layout.format_cpulist(self.housekeeping)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL)

    def stop_monitor(self):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None

    def noise_check(self, seconds: float) -> dict:
        rep = im.check(self.compute, [self.pid], seconds, min(0.5, seconds))
        print(im.describe(rep), flush=True)
        return rep

    # -- rows ------------------------------------------------------------------

    def row_begin(self):
        self.enforce()
        return time.monotonic(), thread_stats(self.pid)

    def row_end(self, token, scenario: str) -> dict:
        t0, a = token
        t1 = time.monotonic()
        b = thread_stats(self.pid)
        workers = self.compute_threads(b)
        m = {"window_s": round(t1 - t0, 3)}
        m["compute"] = stats_delta(a, b, workers - {self.main_tid})
        m["main"] = stats_delta(a, b, [self.main_tid])
        if self.proc is not None:
            deadline = time.monotonic() + 3 * self.args.monitor_interval
            samples = im.read_samples(self.samples_path)
            while (
                not samples or samples[-1]["t"] < t1
            ) and time.monotonic() < deadline:
                time.sleep(self.args.monitor_interval / 4)
                samples = im.read_samples(self.samples_path)
            m["foreign"] = im.window_stats(
                samples, sorted(self.compute), self.siblings, t0, t1
            )
        m["reasons"] = self.reasons(m, scenario)
        m["interfered"] = bool(m["reasons"])
        return m

    def reasons(self, m: dict, scenario: str) -> list:
        a = self.args
        out = []
        if scenario == "cpu_compute":
            c = m["compute"]
            if c["run_delay_max_ms"] > a.interference_delay_ms:
                out.append(
                    f"compute thread run-queue delay {c['run_delay_max_ms']:.2f} ms"
                )
            if c.get("migrations"):
                out.append(
                    f"{c['migrations']} migrations of pinned compute threads"
                )
            f = m.get("foreign") or {}
            if f.get("foreign_max_pct", 0) > a.interference_foreign_pct:
                out.append(
                    f"foreign load {f['foreign_max_pct']}% on CPU {f.get('foreign_max_cpu')}"
                )
            if f.get("sibling_max_pct", 0) > a.interference_foreign_pct:
                out.append(
                    f"foreign load {f['sibling_max_pct']}% on an SMT sibling"
                )
        if m["main"]["run_delay_max_ms"] > a.interference_delay_ms:
            out.append(
                f"main thread run-queue delay {m['main']['run_delay_max_ms']:.2f} ms"
            )
        return out
