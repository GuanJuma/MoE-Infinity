# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""CPU interference monitor for the CPU scenario (stdlib only).

"Foreign" load on a CPU = its busy time in /proc/stat (user, system, irq,
softirq, steal; any process, any container) minus the time this run's own
threads spent there (/proc/<pid>/task/*/stat, attributed to the CPU each
thread last ran on -- exact for the pinned OpenMP workers).  Visible foreign
processes on the watched CPUs are listed too; inside a container only its own
PID namespace is visible, so run ``check`` on the host (or start the
container with ``--pid=host``) to see other tenants' process names.

  check   sample for --seconds, print/write a report (pre-run noise check)
  record  append one JSON line per --interval to --out until the parent
          process exits (background sampler during the sweep)

    python3 interference_monitor.py check --cpus 8-95 --seconds 10
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpu_layout import format_cpulist, parse_cpulist, smt_siblings  # noqa: E402

PROC = "/proc"


def parse_proc_stat(text: str) -> Dict[int, tuple]:
    """{cpu: (busy_ticks, total_ticks)}; busy excludes idle and iowait."""
    out = {}
    for line in text.splitlines():
        if not (line.startswith("cpu") and line[3:4].isdigit()):
            continue
        f = line.split()
        v = [int(x) for x in f[1:9]] + [0] * max(0, 9 - len(f))
        total = sum(v[:8])
        out[int(f[0][3:])] = (total - v[3] - v[4], total)
    return out


def parse_task_stat(text: str) -> tuple:
    """(utime + stime ticks, last CPU, comm) from a /proc/.../stat line."""
    lo, hi = text.index("("), text.rindex(")")
    rest = text[hi + 2 :].split()
    # rest[0] is field 3 (state): utime/stime = fields 14/15, processor = 39
    return int(rest[11]) + int(rest[12]), int(rest[36]), text[lo + 1 : hi]


def _read(path: str) -> Optional[str]:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def own_ticks(pids: Iterable[int]) -> Dict[int, tuple]:
    """{tid: (ticks, cpu)} for every thread of ``pids``."""
    out = {}
    for pid in pids:
        try:
            tids = os.listdir(f"{PROC}/{pid}/task")
        except OSError:
            continue
        for t in tids:
            s = _read(f"{PROC}/{pid}/task/{t}/stat")
            if s:
                ticks, cpu, _ = parse_task_stat(s)
                out[int(t)] = (ticks, cpu)
    return out


def proc_snapshot(exclude: set) -> Dict[int, tuple]:
    """{pid: (ticks, cpu, comm)} for every visible process not in ``exclude``."""
    out = {}
    try:
        names = os.listdir(PROC)
    except OSError:
        return out
    for n in names:
        if not n.isdigit() or int(n) in exclude:
            continue
        s = _read(f"{PROC}/{n}/stat")
        if s:
            try:
                out[int(n)] = parse_task_stat(s)
            except (ValueError, IndexError):
                continue
    return out


def foreign_delta(stat_a, stat_b, own_a, own_b, cpus) -> Dict[int, float]:
    """Fraction of each CPU's time spent on anything but our threads."""
    ours = {}
    for tid, (ticks, cpu) in own_b.items():
        prev = own_a.get(tid, (0, cpu))[0]
        ours[cpu] = ours.get(cpu, 0) + max(0, ticks - prev)
    out = {}
    for c in cpus:
        if c not in stat_a or c not in stat_b:
            continue
        busy = stat_b[c][0] - stat_a[c][0]
        total = stat_b[c][1] - stat_a[c][1]
        if total > 0:
            out[c] = max(0.0, busy - ours.get(c, 0)) / total
    return out


def top_procs(pa, pb, watch: set, dt: float, hz: int, n: int = 8) -> List[list]:
    rows = []
    for pid, (ticks, cpu, comm) in pb.items():
        d = ticks - pa.get(pid, (ticks,))[0]
        if d > 0 and cpu in watch and dt > 0:
            rows.append([pid, comm, cpu, round(100.0 * d / hz / dt, 1)])
    rows.sort(key=lambda r: -r[3])
    return rows[:n]


class Sampler:
    def __init__(self, cpus, own_pids, siblings=True, procs=True):
        self.cpus = sorted(cpus)
        sib = smt_siblings(self.cpus) if siblings else {}
        self.siblings = sorted(
            {s for v in sib.values() for s in v} - set(self.cpus)
        )
        self.watch = set(self.cpus) | set(self.siblings)
        self.own = set(own_pids) | {os.getpid()}
        self.procs = procs
        self.hz = os.sysconf("SC_CLK_TCK")
        self._prev = self._snap(procs)

    def _snap(self, procs):
        return (
            time.monotonic(),
            parse_proc_stat(_read(f"{PROC}/stat") or ""),
            own_ticks(self.own),
            proc_snapshot(self.own) if procs else None,
        )

    def sample(self, procs: Optional[bool] = None) -> dict:
        procs = self.procs if procs is None else procs
        cur = self._snap(procs)
        t0, sa, oa, pa = self._prev
        t1, sb, ob, pb = cur
        if pa is None and procs:
            pa = pb
        f = foreign_delta(sa, sb, oa, ob, self.watch)
        rec = {
            "t0": t0,
            "t": t1,
            "foreign": {str(c): round(v, 4) for c, v in f.items() if v > 0},
        }
        if procs and pb is not None:
            rec["procs"] = top_procs(pa, pb, self.watch, t1 - t0, self.hz)
        self._prev = cur if procs or pa is None else (t1, sb, ob, pa)
        return rec


def window_stats(samples: List[dict], cpus, siblings, t0=None, t1=None) -> dict:
    """Foreign load on ``cpus`` (and their SMT siblings) over samples that
    overlap [t0, t1]: max per sample window and time-weighted mean."""
    sel = [
        s
        for s in samples
        if (t0 is None or s["t"] > t0) and (t1 is None or s["t0"] < t1)
    ]
    cpus = [str(c) for c in cpus]
    sibs = [str(c) for c in siblings]
    out = {
        "samples": len(sel),
        "foreign_max_pct": 0.0,
        "foreign_cores_max": 0.0,
    }
    if not sel:
        return out
    span = sum(s["t"] - s["t0"] for s in sel) or 1.0
    mean = {}
    worst_cpu, worst = None, 0.0
    for s in sel:
        f = s["foreign"]
        dt = s["t"] - s["t0"]
        cores = sum(f.get(c, 0.0) for c in cpus)
        out["foreign_cores_max"] = max(out["foreign_cores_max"], cores)
        for c in cpus:
            v = f.get(c, 0.0)
            mean[c] = mean.get(c, 0.0) + v * dt / span
            if v > worst:
                worst, worst_cpu = v, c
        out["sibling_max_pct"] = max(
            out.get("sibling_max_pct", 0.0),
            100.0 * max((f.get(c, 0.0) for c in sibs), default=0.0),
        )
    out["foreign_max_pct"] = round(100.0 * worst, 1)
    out["foreign_max_cpu"] = int(worst_cpu) if worst_cpu is not None else None
    out["foreign_cores_max"] = round(out["foreign_cores_max"], 3)
    out["sibling_max_pct"] = round(out.get("sibling_max_pct", 0.0), 1)
    top = sorted(mean.items(), key=lambda kv: -kv[1])
    out["foreign_mean_max_pct"] = round(100.0 * (top[0][1] if top else 0.0), 2)
    out["busiest_cpus"] = [
        [int(c), round(100.0 * v, 1)] for c, v in top[:5] if v > 0
    ]
    procs = {}
    for s in sel:
        for pid, comm, cpu, pct in s.get("procs", []):
            key = (pid, comm)
            procs[key] = max(procs.get(key, 0.0), pct)
    out["procs"] = [
        [pid, comm, pct]
        for (pid, comm), pct in sorted(procs.items(), key=lambda kv: -kv[1])[:8]
    ]
    return out


def check(cpus, own_pids, seconds=10.0, interval=0.5, procs=True) -> dict:
    s = Sampler(cpus, own_pids, procs=procs)
    samples = []
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        time.sleep(min(interval, max(0.0, end - time.monotonic())) or interval)
        samples.append(s.sample())
    rep = window_stats(samples, s.cpus, s.siblings)
    rep.update(
        seconds=seconds,
        interval_s=interval,
        cpus=format_cpulist(s.cpus),
        siblings=format_cpulist(s.siblings),
    )
    return rep


def describe(rep: dict) -> str:
    lines = [
        f"noise check {rep.get('seconds')} s on CPUs {rep.get('cpus')} "
        f"(SMT siblings {rep.get('siblings') or '-'}): foreign load mean max "
        f"{rep['foreign_mean_max_pct']}% of a core, worst {rep.get('interval_s')} s "
        f"window {rep['foreign_max_pct']}% (CPU {rep.get('foreign_max_cpu')}), "
        f"siblings max {rep.get('sibling_max_pct', 0)}%"
    ]
    if rep.get("busiest_cpus"):
        lines.append(f"  busiest CPUs (mean %): {rep['busiest_cpus']}")
    if rep.get("procs"):
        lines.append(
            f"  visible foreign processes (pid, comm, max %): {rep['procs']}"
        )
    return "\n".join(lines)


def record(cpus, own_pids, out: Path, interval=0.2, proc_every=1.0) -> None:
    parent = os.getppid()
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    s = Sampler(cpus, own_pids, procs=True)
    next_procs = time.monotonic() + proc_every
    with open(out, "a") as f:
        while not stop and os.getppid() == parent:
            time.sleep(interval)
            want = time.monotonic() >= next_procs
            if want:
                next_procs += proc_every
            f.write(json.dumps(s.sample(procs=want)) + "\n")
            f.flush()


def read_samples(path: Path) -> List[dict]:
    out = []
    try:
        with open(path) as f:
            for line in f:
                if line.endswith("\n"):
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass
    except OSError:
        pass
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("mode", choices=("check", "record"))
    p.add_argument(
        "--cpus", required=True, help="compute CPUs to watch, e.g. 8-95"
    )
    p.add_argument("--exclude-pid", type=int, action="append", default=[])
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--interval", type=float, default=None)
    p.add_argument("--proc-every", type=float, default=1.0)
    p.add_argument("--pin", default=None, help="run this monitor on these CPUs")
    p.add_argument("--no-procs", action="store_true")
    p.add_argument("--json", default=None, help="check: write the report here")
    p.add_argument("--out", default=None, help="record: JSON-lines file")
    a = p.parse_args(argv)
    if a.pin:
        os.sched_setaffinity(0, parse_cpulist(a.pin))
    cpus = parse_cpulist(a.cpus)
    if a.mode == "check":
        rep = check(
            cpus, a.exclude_pid, a.seconds, a.interval or 0.5, not a.no_procs
        )
        print(describe(rep), flush=True)
        if a.json:
            Path(a.json).write_text(json.dumps(rep, indent=2))
        return 0
    if not a.out:
        p.error("record needs --out")
    record(cpus, a.exclude_pid, Path(a.out), a.interval or 0.2, a.proc_every)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
