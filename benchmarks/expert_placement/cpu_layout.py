# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""CPU layout for the CPU scenario (stdlib only; run_sweep.sh calls the CLI).

Default: one OpenMP place per physical core of the GPU's NUMA node.

Soft isolation (``housekeeping`` given): the compute cores are the node's
physical cores minus the housekeeping cores; their SMT siblings stay idle.
With ``main="housekeeping"`` the first OpenMP place is the whole housekeeping
set, so libgomp binds the Python main thread (the OpenMP master, which also
takes 1/T of every parallel region) there and threads it starts later inherit
it; workers get one compute core each.  ``main="compute"`` makes the master
one of the compute places instead.  Nothing outside this process is touched.

    python3 cpu_layout.py --cpus 0,1,...,95 --housekeeping 0-7 [--main housekeeping]
    -> shell ``export`` lines for OMP_* and EP_* variables
"""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import Dict, Iterable, List, Optional

MAIN_MODES = ("housekeeping", "compute")


def parse_cpulist(spec: str) -> set:
    cpus = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition("-")
        cpus.update(range(int(a), int(b or a) + 1))
    return cpus


def format_cpulist(cpus: Iterable[int]) -> str:
    cpus = sorted(set(cpus))
    out, i = [], 0
    while i < len(cpus):
        j = i
        while j + 1 < len(cpus) and cpus[j + 1] == cpus[j] + 1:
            j += 1
        out.append(str(cpus[i]) if i == j else f"{cpus[i]}-{cpus[j]}")
        i = j + 1
    return ",".join(out)


def smt_siblings(
    cpus: Iterable[int], sys_root: str = "/sys"
) -> Dict[int, List[int]]:
    """Other hardware threads of each CPU's core (empty without SMT)."""
    out = {}
    for c in cpus:
        p = Path(
            f"{sys_root}/devices/system/cpu/cpu{c}/topology/thread_siblings_list"
        )
        try:
            sib = parse_cpulist(p.read_text().strip())
        except OSError:
            sib = {c}
        out[c] = sorted(sib - {c})
    return out


def layout(
    cores: Iterable[int],
    threads: Optional[int] = None,
    housekeeping: Optional[Iterable[int]] = None,
    main: str = "housekeeping",
) -> dict:
    """OpenMP places and the EP_* description of the CPU layout.

    ``threads`` caps the number of compute places (OpenMP workers on their own
    core); OMP_NUM_THREADS is the number of places, i.e. one more than that
    when the master sits on the housekeeping place."""
    if main not in MAIN_MODES:
        raise ValueError(f"main must be one of {MAIN_MODES}, not {main!r}")
    cores = list(cores)
    hk_set = set(housekeeping or ())
    hk = [c for c in cores if c in hk_set]
    compute = [c for c in cores if c not in hk_set]
    if threads:
        compute = compute[: int(threads)]
    if hk_set and not hk:
        raise ValueError(
            f"housekeeping CPUs {format_cpulist(hk_set)} are not among the cores "
            f"{format_cpulist(cores)}"
        )
    if not compute:
        raise ValueError(
            "no compute cores left after removing the housekeeping set"
        )
    places = []
    if hk and main == "housekeeping":
        places.append("{%s}" % ",".join(map(str, hk)))
    places += ["{%d}" % c for c in compute]
    env = {
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": ",".join(places),
        "OMP_NUM_THREADS": str(len(places)),
        "EP_ALLOWED_CPUS": ",".join(map(str, sorted(hk + compute))),
    }
    if hk:
        env.update(
            EP_ISOLATE="soft",
            EP_COMPUTE_CPUS=",".join(map(str, compute)),
            EP_HOUSEKEEPING_CPUS=",".join(map(str, hk)),
            EP_MAIN_THREAD=main,
        )
    return {
        "compute": compute,
        "housekeeping": hk,
        "main": main if hk else "compute",
        "threads": len(places),
        "env": env,
    }


def from_env(env) -> Optional[dict]:
    """The soft-isolation layout a parent (run_sweep.sh / re-exec) set up."""
    if env.get("EP_ISOLATE") != "soft" or not env.get("EP_COMPUTE_CPUS"):
        return None
    return {
        "compute": sorted(parse_cpulist(env["EP_COMPUTE_CPUS"])),
        "housekeeping": sorted(
            parse_cpulist(env.get("EP_HOUSEKEEPING_CPUS", ""))
        ),
        "main": env.get("EP_MAIN_THREAD", "housekeeping"),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cpus", required=True, help="physical cores (comma list)")
    p.add_argument(
        "--housekeeping", default="", help="e.g. 0-7; empty = no isolation"
    )
    p.add_argument("--main", default="housekeeping", choices=MAIN_MODES)
    p.add_argument(
        "--threads", type=int, default=0, help="cap on compute places"
    )
    a = p.parse_args(argv)
    cores = [int(c) for c in a.cpus.split(",") if c.strip()]
    lay = layout(
        cores,
        a.threads or None,
        parse_cpulist(a.housekeeping) if a.housekeeping else None,
        a.main,
    )
    for k, v in lay["env"].items():
        print(f"export {k}={shlex.quote(v)}")
    print(f"export EP_N_COMPUTE={len(lay['compute'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
