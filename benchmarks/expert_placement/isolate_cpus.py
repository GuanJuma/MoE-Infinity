#!/usr/bin/env python3
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
"""Restorable CPU isolation on a shared cgroup-v1 docker host (run as root).

Moves everybody else off a set of CPUs for the duration of a measurement and
puts everything back exactly from a state file:

* other running containers: ``docker update --cpuset-cpus=<theirs minus
  isolated>`` (never our container);
* host tasks in the root cpuset: moved into ``<cpuset>/mi_housekeeping``
  (cpus = all minus isolated, mems = all); per-CPU kernel threads are skipped;
* cgroup v1 overwrites the affinity of every task in a cpuset whose CPUs
  change, so custom per-thread affinities of affected tasks are saved and
  re-applied on restore;
* our container: ``keep`` (default; it already covers the isolated CPUs and
  the benchmark pins itself) or ``move`` its processes into
  ``<cpuset>/mi_isolated``.

Not used: ``cset shield`` (its reset returns tasks to the root cpuset instead
of their docker cgroups and drops their cpuset limits).  IRQ affinity is not
changed (irq/softirq time on the isolated CPUs stays visible in ``status``).

Subcommands: plan (default, dry run) | apply | status | restore.
Python >= 3.6, stdlib only (runs on the host's system python3).
"""

import argparse
import errno
import json
import os
import subprocess
import sys
import time

PF_KTHREAD = 0x00200000
OPTIONS = {"A": "80-95", "B": "8-95"}
STATE_DEFAULT = "/var/tmp/mi_isolate_cpus.state.json"
HK, ISO = "mi_housekeeping", "mi_isolated"


def parse_cpulist(spec):
    out = set()
    for part in str(spec or "").replace("\n", ",").split(","):
        part = part.strip()
        if part:
            a, _, b = part.partition("-")
            out.update(range(int(a), int(b or a) + 1))
    return out


def fmt(cpus):
    cpus = sorted(set(cpus))
    out, i = [], 0
    while i < len(cpus):
        j = i
        while j + 1 < len(cpus) and cpus[j + 1] == cpus[j] + 1:
            j += 1
        out.append(str(cpus[i]) if i == j else "%d-%d" % (cpus[i], cpus[j]))
        i = j + 1
    return ",".join(out)


def read(path, default=None):
    try:
        with open(path) as f:
            return f.read()
    except (IOError, OSError):
        return default


class Cgroups(object):
    """cgroup-v1 cpuset hierarchy.  A root containing ``.mi_fake_cgroupfs``
    is emulated (membership = the cgroup.procs files) for tests."""

    def __init__(self, root, proc="/proc"):
        self.root = root.rstrip("/")
        self.proc = proc
        self.fake = os.path.exists(os.path.join(self.root, ".mi_fake_cgroupfs"))

    def path(self, rel):
        rel = rel.strip("/")
        return os.path.join(self.root, rel) if rel else self.root

    def get(self, rel, name):
        return (read(os.path.join(self.path(rel), name), "") or "").strip()

    def set(self, rel, name, value):
        with open(os.path.join(self.path(rel), name), "w") as f:
            f.write(str(value))

    def exists(self, rel):
        return os.path.isdir(self.path(rel))

    def procs(self, rel):
        return [
            int(x) for x in self.get(rel, "cgroup.procs").split() if x.strip()
        ]

    def walk(self):
        """Relative paths of every cgroup, parents first."""
        out = []
        for d, subdirs, _ in os.walk(self.root):
            subdirs.sort()
            rel = os.path.relpath(d, self.root)
            out.append("/" if rel == "." else "/" + rel)
        return out

    def mkdir(self, rel, cpus, mems):
        p = self.path(rel)
        if not os.path.isdir(p):
            os.mkdir(p)
            if self.fake:
                for n in ("cgroup.procs", "cpuset.cpus", "cpuset.mems"):
                    open(os.path.join(p, n), "a").close()
        self.set(rel, "cpuset.mems", mems)
        self.set(rel, "cpuset.cpus", cpus)

    def rmdir(self, rel):
        p = self.path(rel)
        if not os.path.isdir(p):
            return
        if self.fake:
            if self.procs(rel):
                raise OSError(errno.EBUSY, "cgroup not empty", p)
            for n in os.listdir(p):
                os.unlink(os.path.join(p, n))
        os.rmdir(p)

    def is_kthread(self, pid):
        if self.fake:
            return pid in parse_cpulist(
                read(os.path.join(self.root, ".kthreads"), "")
            )
        s = read("%s/%d/stat" % (self.proc, pid))
        if not s:
            return True
        rest = s[s.rindex(")") + 2 :].split()
        return bool(int(rest[6]) & PF_KTHREAD)

    def cgroup_of(self, pid):
        if self.fake:
            for rel in self.walk():
                if pid in self.procs(rel):
                    return rel
            return None
        for line in (
            read("%s/%d/cgroup" % (self.proc, pid), "") or ""
        ).splitlines():
            parts = line.split(":", 2)
            if len(parts) == 3 and "cpuset" in parts[1].split(","):
                return parts[2] or "/"
        return None

    def attach(self, pid, rel):
        if self.fake:
            if self.is_kthread(pid):
                raise OSError(errno.EINVAL, "kernel thread", str(pid))
            for r in self.walk():
                ps = self.procs(r)
                if pid in ps:
                    self.set(
                        r,
                        "cgroup.procs",
                        "\n".join(str(x) for x in ps if x != pid),
                    )
            ps = self.procs(rel) + [pid]
            self.set(rel, "cgroup.procs", "\n".join(str(x) for x in ps))
            return
        with open(os.path.join(self.path(rel), "cgroup.procs"), "w") as f:
            f.write(str(pid))


class Docker(object):
    def __init__(self, exe):
        self.exe = exe.split()

    def _run(self, *args):
        p = subprocess.Popen(
            self.exe + list(args),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        out, err = p.communicate()
        if p.returncode != 0:
            raise RuntimeError("docker %s: %s" % (" ".join(args), err.strip()))
        return out

    def containers(self):
        ids = self._run("ps", "-q", "--no-trunc").split()
        if not ids:
            return []
        fmt_ = (
            "{{.Id}}\t{{.Name}}\t{{.HostConfig.CpusetCpus}}\t"
            "{{.HostConfig.CpusetMems}}\t{{.State.Pid}}"
        )
        out = []
        for line in self._run("inspect", "--format", fmt_, *ids).splitlines():
            f = line.split("\t")
            if len(f) >= 5:
                out.append(
                    {
                        "id": f[0],
                        "name": f[1].lstrip("/"),
                        "cpus": f[2],
                        "mems": f[3],
                        "pid": int(f[4] or 0),
                    }
                )
        return out

    def update_cpus(self, cid, cpus):
        self._run("update", "--cpuset-cpus=" + cpus, cid)


def thread_affinities(proc, pids):
    """{tid: sorted cpus} for every thread of ``pids``."""
    out = {}
    for pid in pids:
        try:
            tids = os.listdir("%s/%d/task" % (proc, pid))
        except OSError:
            continue
        for t in tids:
            try:
                out[int(t)] = sorted(os.sched_getaffinity(int(t)))
            except OSError:
                pass
    return out


class Isolator(object):
    def __init__(self, a):
        self.a = a
        self.cg = Cgroups(a.cgroup_root, a.proc)
        self.docker = Docker(a.docker)
        self.all_cpus = parse_cpulist(self.cg.get("/", "cpuset.cpus"))
        self.all_mems = self.cg.get("/", "cpuset.mems")
        base = parse_cpulist(a.cpus or OPTIONS[a.option])
        self.isolated = set(base)
        if not a.no_siblings:
            for c in base:
                sib = read(
                    "%s/devices/system/cpu/cpu%d/topology/thread_siblings_list"
                    % (a.sys, c)
                )
                if sib:
                    self.isolated |= parse_cpulist(sib.strip())
        self.keep = self.all_cpus - self.isolated

    # -- inspection -----------------------------------------------------------

    def containers(self):
        out = []
        for c in self.docker.containers():
            c["ours"] = c["name"] == self.a.ours
            c["cgroup"] = self.cg.cgroup_of(c["pid"]) if c["pid"] else None
            eff = parse_cpulist(c["cpus"]) if c["cpus"] else set(self.all_cpus)
            c["new"] = fmt(eff - self.isolated)
            c["touch"] = (not c["ours"]) and bool(eff & self.isolated)
            c["conflict"] = c["touch"] and not (eff - self.isolated)
            out.append(c)
        return out

    def root_tasks(self):
        movable, kthreads = [], []
        for pid in self.cg.procs("/"):
            (kthreads if self.cg.is_kthread(pid) else movable).append(pid)
        return movable, kthreads

    def comm(self, pid):
        return (read("%s/%d/comm" % (self.a.proc, pid), "?") or "?").strip()

    def uncovered(self, conts):
        known = {"/", "/" + HK, "/" + ISO}
        known |= {c["cgroup"] for c in conts if c["cgroup"]}
        out = []
        for rel in self.cg.walk():
            if rel in known or any(
                rel.startswith(k + "/") for k in known if k != "/"
            ):
                continue
            n = len(self.cg.procs(rel))
            cpus = parse_cpulist(self.cg.get(rel, "cpuset.cpus"))
            if n and cpus & self.isolated:
                out.append((rel, n, fmt(cpus)))
        return out

    # -- plan -----------------------------------------------------------------

    def header(self):
        print(
            "isolated CPUs : %s (%d logical)"
            % (fmt(self.isolated), len(self.isolated))
        )
        print("left to others: %s" % fmt(self.keep))
        if self.a.option == "B" or len(self.isolated) > 64:
            print("!" * 78)
            print(
                "WARNING: option B / a large isolated set squeezes every other tenant on "
                "this host\ninto %s. It needs explicit admin approval "
                "(--admin-approved)." % fmt(self.keep)
            )
            print("!" * 78)

    def plan(self):
        self.header()
        conts = self.containers()
        ours = [c for c in conts if c["ours"]]
        print("\ncontainers (%d running):" % len(conts))
        for c in conts:
            if c["ours"]:
                act = "OURS: %s" % (
                    "kept as is"
                    if self.a.ours_mode == "keep"
                    else "processes -> " + ISO
                )
            elif c["conflict"]:
                act = "CONFLICT: all of its CPUs are isolated -> apply refuses"
            elif c["touch"]:
                act = "docker update --cpuset-cpus=%s" % c["new"]
            else:
                act = "no overlap, untouched"
            print(
                "  %-28s cpuset=%-14s %s" % (c["name"], c["cpus"] or '""', act)
            )
        if not ours:
            print("  (our container %r is not running)" % self.a.ours)
        elif self.a.ours_mode == "keep":
            have = (
                parse_cpulist(ours[0]["cpus"])
                if ours[0]["cpus"]
                else self.all_cpus
            )
            if not self.isolated <= have:
                print(
                    "  WARNING: our container's cpuset %s does not cover the isolated CPUs"
                    % ours[0]["cpus"]
                )
        movable, kth = self.root_tasks()
        names = {}
        for pid in movable:
            names[self.comm(pid)] = names.get(self.comm(pid), 0) + 1
        print(
            "\nhost tasks in the root cpuset: %d movable -> %s (cpus=%s, mems=%s); "
            "%d kernel threads stay"
            % (len(movable), HK, fmt(self.keep), self.all_mems, len(kth))
        )
        print(
            "  " + ", ".join("%s x%d" % kv for kv in sorted(names.items())[:40])
        )
        for rel, n, cpus in self.uncovered(conts):
            print(
                "  NOT HANDLED: cgroup %s (%d tasks, cpus %s)" % (rel, n, cpus)
            )
        if self.a.ours_mode == "move":
            print(
                "\n%s: cpus=%s mems=%s, our container's processes moved there"
                % (
                    ISO,
                    fmt(
                        self.isolated | parse_cpulist(self.a.ours_housekeeping)
                    ),
                    self.a.mems,
                )
            )
        print(
            "\nper-thread affinities of affected tasks are saved and re-applied on restore"
            "\nstate file: %s" % self.a.state
        )
        return conts

    # -- state ----------------------------------------------------------------

    def load(self):
        s = read(self.a.state)
        return json.loads(s) if s else None

    def save(self, st):
        tmp = self.a.state + ".tmp"
        with open(tmp, "w") as f:
            json.dump(st, f, indent=1)
        os.rename(tmp, self.a.state)

    # -- apply ----------------------------------------------------------------

    def apply(self):
        if self.a.option == "B" and not self.a.admin_approved:
            raise SystemExit("option B needs --admin-approved (see the guide)")
        old = self.load()
        if old and old.get("phase") != "restored":
            raise SystemExit(
                "state %s exists (phase %s): run restore first"
                % (self.a.state, old.get("phase"))
            )
        conts = self.plan()
        bad = [c["name"] for c in conts if c["conflict"]]
        if bad:
            raise SystemExit(
                "refusing: containers fully inside the isolated CPUs: %s" % bad
            )
        if not self.keep:
            raise SystemExit("refusing: nothing left for the other tasks")
        if not self.a.yes:
            sys.stdout.write("\napply? [y/N] ")
            sys.stdout.flush()
            if sys.stdin.readline().strip().lower() not in ("y", "yes"):
                raise SystemExit("aborted, nothing changed")
        touched = [c for c in conts if c["touch"]]
        movable, _ = self.root_tasks()
        aff_pids = list(movable)
        for c in touched:
            if c["cgroup"]:
                for rel in self.cg.walk():
                    if rel == c["cgroup"] or rel.startswith(c["cgroup"] + "/"):
                        aff_pids += self.cg.procs(rel)
        st = {
            "version": 1,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "phase": "applying",
            "isolated": fmt(self.isolated),
            "keep": fmt(self.keep),
            "all_cpus": fmt(self.all_cpus),
            "all_mems": self.all_mems,
            "cgroup_root": self.cg.root,
            "containers": [
                {
                    "id": c["id"],
                    "name": c["name"],
                    "orig_cpus": c["cpus"],
                    "new_cpus": c["new"],
                    "applied": False,
                }
                for c in touched
            ],
            "host_moved": {},
            "ours": {"mode": self.a.ours_mode, "moved": {}},
            "affinity": {
                str(t): m
                for t, m in thread_affinities(self.a.proc, aff_pids).items()
                if set(m) != self.all_cpus
            },
            "errors": [],
        }
        self.save(st)
        self.cg.mkdir(HK, fmt(self.keep), self.all_mems)
        for c in st["containers"]:
            try:
                self.docker.update_cpus(c["id"], c["new_cpus"])
                c["applied"] = True
            except RuntimeError as e:
                st["errors"].append(str(e))
            self.save(st)
        for _ in range(3):
            movable, _ = self.root_tasks()
            if not movable:
                break
            for pid in movable:
                try:
                    self.cg.attach(pid, HK)
                    st["host_moved"][str(pid)] = "/"
                except (IOError, OSError) as e:
                    if e.errno != errno.ESRCH:
                        st["errors"].append(
                            "pid %d (%s): %s" % (pid, self.comm(pid), e)
                        )
            self.save(st)
        if self.a.ours_mode == "move":
            ours = [c for c in conts if c["ours"]]
            if ours and ours[0]["cgroup"]:
                self.cg.mkdir(
                    ISO,
                    fmt(
                        self.isolated | parse_cpulist(self.a.ours_housekeeping)
                    ),
                    self.a.mems,
                )
                for pid in self.cg.procs(ours[0]["cgroup"]):
                    self.cg.attach(pid, ISO)
                    st["ours"]["moved"][str(pid)] = ours[0]["cgroup"]
        st["phase"] = "applied"
        self.save(st)
        left, kth = self.root_tasks()
        print(
            "\napplied: %d containers updated, %d host tasks moved to %s, %d left in the "
            "root cpuset (+%d kernel threads), %d errors"
            % (
                sum(c["applied"] for c in st["containers"]),
                len(st["host_moved"]),
                HK,
                len(left),
                len(kth),
                len(st["errors"]),
            )
        )
        for e in st["errors"][:20]:
            print("  error: " + e)
        print(
            "restore with: sudo bash %s/isolate_cpus.sh restore"
            % os.path.dirname(os.path.abspath(__file__))
        )

    # -- restore --------------------------------------------------------------

    def _drain(self, rel, dest_of):
        for _ in range(5):
            if not self.cg.exists(rel):
                return
            pids = self.cg.procs(rel)
            if not pids:
                break
            for pid in pids:
                dest = dest_of.get(str(pid), "/")
                if not self.cg.exists(dest):
                    dest = "/"
                try:
                    self.cg.attach(pid, dest)
                except (IOError, OSError):
                    try:
                        self.cg.attach(pid, "/")
                    except (IOError, OSError):
                        pass
        try:
            self.cg.rmdir(rel)
        except OSError as e:
            print("WARNING: could not remove %s: %s" % (self.cg.path(rel), e))

    def restore(self):
        st = self.load()
        if not st or st.get("phase") == "restored":
            print(
                "no active state (%s); cleaning up leftovers only"
                % self.a.state
            )
            self._drain(ISO, {})
            self._drain(HK, {})
            return
        live = {c["id"]: c for c in self.docker.containers()}
        for c in st["containers"]:
            if not c.get("applied"):
                continue
            if c["id"] not in live:
                print("  %s is gone; nothing to restore" % c["name"])
                continue
            target = c["orig_cpus"] or st["all_cpus"]
            try:
                self.docker.update_cpus(c["id"], target)
                c["applied"] = False
                note = (
                    ""
                    if c["orig_cpus"]
                    else (
                        '  (was "": docker update cannot clear CpusetCpus; set the full '
                        "list %s, which is equivalent)" % target
                    )
                )
                print("  %s -> --cpuset-cpus=%s%s" % (c["name"], target, note))
            except RuntimeError as e:
                print("  ERROR %s: %s" % (c["name"], e))
            self.save(st)
        self._drain(ISO, st.get("ours", {}).get("moved", {}))
        self._drain(HK, st.get("host_moved", {}))
        bad = 0
        for t, m in st.get("affinity", {}).items():
            try:
                os.sched_setaffinity(int(t), set(m))
            except OSError:
                bad += 1
        print(
            "restored %d thread affinities (%d threads gone)"
            % (len(st.get("affinity", {})) - bad, bad)
        )
        st["phase"] = "restored"
        st["restored"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.save(st)
        os.rename(
            self.a.state,
            self.a.state + ".restored-" + time.strftime("%Y%m%d-%H%M%S"),
        )
        print("done; %s and %s removed" % (HK, ISO))

    # -- status ---------------------------------------------------------------

    def status(self, seconds=1.0):
        st = self.load()
        print(
            "state: %s"
            % ("%s (%s)" % (st["phase"], st["created"]) if st else "none")
        )
        self.header()
        for rel in (HK, ISO):
            if self.cg.exists(rel):
                print(
                    "%s: cpus=%s tasks=%d"
                    % (
                        rel,
                        self.cg.get(rel, "cpuset.cpus"),
                        len(self.cg.procs(rel)),
                    )
                )
        movable, kth = self.root_tasks()
        print(
            "root cpuset: %d user tasks, %d kernel threads"
            % (len(movable), len(kth))
        )
        print("containers:")
        for c in self.containers():
            real = (
                self.cg.get(c["cgroup"], "cpuset.cpus") if c["cgroup"] else "?"
            )
            over = parse_cpulist(real) & self.isolated if real != "?" else set()
            print(
                "  %-28s HostConfig=%-14s cgroup=%-16s %s"
                % (
                    c["name"],
                    c["cpus"] or '""',
                    real,
                    "ours"
                    if c["ours"]
                    else ("ON ISOLATED CPUs" if over else "ok"),
                )
            )
        busy = self._busy(seconds)
        print(
            "busy %% of isolated CPUs over %.1f s (any task, incl. irq): max %.1f%%, "
            "mean %.1f%%"
            % (
                seconds,
                max(busy.values() or [0]),
                sum(busy.values()) / max(1, len(busy)),
            )
        )
        on = self._tasks_on_isolated()
        print("tasks whose last CPU is isolated: %d" % len(on))
        for row in on[:20]:
            print("  pid %-7d tid %-7d %-18s cpu %-4d %s" % row)

    def _busy(self, seconds):
        def snap():
            out = {}
            for line in (read("%s/stat" % self.a.proc, "") or "").splitlines():
                if line.startswith("cpu") and line[3:4].isdigit():
                    v = [int(x) for x in line.split()[1:9]]
                    out[int(line.split()[0][3:])] = (
                        sum(v) - v[3] - v[4],
                        sum(v),
                    )
            return out

        a = snap()
        time.sleep(seconds)
        b = snap()
        out = {}
        for c in self.isolated:
            if c in a and c in b and b[c][1] > a[c][1]:
                out[c] = 100.0 * (b[c][0] - a[c][0]) / (b[c][1] - a[c][1])
        return out

    def _tasks_on_isolated(self):
        rows = []
        for n in os.listdir(self.a.proc):
            if not n.isdigit():
                continue
            pid = int(n)
            try:
                tids = os.listdir("%s/%d/task" % (self.a.proc, pid))
            except OSError:
                continue
            for t in tids:
                s = read("%s/%d/task/%s/stat" % (self.a.proc, pid, t))
                if not s:
                    continue
                rest = s[s.rindex(")") + 2 :].split()
                cpu = int(rest[36])
                if cpu in self.isolated and rest[0] == "R":
                    rows.append(
                        (
                            pid,
                            int(t),
                            s[s.index("(") + 1 : s.rindex(")")],
                            cpu,
                            self.cg.cgroup_of(pid) or "?",
                        )
                    )
        return rows


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="isolate_cpus.sh", description=__doc__.split("\n\n")[0]
    )
    p.add_argument(
        "cmd",
        nargs="?",
        default="plan",
        choices=("plan", "apply", "status", "restore"),
    )
    p.add_argument(
        "--option",
        default="A",
        choices=sorted(OPTIONS),
        help="A: 80-95 (16 cores), B: 8-95 (admin approval)",
    )
    p.add_argument(
        "--cpus",
        default=None,
        help="isolated physical cores (overrides --option)",
    )
    p.add_argument(
        "--no-siblings",
        action="store_true",
        help="do not add the SMT siblings of --cpus",
    )
    p.add_argument(
        "--ours",
        default=os.environ.get("NAME", ""),
        help="our container's name (never updated); default $NAME",
    )
    p.add_argument("--ours-mode", default="keep", choices=("keep", "move"))
    p.add_argument(
        "--ours-housekeeping",
        default="72-79",
        help="move mode: extra CPUs in mi_isolated for our helper threads",
    )
    p.add_argument("--mems", default="0", help="mi_isolated memory nodes")
    p.add_argument("--state", default=STATE_DEFAULT)
    p.add_argument("--yes", action="store_true", help="apply without asking")
    p.add_argument("--admin-approved", action="store_true")
    p.add_argument("--status-seconds", type=float, default=1.0)
    p.add_argument("--cgroup-root", default="/sys/fs/cgroup/cpuset")
    p.add_argument("--docker", default="docker")
    p.add_argument("--proc", default="/proc")
    p.add_argument("--sys", default="/sys")
    a = p.parse_args(argv)
    iso = Isolator(a)
    if a.cmd == "plan":
        iso.plan()
        print("\n(dry run: nothing changed; run with 'apply')")
    elif a.cmd == "apply":
        try:
            iso.apply()
        except SystemExit:
            raise
        except BaseException:
            sys.stderr.write(
                "\n!!! apply failed half way; the state file %s lists what was changed.\n"
                "!!! restore with: sudo bash %s/isolate_cpus.sh restore --state %s\n"
                % (a.state, os.path.dirname(os.path.abspath(__file__)), a.state)
            )
            raise
    elif a.cmd == "restore":
        iso.restore()
    else:
        iso.status(a.status_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
