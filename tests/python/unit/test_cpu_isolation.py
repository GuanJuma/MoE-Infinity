# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""CPU layout / soft isolation, interference accounting, the repeat protocol,
and the restorable host isolation (fake cgroup-v1 root + fake docker)."""

import ctypes
import importlib
import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BENCH = REPO / "benchmarks" / "expert_placement"
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(HERE))

layout_mod = importlib.import_module("cpu_layout")
im = importlib.import_module("interference_monitor")
iso_mod = importlib.import_module("isolation")
NCPU = len(os.sched_getaffinity(0))
FAKE_DOCKER = f"{sys.executable} {HERE / 'fake_docker.py'}"


def test_layout_soft_isolation_places():
    lay = layout_mod.layout(range(8), housekeeping={0, 1})
    env = lay["env"]
    assert env["OMP_PLACES"] == "{0,1},{2},{3},{4},{5},{6},{7}"
    assert env["OMP_NUM_THREADS"] == "7" and env["EP_ISOLATE"] == "soft"
    assert (
        env["EP_COMPUTE_CPUS"] == "2,3,4,5,6,7"
        and env["EP_MAIN_THREAD"] == "housekeeping"
    )
    lay = layout_mod.layout(
        range(8), threads=4, housekeeping={0, 1}, main="compute"
    )
    assert lay["env"]["OMP_PLACES"] == "{2},{3},{4},{5}" and lay["threads"] == 4
    assert layout_mod.from_env(lay["env"])["compute"] == [2, 3, 4, 5]
    plain = layout_mod.layout([4, 5, 6])["env"]
    assert "EP_ISOLATE" not in plain and plain["OMP_PLACES"] == "{4},{5},{6}"
    with pytest.raises(ValueError):
        layout_mod.layout([0, 1], housekeeping={0, 1})
    assert layout_mod.format_cpulist([0, 1, 2, 5, 7, 8]) == "0-2,5,7-8"
    out = subprocess.run(
        [sys.executable, str(BENCH / "cpu_layout.py"), "--cpus", "0,1,2,3", "--housekeeping", "0"],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    assert (
        "export OMP_PLACES='{0},{1},{2},{3}'" in out and "EP_N_COMPUTE=3" in out
    )


@pytest.mark.skipif(NCPU < 3, reason="needs 3 CPUs")
def test_libgomp_binds_master_to_housekeeping_and_workers_to_cores():
    cpus = sorted(os.sched_getaffinity(0))[:3]
    env = dict(
        os.environ, **layout_mod.layout(cpus, housekeeping={cpus[0]})["env"]
    )
    probe = textwrap.dedent(
        """
        import os, sys, torch
        sys.path.insert(0, %r)
        import isolation
        torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
        x = torch.randn(1024, 1024); (x @ x).sum()
        st = isolation.thread_stats()
        print(sorted(st[os.getpid()]["affinity"]))
        print(sorted(sorted(d["affinity"]) for t, d in st.items() if len(d["affinity"]) == 1))
        """
        % str(BENCH)
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert out[0] == str([cpus[0]])
    workers = json.loads(out[1])
    assert [cpus[1]] in workers and [cpus[2]] in workers


def test_proc_parsers_and_foreign_load():
    stat = "cpu  1 2 3 4\ncpu0 100 0 50 800 50 0 0 0 0 0\ncpu1 10 0 0 990 0 0 0 0\n"
    assert im.parse_proc_stat(stat) == {0: (150, 1000), 1: (10, 1000)}
    fields = ["0"] * 50
    fields[0], fields[11], fields[12], fields[36] = (
        "R",
        "7",
        "5",
        "6",
    )  # state, utime, stime, cpu
    line = "123 (a (b) c) " + " ".join(fields)
    assert im.parse_task_stat(line) == (12, 6, "a (b) c")
    a = {0: (0, 0), 1: (0, 0)}
    b = {0: (60, 100), 1: (100, 100)}
    ours_a, ours_b = {7: (0, 1)}, {7: (90, 1)}
    f = im.foreign_delta(a, b, ours_a, ours_b, [0, 1])
    assert f[0] == pytest.approx(0.6) and f[1] == pytest.approx(0.1)
    samples = [
        {
            "t0": 0.0,
            "t": 1.0,
            "foreign": {"0": 0.6},
            "procs": [[5, "rpm", 0, 60.0]],
        },
        {"t0": 1.0, "t": 2.0, "foreign": {"1": 0.1}},
    ]
    w = im.window_stats(samples, [0, 1], [], 0.5, 1.5)
    assert w["foreign_max_pct"] == 60.0 and w["foreign_max_cpu"] == 0
    assert w["procs"] == [[5, "rpm", 60.0]] and w["samples"] == 2
    assert im.window_stats(samples, [0, 1], [], 5, 6)["samples"] == 0


def test_noise_check_and_record_live(tmp_path):
    cpus = sorted(os.sched_getaffinity(0))
    rep = im.check(cpus, [os.getpid()], seconds=0.6, interval=0.3)
    assert rep["samples"] >= 1 and "foreign_mean_max_pct" in rep
    assert "noise check" in im.describe(rep)
    out = tmp_path / "s.jsonl"
    p = subprocess.Popen(
        [sys.executable, str(BENCH / "interference_monitor.py"), "record", "--cpus",
         layout_mod.format_cpulist(cpus), "--interval", "0.1", "--out", str(out)]
    )  # fmt: skip
    time.sleep(0.8)
    p.terminate()
    p.wait(5)
    s = im.read_samples(out)
    assert len(s) >= 3 and all(x["t"] > x["t0"] for x in s)


def test_thread_stats_and_enforce_repins_helper_threads(monkeypatch, tmp_path):
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 2:
        pytest.skip("needs 2 CPUs")
    hk, comp = cpus[0], cpus[1:]
    lay = layout_mod.layout(cpus, housekeeping={hk})
    for k, v in lay["env"].items():
        monkeypatch.setenv(k, v)

    class A:
        engine_cpus = None
        monitor = "off"
        monitor_interval = 0.1
        interference_delay_ms = 1.0
        interference_foreign_pct = 20.0

    iso = iso_mod.Isolation(A(), tmp_path)
    assert iso.soft and iso.compute == set(comp) and iso.housekeeping == {hk}
    iso.before_engine()
    stop = threading.Event()
    t = threading.Thread(target=stop.wait)
    t.start()
    try:
        tid = t.native_id
        os.sched_setaffinity(tid, {comp[-1], hk})
        iso.after_engine()
        assert tid in iso.engine_tids
        assert os.sched_getaffinity(tid) == {hk}
        tok = iso.row_begin()
        m = iso.row_end(tok, "cpu_compute")
        assert set(m) >= {"compute", "main", "interfered", "reasons"}
        assert m["main"]["threads"] == 1
    finally:
        stop.set()
        t.join()
        os.sched_setaffinity(0, set(cpus))
    a = {1: {"wait_ns": 0, "nonvol": 0, "migrations": 0, "cpu": 2}}
    b = {1: {"wait_ns": 3_000_000, "nonvol": 2, "migrations": 1, "cpu": 3}}
    d = iso_mod.stats_delta(a, b, [1])
    assert d["run_delay_max_ms"] == 3.0 and d["nonvol_switches"] == 2
    assert d["migrations"] == 1 and d["cpu_changed"] == 1


def test_graph_node_counter_with_fake_driver():
    bench = importlib.import_module("expert_placement_bench")
    types = [0, 0, 2, 1, 0, 7]

    class Lib:
        def cuGraphGetNodes(self, g, nodes, n):
            if nodes is not None:
                for i in range(len(types)):
                    nodes[i] = 1000 + i
            n._obj.value = len(types)
            return 0

        def cuGraphNodeGetType(self, node, t):
            t._obj.value = types[node.value - 1000]
            return 0

    got = bench.cuda_graph_node_types(0x1234, Lib())
    assert got == {"kernel": 3, "memcpy": 1, "memset": 1, "other": 1}
    assert isinstance(ctypes.c_void_p(0x1234).value, int)


def _fake_ckpt(tmp_path):
    ck = tmp_path / "ckpt"
    subprocess.run(
        [sys.executable, str(BENCH / "make_fake_checkpoint.py"), "--format", "hy3_fp8", str(ck)],
        check=True, capture_output=True,
    )  # fmt: skip
    return ck


BENCH_ARGS = [
    "--device", "cpu", "--warmup", "1", "--repeats", "3", "--min-repeats", "1",
    "--cpu-llc-flush-mb", "4", "--skip-amx-check", "--mi-store-lib", "fake_moe_infinity_store",
]  # fmt: skip


@pytest.mark.skipif(NCPU < 2, reason="needs 2 CPUs")
def test_soft_isolated_sweep_records_interference_and_retries(tmp_path):
    ck = _fake_ckpt(tmp_path)
    cpus = sorted(os.sched_getaffinity(0))
    env = dict(
        os.environ,
        PYTHONPATH=f"{HERE}:{REPO}",
        **layout_mod.layout(cpus, housekeeping={cpus[0]})["env"],
    )
    out = tmp_path / "out"
    p = subprocess.run(
        [sys.executable, str(BENCH / "expert_placement_bench.py"), "--model-dir", str(ck),
         "--layer", "1", "--expert", "0", "--tokens", "1,8", "--scenarios", "gpu,cpu",
         "--gpu-kernels", "torch_scaled_mm", "--cpu-kernels", "torch_bf16", *BENCH_ARGS,
         "--noise-check-s", "0.6", "--monitor", "on", "--interference-delay-ms", "-1",
         "--retry-interfered", "1", "--out-dir", str(out)],
        env=env, capture_output=True, text=True, timeout=600,
    )  # fmt: skip
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    res = json.loads((out / "results.json").read_text())
    assert res["env"]["isolation"]["mode"] == "soft"
    assert res["noise_check"]["samples"] >= 1
    s1 = [r for r in res["rows"] if r["scenario"] == "gpu_resident"]
    s2 = [r for r in res["rows"] if r["scenario"] == "cpu_compute"]
    assert s1 and s2
    for r in s1:
        assert r["host_dispatch_ms"] > 0 and r["hidden_dispatch_ms"] >= 0
        assert r["exposed_overhead_ms"] == pytest.approx(
            r["total_ms"] - r["compute_ms"]
        )
    for r in s2:
        assert r["interfered"] and r["attempts"] == 2
        assert len(r["interference"]["attempts"]) == 2
        assert "foreign" in r["interference"]
    assert (out / "interference_samples.jsonl").stat().st_size > 0
    md = (
        (out / "summary.md").read_text()
        if (out / "summary.md").exists()
        else ""
    )
    summ = importlib.import_module("summarize_placement")
    md = summ.render(res)
    assert (
        "## CPU isolation and interference" in md and "exposed overhead" in md
    )
    assert "host dispatch" in md and "⚠" in md


def _round(res_dir: Path, values: dict, flag=()):
    rows = [
        {"scenario": "cpu_compute", "variant": "k", "tokens": m, "compute_ms": v,
         "total_ms": v + 0.05, "interfered": m in flag}
        for m, v in values.items()
    ]  # fmt: skip
    res_dir.mkdir(parents=True)
    (res_dir / "results.json").write_text(json.dumps({"rows": rows, "env": {}}))


def test_rounds_summary_min_of_medians_spread_and_ab(tmp_path):
    rounds = importlib.import_module("summarize_rounds")
    _round(
        tmp_path / "fp8" / "shared" / "round_1", {1: 0.20, 16: 0.50}, flag=(16,)
    )
    _round(tmp_path / "fp8" / "shared" / "round_2", {1: 0.30, 16: 0.40})
    _round(tmp_path / "fp8" / "isolated" / "round_1", {1: 0.19, 16: 0.36})
    _round(tmp_path / "fp8" / "isolated" / "round_2", {1: 0.19, 16: 0.37})
    md = rounds.render(tmp_path, ab=("shared", "isolated"))
    assert "| 1 | 0.2000 | 0.3000 | **0.2000** | 50% | - |" in md
    assert "| 16 | 0.5000* | 0.4000 | **0.4000** | 25% | 1 |" in md
    assert "| 16 | 0.4000 | 0.3600 | 0.90 | 25% | 3% | 1 / 0 |" in md


def _fake_host(tmp_path):
    """cgroup-v1 cpuset tree + docker state with our container and two others."""
    cg = tmp_path / "cg"
    for d in ("", "docker", "docker/ours", "docker/a", "docker/b"):
        (cg / d).mkdir(parents=True, exist_ok=True)
        (cg / d / "cpuset.cpus").write_text("0-383")
        (cg / d / "cpuset.mems").write_text("0-1")
        (cg / d / "cgroup.procs").write_text("")
    (cg / ".mi_fake_cgroupfs").write_text("")
    procs = [subprocess.Popen(["sleep", "600"]) for _ in range(4)]
    host, pinned, ours, other = (p.pid for p in procs)
    os.sched_setaffinity(pinned, {0})
    (cg / "cgroup.procs").write_text(f"{host}\n{pinned}\n2\n")
    (cg / ".kthreads").write_text("2")
    (cg / "docker/ours/cgroup.procs").write_text(str(ours))
    (cg / "docker/a/cgroup.procs").write_text(str(other))
    st = {"containers": [
        {"id": "o", "name": "ours", "cpus": "0-95,192-287", "mems": "0", "pid": ours, "cgroup": "/docker/ours"},
        {"id": "a", "name": "vllm_ep", "cpus": "", "mems": "", "pid": other, "cgroup": "/docker/a"},
        {"id": "b", "name": "moe_ep", "cpus": "64-95", "mems": "0", "pid": other, "cgroup": "/docker/a"},
    ]}  # fmt: skip
    (tmp_path / "docker.json").write_text(json.dumps(st))
    env = dict(
        os.environ, MI_ISOLATE_TEST="1", FAKE_DOCKER_STATE=str(tmp_path / "docker.json"),
        FAKE_CGROUP_ROOT=str(cg),
    )  # fmt: skip
    args = ["--cgroup-root", str(cg), "--docker", FAKE_DOCKER, "--ours", "ours",
            "--state", str(tmp_path / "state.json"), "--no-siblings"]  # fmt: skip
    return cg, env, args, procs, pinned


def _iso(cmd, env, args, *extra):
    return subprocess.run(
        ["bash", str(BENCH / "isolate_cpus.sh"), cmd, *args, *extra],
        env=env, capture_output=True, text=True, timeout=120,
    )  # fmt: skip


def test_isolate_cpus_plan_apply_restore_roundtrip(tmp_path):
    cg, env, args, procs, pinned = _fake_host(tmp_path)
    try:
        before = (cg / "cgroup.procs").read_text()
        p = _iso("plan", env, args)
        assert p.returncode == 0, p.stderr
        assert "docker update --cpuset-cpus=0-79,96-383" in p.stdout
        assert (
            "docker update --cpuset-cpus=64-79" in p.stdout
            and "OURS: kept" in p.stdout
        )
        assert not (cg / "mi_housekeeping").exists()
        p = _iso("apply", env, args, "--yes")
        assert p.returncode == 0, p.stdout + p.stderr
        assert (
            cg / "mi_housekeeping" / "cpuset.cpus"
        ).read_text() == "0-79,96-383"
        assert (cg / "cgroup.procs").read_text().split() == ["2"]
        conts = {
            c["name"]: c["cpus"]
            for c in json.loads((tmp_path / "docker.json").read_text())[
                "containers"
            ]
        }
        assert conts == {
            "ours": "0-95,192-287",
            "vllm_ep": "0-79,96-383",
            "moe_ep": "64-79",
        }
        assert _iso("apply", env, args, "--yes").returncode != 0
        assert (
            "applied"
            in _iso("status", env, args, "--status-seconds", "0.1").stdout
        )
        os.sched_setaffinity(pinned, set(range(NCPU)))
        p = _iso("restore", env, args)
        assert p.returncode == 0, p.stdout + p.stderr
        assert sorted((cg / "cgroup.procs").read_text().split()) == sorted(
            before.split()
        )
        assert not (cg / "mi_housekeeping").exists()
        assert os.sched_getaffinity(pinned) == {0}
        conts = {
            c["name"]: c["cpus"]
            for c in json.loads((tmp_path / "docker.json").read_text())[
                "containers"
            ]
        }
        assert conts["moe_ep"] == "64-95" and conts["vllm_ep"] == "0-383"
        assert "docker update cannot clear" in p.stdout
        assert "no active state" in _iso("restore", env, args).stdout
    finally:
        for p in procs:
            p.kill()


def test_isolate_cpus_refuses_conflicts_and_option_b(tmp_path):
    cg, env, args, procs, _ = _fake_host(tmp_path)
    try:
        st = json.loads((tmp_path / "docker.json").read_text())
        st["containers"].append(
            {
                "id": "z",
                "name": "zeya-sglang",
                "cpus": "80-87",
                "mems": "0",
                "pid": 1,
                "cgroup": None,
            }
        )
        (tmp_path / "docker.json").write_text(json.dumps(st))
        p = _iso("apply", env, args, "--yes")
        assert p.returncode != 0 and "zeya-sglang" in p.stdout + p.stderr
        assert (
            not (cg / "mi_housekeeping").exists()
            and not (tmp_path / "state.json").exists()
        )
        p = _iso("apply", env, [a for a in args], "--option", "B", "--yes")
        assert p.returncode != 0 and "admin-approved" in p.stdout + p.stderr
        assert "option B" in _iso("plan", env, args, "--option", "B").stdout
    finally:
        for p in procs:
            p.kill()


@pytest.mark.skipif(NCPU < 3, reason="needs 3 CPUs")
def test_ab_runner_isolates_restores_and_summarizes(tmp_path):
    ck = _fake_ckpt(tmp_path)
    cg, env, args, procs, _ = _fake_host(tmp_path)
    cpus = sorted(os.sched_getaffinity(0))
    stub = tmp_path / "bin"
    stub.mkdir()
    (stub / "nvidia-smi").write_text(
        '#!/bin/bash\ncase "$*" in *index*) echo 0;; *memory.used*) echo 3;; '
        "*pci.bus_id*) echo 00000000:00:04.0;; *) echo stub;; esac\n"
    )
    (stub / "nvidia-smi").chmod(0o755)
    out = tmp_path / "ab"
    env.update(
        PATH=f"{stub}:{env['PATH']}", PYTHONPATH=str(HERE), NAME="ours", ME="me",
        ROUNDS="2", GAP_S="0", PRECISIONS="fp8", CPUS=",".join(map(str, cpus[1:])),
        HK=str(cpus[0]), REPO_C=str(REPO), OUT_C=str(out), HOST_OUT=str(out),
        STATE=str(tmp_path / "ab_state.json"), DOCKER=FAKE_DOCKER,
        ISO_ARGS=f"--cgroup-root {cg} --no-siblings", MODEL_DIR=str(ck), TOKENS="1",
        REPEATS="2", NOISE_CHECK_S="0.5", REQUIRE_STORE="0", CPU_KERNELS="torch_bf16",
        BENCH_ARGS=" ".join(BENCH_ARGS),
    )  # fmt: skip
    try:
        p = subprocess.run(
            ["bash", str(BENCH / "ab_isolation.sh")], env=env, capture_output=True,
            text=True, timeout=900,
        )  # fmt: skip
        assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-2000:]
        for leg in ("shared", "isolated"):
            for r in (1, 2):
                res = json.loads(
                    (
                        out / "fp8" / leg / f"round_{r}" / "results.json"
                    ).read_text()
                )
                iso = res["env"]["isolation"]
                assert iso["compute_cpus"] == layout_mod.format_cpulist(
                    cpus[1:]
                )
                assert iso["main_thread"] == "compute"
        md = (out / "ab_summary.md").read_text()
        assert "## A/B: `shared` vs `isolated`" in md and "| 1 |" in md
        assert not (cg / "mi_housekeeping").exists()
        assert len(list(tmp_path.glob("ab_state.json.restored-*"))) == 2
        log = (out / "ab_host.log").read_text()
        assert log.index("round 1: shared") < log.index("round 1: isolated")
        assert log.index("round 2: isolated") < log.index("round 2: shared")
    finally:
        for p in procs:
            p.kill()


def test_new_scripts_parse():
    for name in (
        "isolate_cpus.sh",
        "ab_isolation.sh",
        "s2_rounds.sh",
        "run_sweep.sh",
    ):
        subprocess.run(["bash", "-n", str(BENCH / name)], check=True)
    src = (BENCH / "isolate_cpus.py").read_text()
    assert (
        "capture_output" not in src
        and "from __future__ import annotations" not in src
    )
