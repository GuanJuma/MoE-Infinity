#!/usr/bin/env bash
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
#
# Run the expert-placement sweep inside the container, NUMA-bound to the
# GPU's node: inspect -> dry-run -> sweep -> summary.
#
#   MODEL_DIR=/model LAYER=1 EXPERT=0 bash benchmarks/expert_placement/run_sweep.sh [extra args]
#
# Env knobs: MODEL_DIR LAYER EXPERT PRECISION (fp8|bf16) OUT TOKENS SCENARIOS
#            GPU_KERNELS CPU_KERNELS FETCH_MODES REPEATS CPU_THREADS
#            NUMA_NODE (auto) SKIP_DRY_RUN=1 IDLE_MIB REQUIRE_STORE=0 (--expert-store torch)
#            EXCLUDE_CPUS (e.g. 0-7) OUT_OWNER (uid:gid for the outputs; default: repo owner)
#            OMP_PIN=0 (do not pin OpenMP threads; diagnostics only)
#            CORES (explicit core list instead of the GPU node's, e.g. 72-95)
#            ISOLATE=1 soft isolation: HOUSEKEEPING (0-7) ISO_MAIN (housekeeping|compute)
#            NOISE_CHECK_S (10) NOISE_ABORT_PCT (30, isolation only) RETRY (2 isolated, 0 else)
#            MONITOR (auto: on under isolation)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH="$REPO/benchmarks/expert_placement"
MODEL_DIR="${MODEL_DIR:-/model}"
LAYER="${LAYER:-1}"
EXPERT="${EXPERT:-0}"
PRECISION="${PRECISION:-fp8}"
OUT="${OUT:-/scratch/expert_placement/$(date +%Y%m%d-%H%M%S)_${PRECISION}_L${LAYER}_E${EXPERT}}"
TOKENS="${TOKENS:-1,2,3,4,8,16,32,64,128,256,512,1024,2048,4096,8192}"
SCENARIOS="${SCENARIOS:-gpu,cpu,fetch}"
GPU_KERNELS="${GPU_KERNELS:-default}"
CPU_KERNELS="${CPU_KERNELS:-default}"
FETCH_MODES="${FETCH_MODES:-mi_fetch,mi_fetch_evict,mi_prefetch,raw_pinned,raw_pageable}"
REPEATS="${REPEATS:-30}"
IDLE_MIB="${IDLE_MIB:-2000}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/scratch/torch_extensions}"
mkdir -p "$OUT"

if [ "${REQUIRE_STORE:-1}" = "1" ] && ! python3 -c "import torch, moe_infinity._store" 2>/dev/null; then
    echo "ERROR: moe_infinity._store is not built; S2/S3 need MoE-Infinity's expert store." >&2
    echo "       Build it: python3 $BENCH/build_extensions.py (see the guide)" >&2
    exit 4
fi

# --- GPU: exactly one visible, idle ---------------------------------------
n_gpu=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [ "$n_gpu" -ne 1 ]; then
    echo "WARNING: $n_gpu GPUs visible; the benchmark uses cuda:0." \
        "Start the container with --gpus \"device=N\" (one idle GPU)." >&2
fi
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
if [ "$used" -ge "$IDLE_MIB" ]; then
    echo "ERROR: GPU 0 has ${used} MiB in use (>= ${IDLE_MIB}); someone else is on it." >&2
    exit 3
fi

# --- NUMA node of the GPU and its physical cores ---------------------------
if [ "${NUMA_NODE:-auto}" = "auto" ]; then
    bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i 0 | tr -d ' ' | tr 'A-Z' 'a-z')
    bus="${bus#0000}"  # nvidia-smi prints an 8-digit PCI domain, sysfs uses 4
    NUMA_NODE=$(cat "/sys/bus/pci/devices/${bus}/numa_node" 2>/dev/null || echo 0)
    [ "$NUMA_NODE" -lt 0 ] && NUMA_NODE=0
fi
cpulist=$(cat "/sys/devices/system/node/node${NUMA_NODE}/cpulist" 2>/dev/null || echo "0-$(($(nproc) - 1))")
phys=$(python3 - "$cpulist" <<'EOF'
import sys
def expand(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out
seen, keep = set(), []
for c in expand(sys.argv[1]):
    try:
        sib = open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read().strip()
    except OSError:
        sib = str(c)
    if sib not in seen:
        seen.add(sib)
        keep.append(c)
print(",".join(map(str, keep)))
EOF
)
if [ -n "${EXCLUDE_CPUS:-}" ]; then
    # e.g. EXCLUDE_CPUS=0-7: keep OpenMP off the cores the MoE-Infinity engine's
    # task-pool threads are pinned to (task_thread.cpp: CPU 1, 2, ...).
    phys=$(python3 - "$phys" "$EXCLUDE_CPUS" <<'PYEOF'
import sys
def expand(s):
    out = set()
    for part in s.split(","):
        if part.strip():
            a, _, b = part.partition("-")
            out.update(range(int(a), int(b or a) + 1))
    return out
keep = [c for c in map(int, sys.argv[1].split(",")) if c not in expand(sys.argv[2])]
print(",".join(map(str, keep)))
PYEOF
)
fi
if [ -n "${CORES:-}" ]; then
    # Explicit core set (e.g. CORES=72-95 for the isolated-core A/B): physical
    # cores only, SMT siblings dropped.
    phys=$(python3 - "$BENCH" "$CORES" <<'PYEOF'
import sys
sys.path.insert(0, sys.argv[1])
from cpu_layout import parse_cpulist, smt_siblings
keep, drop = [], set()
for c in sorted(parse_cpulist(sys.argv[2])):
    if c not in drop:
        keep.append(c)
        drop.update(smt_siblings([c])[c])
print(",".join(map(str, keep)))
PYEOF
)
fi
n_phys=$(echo "$phys" | tr ',' '\n' | wc -l)
# Soft isolation (ISOLATE=1): OpenMP workers on the cores minus HOUSEKEEPING
# (SMT siblings idle); main thread (ISO_MAIN=housekeeping) and all our helper
# threads (MoE-Infinity task-pool/AIO, monitor) on HOUSEKEEPING.  Nothing
# outside this container is changed; see the guide for strict isolation.
ISOLATE="${ISOLATE:-0}"
HOUSEKEEPING="${HOUSEKEEPING:-0-7}"
ISO_MAIN="${ISO_MAIN:-housekeeping}"
ISOARGS=()
if [ "$ISOLATE" = "1" ]; then
    eval "$(python3 "$BENCH/cpu_layout.py" --cpus "$phys" --housekeeping "$HOUSEKEEPING" \
        --main "$ISO_MAIN" --threads "${CPU_THREADS:-0}")"
    CPU_THREADS="$OMP_NUM_THREADS"
    ISOARGS=(--noise-check-s "${NOISE_CHECK_S:-10}" --noise-abort-pct "${NOISE_ABORT_PCT:-30}"
             --retry-interfered "${RETRY:-2}" --monitor "${MONITOR:-auto}")
    echo "soft isolation: ${EP_N_COMPUTE} compute cores (${EP_COMPUTE_CPUS}), housekeeping ${EP_HOUSEKEEPING_CPUS}, main thread on ${EP_MAIN_THREAD}"
else
    CPU_THREADS="${CPU_THREADS:-$n_phys}"
    export OMP_NUM_THREADS="$CPU_THREADS"
    # Pin one OpenMP thread per physical core (explicit places, so this holds with
    # numactl or with the in-process binding). Unpinned spin-waiting threads on a
    # fully used core set get stacked and stall a time slice per barrier.
    # OMP_PIN=0 keeps the old unpinned behaviour (diagnostics only).
    if [ "${OMP_PIN:-1}" = "1" ] && [ -z "${OMP_PROC_BIND:-}${OMP_PLACES:-}${GOMP_CPU_AFFINITY:-}" ]; then
        places=$(echo "$phys" | tr ',' '\n' | head -n "$CPU_THREADS" | sed 's/.*/{&}/' | paste -sd, -)
        export OMP_PROC_BIND=close OMP_PLACES="$places"
    fi
    export EP_ALLOWED_CPUS="$(echo "$phys" | tr ',' '\n' | head -n "$CPU_THREADS" | paste -sd, -)"
    ISOARGS=(--noise-check-s "${NOISE_CHECK_S:-10}" --retry-interfered "${RETRY:-0}" --monitor "${MONITOR:-off}")
fi
export EP_REEXEC=1
echo "GPU NUMA node ${NUMA_NODE}; ${n_phys} physical cores (${phys}); ${CPU_THREADS} CPU threads; OMP_PROC_BIND=${OMP_PROC_BIND:-unset}"

PYBIND=()
if command -v numactl >/dev/null 2>&1 && numactl --membind="$NUMA_NODE" true 2>/dev/null; then
    BIND=(numactl --physcpubind="$phys" --membind="$NUMA_NODE")
else
    # No numactl (offline image): bind inside the process instead
    # (sched_setaffinity + set_mempolicy, needs --cap-add SYS_NICE).
    echo "numactl unavailable; binding in-process (--cpu-bind/--membind-node)" >&2
    BIND=(env)
    PYBIND=(--cpu-bind "$phys" --membind-node "$NUMA_NODE")
fi

COMMON=(--model-dir "$MODEL_DIR" --layer "$LAYER" --expert "$EXPERT" --cpu-threads "$CPU_THREADS"
        --precision "$PRECISION" --fetch-modes "$FETCH_MODES")
RUNARGS=("${COMMON[@]}" "${PYBIND[@]}" "${ISOARGS[@]}")
cd "$REPO"
python3 "$BENCH/expert_placement_bench.py" "${COMMON[@]}" --list-experts | tee "$OUT/list_experts.txt"
if [ "${SKIP_DRY_RUN:-0}" != "1" ]; then
    set +e
    "${BIND[@]}" python3 "$BENCH/expert_placement_bench.py" "${RUNARGS[@]}" --dry-run --noise-check-s 0 \
        --scenarios "$SCENARIOS" --gpu-kernels "$GPU_KERNELS" --cpu-kernels "$CPU_KERNELS" \
        --out-dir "$OUT/dry_run" "$@" 2>&1 | tee "$OUT/dry_run.log"
    dry_rc=${PIPESTATUS[0]}
    set -e
    [ "$dry_rc" -eq 0 ] || echo "WARNING: dry-run exited with $dry_rc (see $OUT/dry_run.log); continuing" >&2
fi
set +e
"${BIND[@]}" python3 "$BENCH/expert_placement_bench.py" "${RUNARGS[@]}" \
    --tokens "$TOKENS" --scenarios "$SCENARIOS" --gpu-kernels "$GPU_KERNELS" \
    --cpu-kernels "$CPU_KERNELS" --repeats "$REPEATS" --idle-mib "$IDLE_MIB" \
    --out-dir "$OUT" "$@" 2>&1 | tee "$OUT/run.log"
rc=${PIPESTATUS[0]}
set -e
[ "$rc" -eq 0 ] || echo "WARNING: benchmark exited with $rc (see $OUT/run.log)" >&2
# Results are written after every row, so summarize whatever exists even if
# the run ended badly.  Plots need matplotlib; without it the summary is
# Markdown only (plot the CSV/JSON locally with the same script).
if [ -f "$OUT/results.json" ]; then
    python3 "$BENCH/summarize_placement.py" "$OUT" > /dev/null || echo "WARNING: summary failed" >&2
    echo "results: $OUT/{results.csv,results.json,summary.md,*.png}"
else
    echo "ERROR: no $OUT/results.json" >&2
fi
# Outputs are created as root inside the container; hand them to the owner of
# the repo checkout (the host user) so they can be read/moved/tarred outside.
if [ "$(id -u)" = "0" ]; then
    owner="${OUT_OWNER:-$(stat -c %u:%g "$REPO")}"
    if [ "$owner" != "0:0" ]; then
        chown -R "$owner" "$OUT" "$(dirname "$OUT")" 2>/dev/null || true
    fi
fi
exit "$rc"
