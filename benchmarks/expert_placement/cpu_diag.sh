#!/usr/bin/env bash
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
#
# A/B diagnostics for the CPU scenario (S2): every variant is one S2-only run
# through run_sweep.sh (same NUMA/thread binding as the real sweep), then
# summarize_diag.py prints all variants side by side.
#
#   MODEL_DIR=/model OUT=/scratch/$ME/ep/cpu_diag bash benchmarks/expert_placement/cpu_diag.sh
#
# Env: PRECISIONS ("fp8 bf16"), TOKENS, REPEATS, VARIANTS (subset, space separated),
#      EXTRA_ARGS (appended to every benchmark call)
set -uo pipefail

BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-/scratch/expert_placement/cpu_diag_$(date +%Y%m%d-%H%M%S)}"
PRECISIONS="${PRECISIONS:-fp8 bf16}"
export TOKENS="${TOKENS:-1,2,4,5,8,16,64,512,4096}"
export REPEATS="${REPEATS:-15}"
export SCENARIOS=cpu SKIP_DRY_RUN=1
ALL="base cpuw_torch no_engine omp_bind omp_passive no_flush skip_cpu0_7 threads_48 malloc_off avx2_isa onednn_verbose"
VARIANTS="${VARIANTS:-$ALL}"
mkdir -p "$OUT"

run() {  # run NAME PRECISION [VAR=value ...] -- [bench args ...]
    local name="$1" prec="$2"
    shift 2
    local envs=()
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
    [ "${1:-}" = "--" ] && shift
    echo "=== $prec / $name: ${envs[*]:-} $*"
    env "${envs[@]}" PRECISION="$prec" OUT="$OUT/$prec/$name" \
        bash "$BENCH/run_sweep.sh" --max-seconds-per-point 8 "$@" ${EXTRA_ARGS:-} \
        > "$OUT/$prec.$name.log" 2>&1
    echo "    exit $? -> $OUT/$prec/$name"
}

for P in $PRECISIONS; do
    for V in $VARIANTS; do
        case "$V" in
            base)           run base "$P" ;;
            cpuw_torch)     run cpuw_torch "$P" -- --cpu-weights torch ;;
            no_engine)      run no_engine "$P" REQUIRE_STORE=0 -- --expert-store torch ;;
            omp_bind)       run omp_bind "$P" OMP_PROC_BIND=close OMP_PLACES=cores ;;
            omp_passive)    run omp_passive "$P" OMP_WAIT_POLICY=PASSIVE ;;
            no_flush)       run no_flush "$P" -- --cpu-llc-flush-mb 0 ;;
            skip_cpu0_7)    run skip_cpu0_7 "$P" EXCLUDE_CPUS=0-7 ;;
            threads_48)     run threads_48 "$P" CPU_THREADS=48 ;;
            malloc_off)     run malloc_off "$P" -- --malloc-reuse off ;;
            avx2_isa)       run avx2_isa "$P" ONEDNN_MAX_CPU_ISA=AVX2 ;;
            onednn_verbose) run onednn_verbose "$P" ONEDNN_VERBOSE=1 TOKENS=4,8 REPEATS=2 ;;
            *) echo "unknown variant $V" >&2 ;;
        esac
    done
done
python3 "$BENCH/summarize_diag.py" "$OUT" | tee "$OUT/diag_summary.md"
if [ "$(id -u)" = "0" ]; then
    owner="${OUT_OWNER:-$(stat -c %u:%g "$BENCH")}"
    [ "$owner" != "0:0" ] && chown -R "$owner" "$OUT" 2>/dev/null
fi
echo "diagnostics: $OUT/diag_summary.md (pack: tar czf cpu_diag.tgz -C $(dirname "$OUT") $(basename "$OUT"))"
