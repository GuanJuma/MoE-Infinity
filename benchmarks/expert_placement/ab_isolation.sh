#!/usr/bin/env bash
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
#
# A/B on the HOST (root): S2 for both precisions on the same cores with the
# same thread count, once with the cores shared (as usual) and once isolated
# by isolate_cpus.sh, ROUNDS times; leg order alternates per round and
# isolation is always restored (also on Ctrl-C / errors).
#
#   sudo NAME=hy3-expert-bench-$ME ME=$ME bash benchmarks/expert_placement/ab_isolation.sh
#
# Env: NAME (our container, required) ME ROUNDS (3) GAP_S (300) PRECISIONS ("fp8 bf16")
#      CPUS (80-95: isolated physical cores; SMT siblings added) HK (72-79: our
#      main/helper threads, not isolated) REPO_C (repo inside the container;
#      default: the offline bundle or /scratch/$ME/MoE-Infinity) OUT_C (results
#      inside the container) HOST_SCRATCH (/data1/scratch = /scratch in the container)
#      TOKENS REPEATS LAYER EXPERT  ISO_ARGS (extra isolate_cpus.sh args)  BENCH_ARGS (extra benchmark args)
#      DOCKER  HOST_OUT
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${NAME:?set NAME to our container}"
ME="${ME:-${SUDO_USER:-$(whoami)}}"
ROUNDS="${ROUNDS:-3}"
GAP_S="${GAP_S:-300}"
PRECISIONS="${PRECISIONS:-fp8 bf16}"
CPUS="${CPUS:-80-95}"
HK="${HK:-72-79}"
# The OpenMP master does 1/T of every GEMM: keep it on an isolated core
# (first compute place) so both legs run T = #CPUS threads entirely on CPUS;
# HK only hosts MoE-Infinity's task-pool/AIO threads and the monitor.
ISO_MAIN="${ISO_MAIN:-compute}"
OUT_C="${OUT_C:-/scratch/$ME/ep/ab_L${LAYER:-1}E${EXPERT:-0}_$(date +%Y%m%d-%H%M%S)}"
HOST_SCRATCH="${HOST_SCRATCH:-/data1/scratch}"
HOST_OUT="${HOST_OUT:-$HOST_SCRATCH/${OUT_C#/scratch/}}"
DOCKER="${DOCKER:-docker}"
STATE="${STATE:-/var/tmp/mi_isolate_cpus.ab.state.json}"
iso() {  # iso CMD [args]: every isolate_cpus.sh call shares the state file and ISO_ARGS
    local c="$1"; shift
    bash "$HERE/isolate_cpus.sh" "$c" --cpus "$CPUS" --ours "$NAME" --state "$STATE" \
        --docker "$DOCKER" ${ISO_ARGS:-} "$@"
}
if [ -z "${REPO_C:-}" ]; then
    REPO_C=$($DOCKER exec "$NAME" bash -c "for d in /scratch/$ME/hy3-bundle/MoE-Infinity /scratch/$ME/MoE-Infinity; do [ -f \$d/benchmarks/expert_placement/s2_rounds.sh ] && echo \$d && break; done")
fi
[ -n "$REPO_C" ] || { echo "REPO_C: repo with s2_rounds.sh not found in $NAME" >&2; exit 2; }
mkdir -p "$HOST_OUT"
LOG="$HOST_OUT/ab_host.log"
exec > >(tee -a "$LOG") 2>&1
echo "A/B isolation: container $NAME, repo $REPO_C, isolated $CPUS (+SMT), housekeeping $HK, rounds $ROUNDS, out $OUT_C"

applied=0
cleanup() {
    if [ "$applied" = "1" ]; then
        echo "restoring CPU isolation (exit/interrupt)"
        iso restore
        applied=0
    fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

iso plan > "$HOST_OUT/isolate_plan.txt" || exit 3
grep -q "CONFLICT" "$HOST_OUT/isolate_plan.txt" && { cat "$HOST_OUT/isolate_plan.txt"; exit 3; }

leg() {  # leg NAME ROUND
    $DOCKER exec -e ME="$ME" "$NAME" bash -c "
        cd '$REPO_C' && { [ -f ../env.sh ] && . ../env.sh; true; } &&
        ISOLATE=1 CORES='$HK,$CPUS' HOUSEKEEPING='$HK' ISO_MAIN='$ISO_MAIN' NOISE_ABORT_PCT=0 RETRY=0 \
        MODEL_DIR=\${MODEL_DIR:-/model} LAYER='${LAYER:-1}' EXPERT='${EXPERT:-0}' \
        ${TOKENS:+TOKENS='$TOKENS'} ${REPEATS:+REPEATS='$REPEATS'} \
        PRECISIONS='$PRECISIONS' ROUNDS=1 FIRST_ROUND=$2 GAP_S=0 LEG=$1 OUT='$OUT_C' \
        bash benchmarks/expert_placement/s2_rounds.sh ${BENCH_ARGS:-}"
}

for r in $(seq 1 "$ROUNDS"); do
    if [ $((r % 2)) -eq 1 ]; then order="shared isolated"; else order="isolated shared"; fi
    for L in $order; do
        echo "=== round $r: $L ($(date +%H:%M:%S))"
        if [ "$L" = "isolated" ]; then
            iso apply --yes \
                > "$HOST_OUT/isolate_apply_r$r.txt" || { cat "$HOST_OUT/isolate_apply_r$r.txt"; exit 4; }
            applied=1
            tail -3 "$HOST_OUT/isolate_apply_r$r.txt"
            iso status \
                > "$HOST_OUT/isolate_status_r$r.txt" 2>&1
            grep -E "busy|tasks whose" "$HOST_OUT/isolate_status_r$r.txt"
        fi
        leg "$L" "$r"
        if [ "$L" = "isolated" ]; then
            iso restore > "$HOST_OUT/isolate_restore_r$r.txt" 2>&1
            applied=0
            tail -2 "$HOST_OUT/isolate_restore_r$r.txt"
        fi
    done
    if [ "$r" -lt "$ROUNDS" ] && [ "$GAP_S" -gt 0 ]; then
        echo "    sleeping ${GAP_S}s"
        sleep "$GAP_S"
    fi
done
$DOCKER exec "$NAME" bash -c "cd '$REPO_C' && python3 benchmarks/expert_placement/summarize_rounds.py '$OUT_C' --ab shared isolated > '$OUT_C/ab_summary.md'"
iso status > "$HOST_OUT/isolate_status_final.txt" 2>&1
owner="${OUT_OWNER:-$(stat -c %u:%g "$HOST_SCRATCH/$ME" 2>/dev/null || echo 0:0)}"
[ "$owner" != "0:0" ] && chown -R "$owner" "$HOST_OUT" 2>/dev/null
echo "A/B summary: $HOST_OUT/ab_summary.md"
