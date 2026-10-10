#!/usr/bin/env bash
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
#
# Repeat protocol for S2 (inside the container): ROUNDS S2-only sweeps per
# precision, GAP_S seconds apart, through run_sweep.sh (so ISOLATE, CORES,
# HOUSEKEEPING, ... apply), then per-M min-of-medians and spread.
#
#   ISOLATE=1 MODEL_DIR=/model OUT=/scratch/$ME/ep/s2_rounds bash benchmarks/expert_placement/s2_rounds.sh
#
# Env: ROUNDS (3) GAP_S (300) PRECISIONS ("fp8 bf16") LEG (optional sub-directory,
#      used by ab_isolation.sh) + everything run_sweep.sh reads.
set -uo pipefail
BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-/scratch/expert_placement/s2_rounds_$(date +%Y%m%d-%H%M%S)}"
ROUNDS="${ROUNDS:-3}"
GAP_S="${GAP_S:-300}"
PRECISIONS="${PRECISIONS:-fp8 bf16}"
FIRST="${FIRST_ROUND:-1}"
export SCENARIOS=cpu SKIP_DRY_RUN=1
mkdir -p "$OUT"
last=$((FIRST + ROUNDS - 1))
for r in $(seq "$FIRST" "$last"); do
    for P in $PRECISIONS; do
        dir="$OUT/$P/${LEG:+$LEG/}round_$r"
        mkdir -p "$(dirname "$dir")"
        echo "=== round $r / $P -> $dir ($(date +%H:%M:%S))"
        PRECISION="$P" OUT="$dir" bash "$BENCH/run_sweep.sh" "$@" > "$dir.log" 2>&1 \
            || echo "    exit $? (see $dir.log)"
        grep -E "noise check|INTERFERENCE|soft isolation" "$dir.log" | head -5 | sed 's/^/    /'
    done
    if [ "$r" -lt "$last" ] && [ "$GAP_S" -gt 0 ]; then
        echo "    sleeping ${GAP_S}s"
        sleep "$GAP_S"
    fi
done
python3 "$BENCH/summarize_rounds.py" "$OUT" ${AB:+--ab $AB} > "$OUT/rounds_summary.md"
echo "summary: $OUT/rounds_summary.md"
if [ "$(id -u)" = "0" ]; then
    owner="${OUT_OWNER:-$(stat -c %u:%g "$BENCH")}"
    [ "$owner" != "0:0" ] && chown -R "$owner" "$OUT" 2>/dev/null
fi
exit 0
