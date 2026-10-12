#!/usr/bin/env bash
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
#
# Restorable CPU isolation on the shared docker host (cgroup v1).  Run on the
# HOST as root.  Default: option A = physical cores 80-95 + SMT siblings.
#
#   sudo bash isolate_cpus.sh plan    [--option A|--cpus 80-95] [--ours $NAME]   # dry run
#   sudo bash isolate_cpus.sh apply   --ours $NAME [--yes]
#   sudo bash isolate_cpus.sh status
#   sudo bash isolate_cpus.sh restore                       # idempotent
#
# Option B (--option B: 8-95) squeezes every other tenant and needs admin
# approval (--admin-approved).  See isolate_cpus.py for what is changed.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cmd="${1:-plan}"
if [ "$cmd" != "plan" ] && [ "$(id -u)" != "0" ] && [ -z "${MI_ISOLATE_TEST:-}" ]; then
    echo "run as root: sudo bash $0 $*" >&2
    exit 1
fi
if [ "$cmd" = "apply" ]; then
    trap 'echo "!!! isolate_cpus.sh apply interrupted; undo with: sudo bash $HERE/isolate_cpus.sh restore" >&2' INT TERM ERR
fi
python3 "$HERE/isolate_cpus.py" "$@"
rc=$?
if [ "$cmd" = "apply" ] && [ "$rc" -ne 0 ]; then
    echo "!!! apply exited with $rc; if anything was changed: sudo bash $HERE/isolate_cpus.sh restore" >&2
fi
exit "$rc"
