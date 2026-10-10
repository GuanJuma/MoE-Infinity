#!/usr/bin/env bash
# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0
#
# Run LOCALLY (macOS or Linux, with internet) to pack everything the server
# needs for the expert-placement benchmark into one tar.gz.  The server then
# installs/builds fully offline with offline_setup.sh from the bundle.
#
#   bash benchmarks/expert_placement/make_offline_bundle.sh            # -> ./moe-ep-offline-<date>.tar.gz
#   REF=cursor/hy3-expert-placement-bench-5c8f OUT=/tmp/b.tar.gz bash .../make_offline_bundle.sh
#
# Contents (top dir moe-ep-offline/):
#   MoE-Infinity/       repo at $REF (git archive, no .git)
#   cutlass/            CUTLASS v3.9.2 headers only: include/ + tools/util/include/
#                       (the two dirs setup.py / core/CMakeLists.txt put on -I)
#   moe-store/          moe-store v0.2.2 source (pip-installable; csrc used by the build)
#   wheels/             ninja + setuptools-scm (and deps), py3 manylinux x86_64 / any
#   uuid/include/uuid/  libuuid's uuid.h (util-linux, BSD-3-Clause), for images
#                       without uuid-dev
#   MANIFEST.txt, SHA256SUMS
#
# Needs: git, curl, tar, python3 with pip.  Optional pre-downloaded inputs
# (if GitHub is slow locally): CUTLASS_TGZ, MOE_STORE_TGZ, UTIL_LINUX_TXZ.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REF="${REF:-HEAD}"
CUTLASS_TAG="${CUTLASS_TAG:-v3.9.2}"
MOE_STORE_TAG="${MOE_STORE_TAG:-v0.2.2}"
UTIL_LINUX_VER="${UTIL_LINUX_VER:-2.40.2}"
PYVER="${PYVER:-3.12}"
OUT="${OUT:-$PWD/moe-ep-offline-$(date +%Y%m%d).tar.gz}"
export COPYFILE_DISABLE=1   # macOS tar: no ._ AppleDouble files

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
B="$WORK/moe-ep-offline"
mkdir -p "$B/wheels" "$B/uuid/include/uuid" "$WORK/dl"

fetch() {  # fetch URL DEST [PREDOWNLOADED]
    if [ -n "${3:-}" ]; then
        cp "$3" "$2"
    else
        echo "  downloading $1"
        curl -fL --retry 3 -o "$2" "$1"
    fi
}

echo "[1/5] repo $REF (git archive, no .git)"
git -C "$REPO" archive --format=tar --prefix=MoE-Infinity/ "$REF" | tar -x -C "$B"
git -C "$REPO" rev-parse "$REF" > "$B/MoE-Infinity/.offline-bundle-commit"

echo "[2/5] CUTLASS $CUTLASS_TAG headers"
fetch "https://github.com/NVIDIA/cutlass/archive/refs/tags/$CUTLASS_TAG.tar.gz" \
    "$WORK/dl/cutlass.tgz" "${CUTLASS_TGZ:-}"
tar -xzf "$WORK/dl/cutlass.tgz" -C "$WORK/dl"
CUT_SRC="$(find "$WORK/dl" -maxdepth 1 -type d -name 'cutlass-*' | head -1)"
mkdir -p "$B/cutlass/tools/util"
cp -R "$CUT_SRC/include" "$B/cutlass/include"
cp -R "$CUT_SRC/tools/util/include" "$B/cutlass/tools/util/include"
cp "$CUT_SRC/LICENSE.txt" "$B/cutlass/"
test -f "$B/cutlass/include/cutlass/cutlass.h"
test -d "$B/cutlass/include/cute"
test -d "$B/cutlass/tools/util/include/cutlass/util"

echo "[3/5] moe-store $MOE_STORE_TAG source"
fetch "https://github.com/EfficientMoE/moe-store/archive/refs/tags/$MOE_STORE_TAG.tar.gz" \
    "$WORK/dl/moe-store.tgz" "${MOE_STORE_TGZ:-}"
mkdir -p "$B/moe-store"
tar -xzf "$WORK/dl/moe-store.tgz" -C "$B/moe-store" --strip-components 1
test -f "$B/moe-store/moe_store/csrc/store/index_v2.h"
echo "${MOE_STORE_TAG#v}" > "$B/moe-store/.offline-version"

echo "[4/5] wheels (ninja, setuptools-scm) for Linux x86_64 / Python $PYVER"
python3 -m pip download --quiet --only-binary=:all: --dest "$B/wheels" \
    --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 \
    --platform manylinux_2_28_x86_64 --python-version "$PYVER" --implementation cp \
    ninja setuptools-scm
ls "$B/wheels" | grep -q '^ninja-.*manylinux.*x86_64\.whl$'

echo "[5/5] libuuid header (util-linux $UTIL_LINUX_VER)"
UL_MINOR="${UTIL_LINUX_VER%.*}"
fetch "https://mirrors.edge.kernel.org/pub/linux/utils/util-linux/v$UL_MINOR/util-linux-$UTIL_LINUX_VER.tar.xz" \
    "$WORK/dl/util-linux.tar.xz" "${UTIL_LINUX_TXZ:-}"
tar -xJf "$WORK/dl/util-linux.tar.xz" -C "$WORK/dl" \
    "util-linux-$UTIL_LINUX_VER/libuuid/src/uuid.h" \
    "util-linux-$UTIL_LINUX_VER/Documentation/licenses/COPYING.BSD-3-Clause"
cp "$WORK/dl/util-linux-$UTIL_LINUX_VER/libuuid/src/uuid.h" "$B/uuid/include/uuid/uuid.h"
cp "$WORK/dl/util-linux-$UTIL_LINUX_VER/Documentation/licenses/COPYING.BSD-3-Clause" "$B/uuid/COPYING"

{
    echo "moe-ep-offline bundle, built $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(uname -sm)"
    echo "MoE-Infinity  $REF $(cat "$B/MoE-Infinity/.offline-bundle-commit")"
    echo "CUTLASS       $CUTLASS_TAG (include/, tools/util/include/ only)"
    echo "moe-store     $MOE_STORE_TAG"
    echo "util-linux    $UTIL_LINUX_VER (libuuid/src/uuid.h only)"
    echo "wheels:"
    ls "$B/wheels" | sed 's/^/  /'
    echo
    echo "On the server (inside the container):"
    echo "  tar xzf $(basename "$OUT") -C /data1/scratch/<you>/"
    echo "  bash /scratch/<you>/moe-ep-offline/MoE-Infinity/benchmarks/expert_placement/offline_setup.sh \\"
    echo "       /scratch/<you>/moe-ep-offline"
} > "$B/MANIFEST.txt"
if command -v shasum >/dev/null; then SHA=(shasum -a 256); else SHA=(sha256sum); fi
(cd "$B" && find . -type f ! -name SHA256SUMS | LC_ALL=C sort | xargs "${SHA[@]}" > SHA256SUMS)

tar -czf "$OUT" -C "$WORK" moe-ep-offline
echo "bundle: $OUT ($(du -h "$OUT" | cut -f1))"
cat "$B/MANIFEST.txt"
