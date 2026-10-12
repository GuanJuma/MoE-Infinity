# Copyright (c) EfficientMoE.
# SPDX-License-Identifier: Apache-2.0

"""Build only the MoE-Infinity extensions this benchmark needs, in place.

    MOE_ENABLE_SM120=1 MOE_ENABLE_SM90=0 CUTLASS_DIR=... \\
        python benchmarks/expert_placement/build_extensions.py [--ext _store _cpu_moe]

Runs the repository's own ``setup.py`` (same sources, flags and arch
selection as ``pip install -e .``) but keeps only the selected extensions and
writes them next to the sources (``build_ext --inplace``), so
``PYTHONPATH=<repo>`` is enough and the image's torch / sglang-kernel are
left untouched.  ``_store`` holds the expert store, pinned host pool, device
pool, task pool and fetch/prefetch code; ``_cpu_moe`` the vendored SGLang
CPU kernels (otherwise JIT-built on first use).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ext", nargs="+", default=["_store", "_cpu_moe"])
    a = p.parse_args(argv)
    keep = {f"moe_infinity.{e.lstrip('.')}" for e in a.ext}

    import setuptools

    orig = setuptools.setup

    def only_selected(**kw):
        exts = [e for e in kw.get("ext_modules", []) if e.name in keep]
        missing = keep - {e.name for e in exts}
        if missing:
            raise SystemExit(
                f"setup.py did not configure {sorted(missing)} (no CUDA torch for "
                f"_store? MOE_BUILD_CPU_MOE=0 for _cpu_moe?)"
            )
        print("building:", [e.name for e in exts], flush=True)
        kw["ext_modules"] = exts
        return orig(**kw)

    setuptools.setup = only_selected
    os.chdir(REPO)
    sys.argv = ["setup.py", "build_ext", "--inplace"]
    setup_py = REPO / "setup.py"
    code = compile(setup_py.read_text(), str(setup_py), "exec")
    exec(code, {"__name__": "__main__", "__file__": str(setup_py)})


if __name__ == "__main__":
    main()
