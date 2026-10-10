# Single-expert placement benchmark

`benchmarks/expert_placement/` pins one routed expert `(layer, expert)` of a
HuggingFace safetensors MoE checkpoint and measures three placements for each
token count M, where M is the number of rows routed to that expert:

| Scenario | Weights live in | Computed on | Overhead | Compute |
| --- | --- | --- | --- | --- |
| `gpu_resident` (S1) | GPU memory | GPU | `total - compute`: Python dispatch, kernel launches, sync | CUDA-graph replay of the expert, timed with CUDA events |
| `cpu_compute` (S2) | MoE-Infinity's pinned host pool | CPU | activations GPU→pinned host (`d2h`) + result host→GPU (`h2d`) | CPU kernel wall time; `act_quant_ms` is the FP8 activation rounding inside it |
| `cpu_store_gpu_compute` (S3) | MoE-Infinity's pinned host pool | GPU | `xfer`: MoE-Infinity's own move of the expert node, plus `release` and launch | GPU kernel right after the move |

`overhead_ms = total_ms - compute_ms` throughout. Values are medians, with
p10/p90 alongside. Before every timed repeat the GPU L2 and the CPU LLC are
flushed, outside the timed region.

The following one-time costs are kept under `setup` and never enter
per-token numbers:

- checkpoint read and BF16 dequantization;
- layout preparation and CPU packing;
- store offload and topology init;
- upload and pinned allocation.

## Precision

| | `--precision fp8` (default) | `--precision bf16` |
| --- | --- | --- |
| Expert bytes (Hy3: K=4096, N=1536) | The checkpoint's FP8 e4m3 codes, about 18 MiB, in every scenario | The chosen Hy3-FP8 expert dequantized once to BF16, about 36 MiB, in every scenario |
| GPU kernels (default) | `sglang_triton` (SGLang Triton FP8 W8A8 `fused_experts`), `torch_scaled_mm` (cuBLASLt FP8) | `batchgen_triton` (PR #1's BatchGen `fused_moe_bf16`, JIT-compiled by Triton for the running GPU), `torch_bf16` (cuBLAS reference), `sglang_triton` (BF16) |
| CPU kernels (default) | `sglang_fp8_w8a16`, `sglang_fp8_w8a8_emu` | `sglang_bf16` (AMX-BF16 / AVX512-BF16, no dequantization) |

In FP8 mode, BF16 and INT8 kernels are opt-in only, and the output labels them
as non-FP8 comparisons.

**BF16 dequantization.** Each projection is computed as
`fp8_code × that projection's own weight_scale` and rounded once to BF16.
Gate and up keep their separate scales. This differs from the FP8 GPU path,
which follows SGLang's `Fp8MoEMethod`: it requantizes gate and up to their
shared maximum scale. The source is recorded in `setup.expert_source`.

**CPU FP8 kernels.** x86 CPUs have no FP8 matmul units, so both CPU kernels
keep the e4m3 weights in memory and widen them to BF16 inside the kernel.

- `sglang_fp8_w8a16` takes BF16 activations.
- `sglang_fp8_w8a8_emu` reproduces W8A8 numerics:
  1. It uses the same FP8 weight codes as the GPU kernels.
  2. It rounds activations to e4m3 with the same static `input_scale` and
     clamp. This applies to `x` and to `act(gate)·up`.
  3. It runs SGLang's `fp8_scaled_mm_cpu`, with the activation scale folded
     into the weight block scales.

  It is not faster than W8A16.

`rel_err_w8a8` compares W8A8 kernels with an FP32 W8A8 reference. The
`parity` section of `results.json` reports kernel-to-kernel differences on
identical inputs, for example CPU emulation against GPU W8A8.

## How the expert moves (MoE-Infinity, not a plain copy)

`mi_expert_store.py` builds a one-MoE-layer MoE-Infinity engine from the
`moe_infinity._store` extension, using the same calls `OffloadEngine` makes:

1. `prefetch_handle` creates the engine.
2. `offload` writes each tensor to the offload store.
3. `register` registers the placeholder (`param.data`).
4. `set_topology_v2` lays out the stages, using `build_topology_specs`.

Topology init reads every expert node into `kHostMemoryPool`, which is pinned
and holds one contiguous buffer per node, and re-points the registered
tensors at that buffer.

S3 then uses one of these modes:

| Mode | Path |
| --- | --- |
| `mi_fetch` | `begin()`: `AcquireTensor` does the sparse-cache bookkeeping, then an `ArcherTaskPool` on-demand task, then `Node::SetDevice`. That allocates from the device pool, runs `cudaMemcpyAsync` from the pinned buffer on the H2D stream, syncs on an event and re-points the tensors. `end()` follows, as in the post-forward hook. |
| `mi_fetch_evict` | Same as `mi_fetch`, but with the expert cache full: `RemoveCachedSparseNode` evicts the other cached expert first. |
| `mi_prefetch` | `prefetch_tensors()` (`EnqueuePrefetchTensors`), then waits until the node is resident. |
| `raw_pinned`, `raw_pageable` | Reference lines only: a torch `copy_` of the same bytes. |

Between repeats, `resize_expert_cache(dev, 0)` evicts the node back to the
host pool. `mi_h2d_bytes` confirms the bytes went through
`Node::SetDevice`.

This path is the same as the runtime's demand fetch in `ExpertDispatcher` from
`Node::SetDevice` down: the same host pool, device pool, H2D copy and view
re-pointing. Two differences remain:

- The dispatcher uses its own cache-slot accounting and hands the transfer
  event to its exec thread.
- That exec thread computes with MoE-Infinity's `MoEMLP`, which dequantizes
  FP8 to BF16 and runs CUTLASS.

To keep the compute FP8, the harness drives the task-pool and prefetch entry
points and runs the chosen kernel itself.

## Build inside the SGLang v0.5.19 image

The image ships torch 2.13.0+cu130. Only `_store` and `_cpu_moe` are needed:

```bash
pip install --no-deps 'moe-store @ git+https://github.com/EfficientMoE/moe-store.git@v0.2.2'
apt-get install -y uuid-dev && git clone --depth 1 -b v3.9.2 https://github.com/NVIDIA/cutlass /scratch/cutlass
MOE_ENABLE_SM120=1 MOE_ENABLE_SM90=0 MOE_BUILD_BATCHGEN=0 TORCH_CUDA_ARCH_LIST=12.0 \
CUTLASS_DIR=/scratch/cutlass MAX_JOBS=16 python benchmarks/expert_placement/build_extensions.py
```

`setup.py` warns that sm_120 is validated only on torch 2.12 (#245). That
warning is about `fused_moe_ffn_into`, the CUTLASS fused MLP in `MoEMLP`,
which this benchmark never calls. The store, pools, task pool and fetch code
are not tied to a torch version.

### Offline servers (no GitHub, no PyPI)

On a machine with network access (macOS or Linux), run:

```bash
bash benchmarks/expert_placement/make_offline_bundle.sh
```

This writes one `moe-ep-offline-<date>.tar.gz`, about 6 MB:

| Path | Contents |
| --- | --- |
| `MoE-Infinity/` | The repo via `git archive`, without `.git` |
| `cutlass/` | CUTLASS v3.9.2, only `include/` and `tools/util/include/`. These are the only CUTLASS dirs `setup.py` puts on `-I` |
| `moe-store/` | moe-store v0.2.2 source |
| `wheels/` | `ninja` and `setuptools-scm` |
| `uuid/` | libuuid's `uuid.h`, for images without uuid-dev |

Upload it with `rz -be` or `scp`, extract it under `/data1/scratch/<user>/`, and run inside the container:

```bash
bash <bundle>/MoE-Infinity/benchmarks/expert_placement/offline_setup.sh <bundle>
source <bundle>/env.sh
```

`offline_setup.sh` does the following, with `PIP_NO_INDEX=1` throughout:

1. Installs ninja and setuptools-scm from `wheels/`, only if they are missing.
2. Runs `pip install --no-deps --no-build-isolation <bundle>/moe-store`. The
   build itself reads `MOE_STORE_CSRC` from the bundle, so it does not depend
   on this install succeeding.
3. Falls back to the bundled `uuid.h` and the system `libuuid.so.1` when
   uuid-dev is missing.
4. Builds `_store` and `_cpu_moe` for sm_120.

`numactl` and `matplotlib` are optional:

- Without numactl, `run_sweep.sh` binds in-process (`--cpu-bind`,
  `--membind-node`), using `sched_setaffinity` and `set_mempolicy`.
- Without matplotlib, the summary is Markdown only. Run the same
  `summarize_placement.py` locally to plot from the CSV/JSON.

## Usage

```bash
python benchmarks/expert_placement/expert_placement_bench.py --model-dir /model --list-experts
MODEL_DIR=/model LAYER=1 EXPERT=0 bash benchmarks/expert_placement/run_sweep.sh                  # FP8
MODEL_DIR=/model LAYER=1 EXPERT=0 PRECISION=bf16 bash benchmarks/expert_placement/run_sweep.sh   # BF16
python benchmarks/expert_placement/summarize_placement.py /scratch/expert_placement/<run>
```

`run_sweep.sh` first checks that exactly one idle GPU is visible and that
`_store` is built. It then binds to the GPU's NUMA node with `numactl`, which
needs `--cap-add SYS_NICE` inside Docker, and runs list, dry-run, sweep and
summary.

## Verification status

On a CPU-only VM (x86 with AVX512-BF16, AMX disabled):

- **Build:** `_store` and `_cpu_moe` compile and link against torch
  2.13.0+cu130 with CUDA 13.0, `MOE_ENABLE_SM120=1`, emitting sm_120 code.
  `_store` imports and exposes every binding the harness calls.
- **Harness:** tested against a CPU stand-in that mirrors the `core/`
  semantics.
- **Kernels:**
  - CPU kernels, the BF16 dequantization and the W8A8 emulation were checked
    against references. The emulation is within about 1e-2 of the GPU W8A8
    math, and exactly equal at small M.
  - BatchGen's wiring was checked under the Triton interpreter in FP32.
  - The SGLang adapter was checked against a stand-in with v0.5.19's
    signatures.

The real engine stops at its first `cudaPointerGetAttributes` without a
driver. Nothing has run on a GPU yet, so every GPU timing, the real
MoE-Infinity fetch, and BatchGen/SGLang on sm_120 are unverified. Tests are
in `tests/python/unit/test_expert_placement_bench.py`.
