# BatchGen expert kernels

MoE-Infinity can run the on-GPU FFN of each fetched expert on the MoE kernels
from [BatchGen](https://github.com/batchgen-project/batchgen) (Apache-2.0)
instead of its CUTLASS fused MLP. Expert offloading, fetch, prefetch and the
activation-aware cache are unchanged: the switch only affects what happens
after an expert's weights are resident on the GPU.

## Enabling it

```bash
export MOE_EXPERT_KERNEL=batchgen        # or "default"
```

or per engine:

```python
model = MoE(checkpoint, {"offload_path": "...", "expert_kernel": "batchgen"})
```

`ArcherConfig.expert_kernel` (`None` | `"default"` | `"batchgen"`) wins over the
environment variable when set. `None` follows `MOE_EXPERT_KERNEL`.

Inspect what the native engine actually ran:

```python
from moe_infinity import _store
_store.batchgen_expert_kernel_stats()
# {'compiled_in': True, 'compiled_archs': [80, 90], 'loaded_archs': [90],
#  'active_kernel': 'batchgen', 'expert_calls': ..., 'packed_gate_up_calls': ...,
#  'fallback_calls': 0, 'last_fallback_reason': ''}
```

## What runs

BatchGen's portable BF16 MoE path is the Triton kernel set in
`batchgen_kernels/triton/fused_moe_bf16.py`: a grouped GEMM over
expert-sorted token slots (`_fused_moe_gemm`), `silu_and_mul`, a second grouped
GEMM, and a weighted top-k reduce. Its CUDA MoE kernels are INT4 / MXFP4 / FP8
WGMMA kernels that target SM90a only, so they are not used here.

MoE-Infinity computes experts one at a time in C++ dispatcher threads, as
each one lands on the GPU. So each expert runs BatchGen's grouped GEMM in its
single-expert form:

1. Stage 1 `_fused_moe_gemm` writes `[gate | up]` into a `[M, 2I]` scratch.
   When `up` directly follows `gate` in memory (BatchGen's packed `w13`
   layout), this is a single GEMM. Otherwise it is two GEMMs into the two halves.
2. `_silu_and_mul_kernel` computes `[M, I]`.
3. Stage 2 `_fused_moe_gemm` against `down` writes the expert output.
4. The router-weighted reduce stays MoE-Infinity's existing `index_add_` in
   `ExpertDispatcher::OutputFunc`. That reduce replaces BatchGen's
   `moe_weighted_sum`.

Token order is the identity, every M-tile maps to expert 0, `top_k = 1`, and
BatchGen's config rule picks the tile size: a 16-row M-tile up to 1024 rows,
64 rows above that. Rows beyond the scratch capacity (4096 for the gate/up
buffer) run in chunks.

## Build

`setup.py` runs `moe_infinity/kernel/batchgen/aot.py`, which compiles the
vendored Triton kernels to cubins for every enabled SM arch (sm_80, plus
sm_90 with `MOE_ENABLE_SM90=1` and sm_120 with `MOE_ENABLE_SM120=1`). No
GPU is needed for this step because Triton ships `ptxas`. The script writes
`build/batchgen_aot/batchgen_moe_cubins.h`, which is compiled into `_store`
with `-DMOE_WITH_BATCHGEN=1`. `core/model/batchgen_moe.cpp` loads the cubins
per device through the CUDA driver API and launches them on the dispatcher's
exec stream. Python is not involved at runtime.

`MOE_BUILD_BATCHGEN` controls the build step. When unset, the cubins are
built if Triton can compile them. `"1"` makes a failure fatal, and `"0"`
skips the step. CMake builds use the `MOE_BUILD_BATCHGEN` option.

## Fallback rules

`batchgen::ExpertFFN` returns without enqueueing work, and the CUTLASS kernel
runs instead, when any of these holds:

- the build has no cubins;
- no cubin matches the device's SM major version with a minor version at or
  below the device's;
- tensors are not BF16, contiguous, and 16-byte aligned;
- the hidden or intermediate size is not a multiple of 16.

The first occurrence of each reason is logged, and `fallback_calls` counts
every fallback. GPT-OSS (MXFP4, clamped SwiGLU with biases) and NLLB / FSGPT
(ReLU with biases) experts always use their existing paths.

## Validation

| Check | Where | Needs a GPU |
| --- | --- | --- |
| Single-expert decomposition vs FP64 reference: FP32/FP16 (and BF16 on GPU), packed and split gate/up, `EVEN_K` on and off, small and large M-tiles | `tests/python/unit/test_batchgen_expert_kernel.py` | No, with `TRITON_INTERPRET=1` |
| MoE-Infinity's per-expert dispatch + `index_add_` reduce == BatchGen's grouped `fused_moe_bf16` | same | No, with `TRITON_INTERPRET=1` |
| Kernel-parameter order == the native launcher's packing; offline cubin build | same | No |
| Native launcher vs `fused_moe_ffn_into` on DeepSeek-V2-Lite, Qwen3-30B-A3B and Mixtral expert shapes; fallback accounting | `tests/python/ops/test_batchgen_expert_kernel_native.py` | Yes |
| Kernel latency A/B and end-to-end offloaded generation A/B (greedy-output identity, prefill latency, decode tok/s) | `benchmarks/ab_batchgen_expert_kernel.py micro` / `e2e` | Yes |

```bash
TRITON_INTERPRET=1 pytest tests/python/unit/test_batchgen_expert_kernel.py
pytest tests/python/ops/test_batchgen_expert_kernel_native.py
python benchmarks/ab_batchgen_expert_kernel.py micro --output-json micro.json
python benchmarks/ab_batchgen_expert_kernel.py e2e \
    --model allenai/OLMoE-1B-7B-0924-Instruct --offload-dir /tmp/moe-offload
```

BF16 is not checked on CPU, because Triton's interpreter computes raw BF16
`tl.dot` operands incorrectly. No speedup is claimed until the GPU benchmarks
above have been run.

## Licensing

`moe_infinity/kernel/batchgen/vendor/` holds BatchGen's files with their
Apache-2.0 `LICENSE`, a `NOTICE` that names the upstream commit, and
per-file headers that list the modifications. BatchGen describes
`fused_moe_bf16` as a Triton port of vLLM's `fused_moe` (Apache-2.0). Please
cite BatchGen (see `CITATIONS.md`) when using this path.
