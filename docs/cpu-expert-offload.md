# CPU expert compute

MoE-Infinity normally fetches every routed expert to the GPU before running
it. With CPU expert compute, a chosen set of `(layer, expert)` pairs is kept
in host memory and computed on the CPU instead. Those experts are never
fetched, cached or executed on the GPU, while the remaining experts follow
the usual offload / prefetch / cache path. Both partial results are summed
per layer.

The CPU side runs SGLang's x86 fused-MoE kernels (`fused_experts_cpu`:
gate/up GEMM + activation + down GEMM + top-k weighted sum in one call),
vendored unmodified from [SGLang](https://github.com/sgl-project/sglang)
(Apache-2.0) under `extensions/kernel/cpu/sglang/`.

Status: opt-in, default off. The CPU path and the CPU/GPU split are covered by
CPU-only tests (a fake GPU dispatcher stands in for the native engine). The
combined path has **not** been run on a GPU yet.

## Enabling it

```bash
export MOE_CPU_EXPERTS="ratio:0.25"      # see spec syntax below
```

or per engine:

```python
model = MoE(checkpoint, {
    "offload_path": "...",
    "cpu_experts": "ratio:0.25",
    "cpu_expert_quant": "int8_w8a8",     # "bf16" (default) | "int8_w8a8" | "fp8_w8a16"
    "cpu_expert_threads": 0,             # 0 = torch.get_num_threads()
})
```

`ArcherConfig.cpu_experts` wins over `MOE_CPU_EXPERTS` when set.

| Spec | Meaning |
| --- | --- |
| `all` | every expert of every MoE layer |
| `0-7,12` | these expert ids in every MoE layer |
| `ratio:0.25` | the highest 25% of expert ids in every MoE layer |
| `3:0-7;5:1,2` | per layer: `layer:ids`, separated by `;` |

Requirements:

- x86_64 Linux with AVX512F + AVX512-BF16. AMX-BF16/INT8 (Sapphire Rapids
  and newer) is used automatically through oneDNN brgemm when present.
- Dense BF16/FP16/FP32 routed experts in the HF safetensors shards
  (`gate_proj`/`up_proj`/`down_proj` or Mixtral `w1`/`w3`/`w2`). The host
  copy is read from the checkpoint at engine start.
- A model whose MoE block uses the `expert_executor.dispatch_local` /
  `wait_dispatch_local` contract: Mixtral, OLMoE, Qwen3/Qwen3.5-MoE, DeepSeek
  V2/V3, DBRX, Jamba, NLLB-MoE, GLM. gpt-oss (packed MXFP4 experts) is not
  supported.

Run the AMX self-check once per host image:

```python
from moe_infinity.kernel.cpu import amx_selfcheck
amx_selfcheck()   # {'ok': True, 'deterministic': True, 'max_rel_err': 0.0029, ...}
```

On one KVM guest (4 vCPU Xeon with AMX), AMX tile state was not preserved
across vCPU preemption. BF16/INT8 GEMMs on the AMX path returned different,
wrong results from run to run (up to 70% relative error), in SGLang's kernels
and in PyTorch's own oneDNN matmul alike. The same work with AMX disabled was
bit-for-bit reproducible. If `ok` is False, start the process with
`ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16`, which keeps the AVX512-BF16/VNNI paths
and is exact. A passing check is not proof: on that guest it passed once while
later calls in the same process were still corrupted, so validate new VM types
with repeated runs.

## How it works

```
SyncXxxMoeBlock.forward
  └─ expert_executor.dispatch_local(layer, hidden[T,H], router_mask[T,E], router_weights[T,E])
       ├─ expert-drop (CPU-placed experts count as resident, so they are not dropped as misses)
       ├─ CpuExpertBackend.submit(...)
       │    ├─ gpu_mask = router_mask & ~cpu_placed   (and weights zeroed)  -> native dispatcher
       │    └─ worker thread: compact (token, expert) pairs routed to CPU experts,
       │       fused_experts_cpu(top_k=1) on the prepacked layer bank, index_add -> [T,H] fp32
       └─ native ExpertDispatcher: fetch / cache / run only the GPU experts
  └─ expert_executor.wait_dispatch_local()
       └─ GPU result + CPU result (copied to the GPU)
```

- `moe_infinity/runtime/cpu_experts.py`: `CpuExpertPlacement` (spec parsing,
  per-layer masks) and `CpuExpertBackend` (loading, packing, worker thread).
- `moe_infinity/kernel/cpu/`: `pack_experts` (BF16, INT8 W8A8 with
  per-channel weight scales and dynamic per-token activation quantization,
  FP8 W8A16 with 128x128 block scales), `fused_experts`, a PyTorch reference,
  `amx_selfcheck`.
- With a CUDA input, hidden states and routing are staged into pinned host
  buffers with non-blocking copies and an event, so `dispatch_local` does not
  block the GPU stream. The worker thread waits on the event.
- Prefetch correction and tracing see the GPU-only routing, so CPU experts
  are not corrected into the GPU prefetch set.

## Build

`setup.py` builds `moe_infinity._cpu_moe` on x86_64 Linux when the compiler
accepts the AMX/AVX512-BF16 flags (`MOE_BUILD_CPU_MOE`: unset = auto, `1` =
required, `0` = skip). CMake builds the `cpu_moe` target. It can also build
standalone without CUDA:

```bash
cmake -S extensions/kernel/cpu -B build/cpu_moe && cmake --build build/cpu_moe -j
```

Without a prebuilt module, `moe_infinity.kernel.cpu` JIT-compiles the same
sources on first use, which takes about 20 s on 4 cores (`MOE_CPU_MOE_JIT=0`
disables this). The ops are registered as `torch.ops.moe_infinity_cpu.*`, so
they can be loaded next to `sgl_kernel`.

## Performance

Measured on a 4-vCPU Intel Xeon KVM guest (Emerald-Rapids class, AMX +
AVX512-BF16, about 55 GB/s DRAM read bandwidth, no GPU). Weights were streamed
from DRAM: the LLC was flushed before every timed call. Numbers are with AMX
disabled, because of the AMX defect above. Median ms for one MoE layer's
expert FFN; `torch` is a per-expert PyTorch BF16 loop on oneDNN.

| Model shape | M | SGLang BF16 | SGLang INT8 | SGLang FP8 | torch BF16 |
| --- | --- | --- | --- | --- | --- |
| OLMoE-1B-7B (2048x1024, top-8) | 1 | 2.30 | 0.90 | 1.05 | 4.02 |
| | 64 | 29.5 | 10.2 | 21.6 | 42.4 |
| | 1024 | 236 | 50.8 | 237 | 258 |
| Qwen3-30B-A3B (2048x768, top-8) | 1 | 1.80 | 0.71 | 0.81 | 3.28 |
| | 64 | 38.2 | 13.4 | 18.7 | 58.4 |
| DeepSeek-V2-Lite (2048x1408, top-6) | 1 | 2.39 | 0.82 | 1.02 | 3.86 |
| Mixtral-8x7B (4096x14336, top-2) | 1 | 15.0 | 5.5 | 6.6 | 20.2 |
| | 64 | 138 | 23.9 | 131 | 107 |
| DeepSeek-V3 (7168x2048, top-8) | 1 | 14.6 | 5.6 | 6.3 | 19.8 |

Relative error vs FP32 on the unquantized weights: BF16 0.3%, INT8 W8A8 about 3%,
FP8 W8A16 about 4.7% (mostly quantization error).

Decode (small M) is bound by memory bandwidth. SGLang BF16 streams weights at
42–49 GB/s, while INT8 halves the bytes and reaches 53–67 GB/s, so INT8
W8A8 is 3–4.7x faster than the PyTorch loop at M=1 (BF16 1.35–1.8x). With AMX
enabled (timing only; numerics were corrupted on this guest), prefill
improves a lot: OLMoE M=1024 drops from 236 to 47 ms (BF16) and from 51 to
30 ms (INT8). Without AMX, SGLang BF16 is no faster than PyTorch for
Mixtral/DeepSeek-V3-sized experts at M ≥ 16. Reproduce with
`benchmarks/cpu_expert_kernels/bench_cpu_moe.py` (`--flush-mb 0` for warm
caches) and `summarize.py`.

## Limitations

- Host memory: the backend keeps its own packed copy of each CPU expert, next
  to the native store's copy.
- Speculative prefetch from `router_logits` can still prefetch CPU-placed
  experts of the next layer to the GPU. Correctness is unaffected, but PCIe
  bandwidth is wasted.
- The adaptive precision policy still counts CPU experts as observed.
- The fused native expert-drop (`MOE_EXPERT_DROP_FUSED=1`) is not CPU-aware.
  Do not combine it with CPU experts.
- The RPC `dispatch()` path is not supported. It is also unused today, since
  nothing sets a device map manager.
- Checkpoints with transformers-v5 batched experts, GPTQ, MXFP4 or FP8
  expert tensors cannot be loaded as CPU experts yet.
- The CPU worker shares cores with the Python thread that drives the GPU. Set
  `cpu_expert_threads` below the core count on busy hosts.
