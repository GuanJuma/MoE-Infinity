# Single-expert placement benchmark

`benchmarks/expert_placement/` pins one routed expert `(layer, expert)` of a
HuggingFace safetensors MoE checkpoint and measures, for each token count M
(the rows routed to that expert), three placements:

| Scenario | Weights live in | Computed on | Overhead | Compute |
| --- | --- | --- | --- | --- |
| `gpu_resident` (S1) | GPU memory | GPU | `total - compute`: Python dispatch, kernel launches, sync | CUDA-graph replay of the expert, timed with CUDA events (pure device time) |
| `cpu_compute` (S2) | host memory | CPU | activations GPU→pinned host (`d2h`) + result host→GPU (`h2d`), each with a stream sync | CPU kernel wall time (result written in place) |
| `cpu_store_gpu_compute` (S3) | host memory (pinned and pageable runs) | GPU | H2D copy of the expert's GPU-ready weights (`xfer`, CUDA events) + launch/dispatch | GPU kernel right after the copy (graph replay, events) |

All three report `total_ms` as host wall time for one end-to-end call, and
`overhead_ms = total_ms - compute_ms`. S3 also reports
`pipelined_ms = max(xfer, compute)`, the cost if the copy were fully
prefetched. One-time costs are kept under `setup` in `results.json` and never
enter per-token numbers: checkpoint read, weight layout preparation, CPU
packing, GPU upload and pinned allocation. Values are medians, with p10/p90
alongside. Before every timed repeat the GPU L2 and CPU LLC are flushed, by
default with 2× their size, outside the timed region. This way weights stream
from HBM/DRAM, as they do when other layers run in between.

## Kernels

| | Kernel | Notes |
| --- | --- | --- |
| GPU | `sglang_triton` | SGLang's Triton `fused_experts` with E=1 and top-1. SGLang uses this path to serve per-tensor FP8 MoE on sm_120, where DeepGEMM and CUTLASS block-FP8 MoE are unavailable. The weights are prepared like SGLang's `Fp8MoEMethod`: gate and up are requantized to one scale (their max), and activations use the checkpoint's static `input_scale`. It runs in place on a work buffer, which avoids the TP-group allocation path, and it publishes `ServerArgs(model_path="dummy")` once. |
| GPU | `torch_scaled_mm` | Two cuBLASLt FP8 GEMMs (`torch._scaled_mm`, per-tensor scales). Per-tensor FP8 only. |
| GPU | `torch_bf16` | Weights dequantized once to BF16, then cuBLAS. Works for any format and GPU, but S3 copies twice the bytes. |
| CPU | `sglang_fp8_w8a16` | Vendored SGLang `fused_experts_cpu`. The e4m3 weights stay FP8: per-tensor scales are broadcast exactly to 128×128 block scales through `pack_fp8_experts`. |
| CPU | `sglang_bf16`, `sglang_int8_w8a8` | Same kernel on BF16 weights, or on requantized INT8 weights. |
| CPU | `torch_bf16` | PyTorch/oneDNN baseline. |

The PR #1 BatchGen cubins (BF16, sm_80/sm_90 by default) and the MoE-Infinity
CUDA extensions are not used. Only `moe_infinity.kernel.cpu` is needed, and it
JIT-builds from `extensions/kernel/cpu` without CUDA.

## Checkpoint detection

`--list-experts` reads only the safetensors headers and prints:

- the MoE layers;
- the expert layout: per-expert `experts.E.{gate,up,down}_proj` or `w1/w3/w2`,
  fused `gate_up_proj`, or stacked `experts.gate_up_proj`;
- for the selected expert, every tensor with its shape and dtype, the scale
  scheme (per-tensor, per-channel or block), and the static input scales.

It then reads just the selected expert's bytes. Hy3-FP8, for example, has
192 experts in layers 1–80 (layer 80 is the MTP layer). Each projection is
`F8_E4M3` with a scalar BF16 `weight_scale` and an F32 `[1]` `input_scale`,
under `quantization_config = {quant_method: fp8, activation_scheme: static}`.
An expert is K=4096, N=1536 and 18.0 MiB.

## Usage

```bash
python benchmarks/expert_placement/expert_placement_bench.py --model-dir /model --list-experts
python benchmarks/expert_placement/expert_placement_bench.py --model-dir /model --layer 1 --expert 0 --dry-run
MODEL_DIR=/model LAYER=1 EXPERT=0 bash benchmarks/expert_placement/run_sweep.sh   # NUMA-bound sweep
python benchmarks/expert_placement/summarize_placement.py /scratch/expert_placement/<run>
```

`run_sweep.sh` does the following:

1. Checks that exactly one idle GPU is visible.
2. Finds the GPU's NUMA node from sysfs.
3. Binds to that node's physical cores and memory with `numactl`. Inside
   Docker this needs `--cap-add SYS_NICE`.
4. Runs list, dry-run, sweep and summary in that order.

The outputs are `results.csv`, `results.json`, `env/*.txt` (lscpu, numactl,
nvidia-smi, topology), `summary.md` and plots. The CPU-only debug flag
`--device cpu` runs the GPU-scenario code on CPU tensors. Its timings are
meaningless, and it exists only for tests.

`make_fake_checkpoint.py` writes small synthetic checkpoints for smoke tests:
the Hy3-FP8 layout, block FP8, BF16 and stacked.

## Verification status

The following has been verified on CPU (x86 with AVX512-BF16; AMX disabled):

- checkpoint parsing against the `safetensors` library;
- detection of the real Hy3-FP8 index;
- the CPU sweep on real Hy3-FP8 layer 1 expert 7 bytes;
- CPU kernel numerics against an FP32 reference: about 3e-3 for FP8 W8A16;
- the `torch_scaled_mm` quantization logic, through an emulation;
- the SGLang adapter's argument wiring, against a stand-in module with SGLang
  v0.5.19's exact signatures.

Nothing has run on a GPU yet, including the Triton kernel, cuBLASLt FP8, CUDA
graphs, PCIe copies and the timings themselves. Tests:
`tests/python/unit/test_expert_placement_bench.py`.
