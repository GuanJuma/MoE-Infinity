# Hy3 on MoE-Infinity: tokenhub_bench traces for archer-rel-simulation

Run the tokenhub_bench datasets on Hy3-FP8 with MoE-Infinity, record every
MoE layer's top-k expert ids per forward pass, and feed them to the expert
cache simulators in `archer-rel-simulation.tar`.

```
tokenhub jsonl ──► Hy3 on MoE-Infinity (1 GPU, experts in host memory)
   (messages)        │  per request: traces/<dataset>/<row>.pt
                     ▼
            archer-rel-simulation (sequence_belady_cache_simulator.py,
            cache_strategy.py) + LRU/LFU ──► results.csv, summary.md
```

## Components

| Path | What it does |
| --- | --- |
| `moe_infinity/hy3/` | Hy3 engine: transformers `HYV3ForCausalLM` backbone (BF16, resident) with each sparse layer's MoE replaced by `Hy3OffloadMoE` (same router: sigmoid + `e_score_correction_bias` top-8, renormalized, x `router_scaling_factor`; same shared expert). Routed experts come from MoE-Infinity's store and run as FP8 GEMMs. |
| `python -m moe_infinity.hy3` | `inspect`, `plan` (memory budget), `build-store`, `generate` (smoke test). |
| `python -m moe_infinity.hy3.server` | Minimal OpenAI-compatible server (`/v1/chat/completions`, streaming, usage), one trace per request; tokenhub_bench talks to it over HTTP. |
| `benchmarks/hy3_trace/run_tokenhub_traces.py` | Offline driver: reads tokenhub jsonl rows directly, applies the chat template with tokenhub's Hy3 thinking-off setting (`reasoning_effort=no_think`), writes traces and `requests.jsonl`. |
| `benchmarks/hy3_trace/simulate_traces.py` | Runs the simulator's own `extract_access_sequence` + `belady_simulate` (and optionally `cache_strategy.HeapExpertCache`) plus LRU/LFU over a capacity sweep. |
| `benchmarks/hy3_trace/export_legacy_pickles.py` | Prefill/decode pickles for `cache_strategy.py` / Archer-era scripts. |
| `benchmarks/hy3_trace/sglang_dump_to_traces.py` | Fallback: split SGLang `per_token` expert-distribution dumps into the same trace files. |
| `benchmarks/hy3_trace/tokenhub/` | tokenhub_bench hardware + job YAML for the MoE-Infinity server. |

## Why a dedicated Hy3 path

MoE-Infinity's generic `MoE()`/`OffloadEngine` path cannot serve Hy3-FP8:

- `HYV3ForCausalLM` is not in moe-store's `MODEL_MAPPING_NAMES`;
- only block-128 FP8 (`weight_scale_inv`) is dequantized; Hy3 is per-tensor
  FP8 with `weight_scale` + static `input_scale` on every linear except
  `lm_head`/`embed_tokens`;
- the C++ expert dispatcher computes routed experts with `MoEMLP`, i.e.
  BatchGen AOT cubins (sm_80/sm_90 only) or CUTLASS `fused_moe_ffn_into`,
  which can fail on sm_120 with torch >= 2.13 (issue #245).

`moe_infinity/hy3` reuses MoE-Infinity's native store (`_store`: offload
files, pinned host pool, `begin`/`end`/`prefetch_tensors`, sparse-cache
eviction) exactly as `benchmarks/expert_placement/mi_expert_store.py` does,
and computes experts with `torch._scaled_mm` (cuBLASLt FP8, the
`torch_scaled_mm` kernel validated on the RTX PRO 5000 in the expert
placement benchmark) or BF16 after on-device dequantization. Neither path
touches `MoEMLP`, so #245 does not apply and `_store` needs no rebuild.

## Trace format

One `.pt` per request, compatible with SGLang's per-token recorder dumps and
with `sequence_belady_cache_simulator.py`:

```python
{"records": [{"forward_mode": "extend", "topk_ids_of_layer": int32[80, T, 8], ...},
             {"forward_mode": "decode", "topk_ids_of_layer": int32[80, 1, 8], ...}, ...],
 "meta": {"rid", "dataset", "row", "moe_layers": [1..79], "expert_bytes", "stats", ...}}
```

`records[0]` is the whole prefill (chunks merged); layer 0 (dense) rows are
`-1`; the MTP layer is not run. Pass `--ignore-layers 1` to the simulator.

## Memory plan (one RTX PRO 5000, `python -m moe_infinity.hy3 plan`)

| Item | Size |
| --- | --- |
| Resident backbone, BF16 (attention 6.04 B, shared experts 1.49 B, embed + lm_head 0.99 B, dense layer 0 0.16 B, routers 0.06 B params) | 16.3 GiB |
| KV cache (BF16, 320 KiB/token) at `--max-context 20480` | 6.25 GiB |
| GPU expert cache `--expert-cache-gib 32` | 32 GiB = 1,820 of 15,168 experts (12%) |
| Workspace + CUDA context (estimate) | ~5.5 GiB |
| Routed experts, host (79 layers x 192 x 18 MiB FP8) | 266.6 GiB |
| `--store mi` pinned host pool (torch rounds each 18 MiB node to 32 MiB) | 474 GiB |
| `--store mi` offload files on disk | 266.6 GiB |

## Verified

CPU-only, on a tiny random checkpoint with Hy3-FP8's exact layout
(`moe_infinity/hy3/tiny_checkpoint.py`), `tests/python/unit/test_hy3_engine.py`:
logits and every layer's routing equal transformers `HYV3ForCausalLM` with the
same dequantized weights; greedy continuation identical with chunked prefill;
FP8 W8A8 within 7% logits error; the MoE-Infinity store path (CPU fake of
`_store`) and the pinned LRU store under eviction stay exact; the simulator
and the tokenhub_bench `cache_hit` suite ran against the outputs/server.
Not verified: real Hy3-FP8 weights, GPU kernels, the native `_store` at
15,168 nodes, throughput.
