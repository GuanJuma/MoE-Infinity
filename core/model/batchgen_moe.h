// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

// Native launcher for BatchGen's expert-FFN kernels
// (https://github.com/batchgen-project/batchgen, Apache-2.0).
//
// The Triton kernels in moe_infinity/kernel/batchgen/vendor are compiled to
// cubins at build time (moe_infinity/kernel/batchgen/aot.py) and launched
// here through the CUDA driver API, so the dispatcher's exec threads never
// touch Python.  Expert fetch, caching and prefetch are unchanged; only the
// on-GPU FFN of a fetched expert is swapped.

#pragma once

#include <cuda_runtime.h>
#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <vector>

namespace batchgen {

enum class ExpertKernel : int { kDefault = 0, kBatchGen = 1 };

// Initialised from MOE_EXPERT_KERNEL ("default" | "batchgen") on first use.
ExpertKernel GetExpertKernel();
// Accepts "default" or "batchgen"; throws std::invalid_argument otherwise.
void SetExpertKernel(const std::string& name);
std::string ExpertKernelName(ExpertKernel kernel);

// True when the build embedded BatchGen cubins.
bool CompiledIn();
std::vector<int> CompiledArchs();

// Computes output = (silu(input @ gate^T) * (input @ up^T)) @ down^T with
// BatchGen's grouped GEMM + silu_and_mul kernels, enqueued on `stream`.
//
//   input  [M, K]  gate, up [N, K]  down [H, N]  output [M, H]  (BF16)
//   gate_up_buf  >= rows * 2N elements, act_buf >= rows * N elements of
//                scratch; M is processed in row chunks that fit both.
//
// Returns false without enqueuing work when the device, dtype, layout or
// shapes are not covered by the embedded cubins; the caller then runs the
// default kernel.  The first fallback reason per process is logged.
bool ExpertFFN(const torch::Tensor& input, const torch::Tensor& gate,
               const torch::Tensor& up, const torch::Tensor& down,
               torch::Tensor& gate_up_buf, torch::Tensor& act_buf,
               torch::Tensor& output, cudaStream_t stream);

struct Stats {
  std::uint64_t expert_calls = 0;
  std::uint64_t packed_gate_up_calls = 0;
  std::uint64_t fallback_calls = 0;
  std::string last_fallback_reason;
  std::vector<int> loaded_archs;
};
Stats GetStats();
void ResetStats();

}  // namespace batchgen
