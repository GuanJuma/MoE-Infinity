// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// Op registration for the vendored SGLang CPU MoE/GEMM kernels in ./sglang.
// Function declarations and schemas mirror SGLang's torch_extension_cpu.cpp
// (commit pinned in ./sglang/NOTICE); the namespace differs so this module can
// coexist with sgl_kernel in one process.

#include <ATen/ATen.h>
#include <Python.h>
#include <torch/library.h>

#include <optional>
#include <string>
#include <tuple>
#include <vector>

at::Tensor convert_weight_packed(at::Tensor& weight);

std::tuple<at::Tensor, at::Tensor> per_token_quant_int8_cpu(at::Tensor& A);

at::Tensor weight_packed_linear(at::Tensor& mat1, at::Tensor& mat2,
                                const std::optional<at::Tensor>& bias,
                                bool is_vnni);

at::Tensor int8_scaled_mm_with_quant(at::Tensor& mat1, at::Tensor& mat2,
                                     at::Tensor& scales2,
                                     const std::optional<at::Tensor>& bias,
                                     at::ScalarType out_dtype, bool is_vnni);

at::Tensor fp8_scaled_mm_cpu(at::Tensor& mat1, at::Tensor& mat2,
                             at::Tensor& scales2,
                             std::vector<int64_t> block_size,
                             const std::optional<at::Tensor>& bias,
                             at::ScalarType out_dtype, bool is_vnni);

at::Tensor fused_experts_cpu(
    at::Tensor& hidden_states, at::Tensor& w1, at::Tensor& w2,
    at::Tensor& topk_weights, at::Tensor& topk_ids, bool inplace,
    int64_t moe_comp_method, const std::optional<at::Tensor>& w1_scale,
    const std::optional<at::Tensor>& w2_scale,
    const std::optional<at::Tensor>& w1_zero,
    const std::optional<at::Tensor>& w2_zero,
    const std::optional<std::vector<int64_t>> block_size,
    const std::optional<at::Tensor>& w1_bias,
    const std::optional<at::Tensor>& w2_bias,
    const std::optional<double>& alpha, const std::optional<double>& limit,
    bool is_vnni, const std::optional<std::string>& activation);

at::Tensor shared_expert_cpu(
    at::Tensor& hidden_states, at::Tensor& w1, at::Tensor& w2,
    const std::optional<at::Tensor>& fused_experts_out,
    const std::optional<double> routed_scaling_factor, bool inplace,
    bool use_int8_w8a8, bool use_fp8_w8a16,
    const std::optional<at::Tensor>& w1_scale,
    const std::optional<at::Tensor>& w2_scale,
    const std::optional<std::vector<int64_t>> block_size, bool is_vnni);

TORCH_LIBRARY_FRAGMENT(moe_infinity_cpu, m) {
  m.def("convert_weight_packed(Tensor weight) -> Tensor");
  m.impl("convert_weight_packed", c10::DispatchKey::CPU,
         &convert_weight_packed);

  m.def("per_token_quant_int8_cpu(Tensor A) -> (Tensor, Tensor)");
  m.impl("per_token_quant_int8_cpu", c10::DispatchKey::CPU,
         &per_token_quant_int8_cpu);

  m.def(
      "weight_packed_linear(Tensor mat1, Tensor mat2, Tensor? bias, bool "
      "is_vnni) -> Tensor");
  m.impl("weight_packed_linear", c10::DispatchKey::CPU, &weight_packed_linear);

  m.def(
      "int8_scaled_mm_with_quant(Tensor mat1, Tensor mat2, Tensor scales2, "
      "Tensor? bias, ScalarType out_dtype, bool is_vnni) -> Tensor");
  m.impl("int8_scaled_mm_with_quant", c10::DispatchKey::CPU,
         &int8_scaled_mm_with_quant);

  m.def(
      "fp8_scaled_mm_cpu(Tensor mat1, Tensor mat2, Tensor scales2, int[] "
      "block_size, Tensor? bias, ScalarType out_dtype, bool is_vnni) -> "
      "Tensor");
  m.impl("fp8_scaled_mm_cpu", c10::DispatchKey::CPU, &fp8_scaled_mm_cpu);

  m.def(
      "fused_experts_cpu(Tensor hidden_states, Tensor w1, Tensor w2, Tensor "
      "topk_weights, Tensor topk_ids, bool inplace, int moe_comp_method, "
      "Tensor? w1_scale, Tensor? w2_scale, Tensor? w1_zero, Tensor? w2_zero, "
      "int[]? block_size, Tensor? w1_bias, Tensor? w2_bias, float? alpha, "
      "float? limit, bool is_vnni, str? activation=None) -> Tensor");
  m.impl("fused_experts_cpu", c10::DispatchKey::CPU, &fused_experts_cpu);

  m.def(
      "shared_expert_cpu(Tensor hidden_states, Tensor w1, Tensor w2, Tensor? "
      "fused_experts_out, float? routed_scaling_factor, bool inplace, bool "
      "use_int8_w8a8, bool use_fp8_w8a16, Tensor? w1_scale, Tensor? w2_scale, "
      "int[]? block_size, bool is_vnni) -> Tensor");
  m.impl("shared_expert_cpu", c10::DispatchKey::CPU, &shared_expert_cpu);
}

static PyModuleDef cpu_moe_module = {PyModuleDef_HEAD_INIT, "_cpu_moe", nullptr,
                                     -1, nullptr};

PyMODINIT_FUNC PyInit__cpu_moe(void) {
  return PyModule_Create(&cpu_moe_module);
}
