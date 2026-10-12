// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// Benchmark-only bindings, JIT-built by bench_cpu_attention.py /
// bench_cpu_moe.py from external source trees; nothing here is shipped.
//
//   moegen_gqa      MoE-Gen grouped_query_attention_cpu_avx2 (OpenMP variant,
//                   compiled from the MoE-Gen source tree)
//   sglang_decode   SGLang decode_attention_cpu (compiled from SGLang)
//   moegen_style_expert_ffn
//                   expert FFN written with MoE-Gen's AVX2 helpers. MoE-Gen
//                   ships no CPU expert kernel; this is the obvious extension
//                   of its attention code to expert GEMVs, used as a baseline.

#include <immintrin.h>
#include <omp.h>
#include <torch/extension.h>

#include <cmath>
#include <cstring>
#include <optional>
#include <vector>

torch::Tensor grouped_query_attention_cpu_avx2(
    const torch::Tensor& query_states,
    const std::vector<c10::BFloat16*>& key_states,
    const std::vector<c10::BFloat16*>& value_states,
    const torch::Tensor& attention_mask, int64_t cache_length,
    int64_t group_size, int64_t num_attention_heads, int64_t num_kv_head,
    int64_t head_dim, int64_t num_threads);

void decode_attention_cpu(
    at::Tensor& query, at::Tensor& k_buffer, at::Tensor& v_buffer,
    double k_buf_scale, double v_buf_scale, at::Tensor& output,
    const std::optional<at::Tensor>& key,
    const std::optional<at::Tensor>& value, at::Tensor& loc,
    at::Tensor& attn_logits, at::Tensor& req_to_token,
    at::Tensor& req_pool_indices, at::Tensor& seq_lens, double sm_scale,
    double logit_cap, bool is_cross_attn, int64_t sliding_window_size,
    std::optional<at::Tensor> encoder_lens, std::optional<at::Tensor> sinks);

static torch::Tensor moegen_gqa(const torch::Tensor& query,
                                const std::vector<torch::Tensor>& keys,
                                const std::vector<torch::Tensor>& values,
                                const torch::Tensor& mask, int64_t cache_length,
                                int64_t group_size, int64_t num_heads,
                                int64_t num_kv_heads, int64_t head_dim,
                                int64_t num_threads) {
  std::vector<c10::BFloat16*> k_ptrs, v_ptrs;
  for (size_t i = 0; i < keys.size(); ++i) {
    k_ptrs.push_back(keys[i].data_ptr<c10::BFloat16>());
    v_ptrs.push_back(values[i].data_ptr<c10::BFloat16>());
  }
  return grouped_query_attention_cpu_avx2(query, k_ptrs, v_ptrs, mask,
                                          cache_length, group_size, num_heads,
                                          num_kv_heads, head_dim, num_threads);
}

static void sglang_decode(at::Tensor query, at::Tensor k_buffer,
                          at::Tensor v_buffer, at::Tensor output,
                          at::Tensor loc, at::Tensor attn_logits,
                          at::Tensor req_to_token, at::Tensor req_pool_indices,
                          at::Tensor seq_lens, double sm_scale) {
  decode_attention_cpu(query, k_buffer, v_buffer, 1.0, 1.0, output,
                       std::nullopt, std::nullopt, loc, attn_logits,
                       req_to_token, req_pool_indices, seq_lens, sm_scale, 0.0,
                       false, -1, std::nullopt, std::nullopt);
}

// The two helpers below are MoE-Gen's
// (grouped_query_attention_cpu_avx2_omp.cpp, Apache-2.0, EfficientMoE team),
// kept as-is apart from the names.
static inline __m256 mg_bf16_to_fp32(__m128i bf16_vals) {
  __m256i tmp = _mm256_cvtepu16_epi32(bf16_vals);
  tmp = _mm256_slli_epi32(tmp, 16);
  return _mm256_castsi256_ps(tmp);
}

static inline c10::BFloat16 mg_dot_product(const c10::BFloat16* a,
                                           const c10::BFloat16* b,
                                           int64_t num_elements) {
  auto tmp = 0.0f;
  for (int64_t i = 0; i < num_elements; i += 8) {
    __m128i a_bf16 = _mm_loadu_si128((__m128i*)(&a[i]));
    __m128i b_bf16 = _mm_loadu_si128((__m128i*)(&b[i]));
    __m256 a_fp32 = mg_bf16_to_fp32(a_bf16);
    __m256 b_fp32 = mg_bf16_to_fp32(b_bf16);
    __m256 result = _mm256_dp_ps(a_fp32, b_fp32, 0xf1);
    float unpacked[8];
    std::memcpy(unpacked, &result, sizeof(float) * 8);
    tmp += unpacked[0] + unpacked[4];
  }
  return c10::BFloat16(tmp);
}

// x: [M, K] bf16, w13: [E, 2N, K] bf16, w2: [E, K, N] bf16 (plain row-major),
// topk_ids: [M, topk] int32, topk_weights: [M, topk] float -> [M, K] float.
static torch::Tensor moegen_style_expert_ffn(
    const torch::Tensor& x, const torch::Tensor& w13, const torch::Tensor& w2,
    const torch::Tensor& topk_ids, const torch::Tensor& topk_weights) {
  const int64_t M = x.size(0), K = x.size(1);
  const int64_t N = w13.size(1) / 2;
  const int64_t topk = topk_ids.size(1);
  TORCH_CHECK(K % 8 == 0 && N % 8 == 0, "K and N must be multiples of 8");
  auto out = torch::zeros({M, K}, x.options().dtype(torch::kFloat));
  auto h = torch::empty({2 * N}, x.options());
  auto act = torch::empty({N}, x.options());
  const auto* xp = x.data_ptr<c10::BFloat16>();
  const auto* w13p = w13.data_ptr<c10::BFloat16>();
  const auto* w2p = w2.data_ptr<c10::BFloat16>();
  const auto* ids = topk_ids.data_ptr<int32_t>();
  const auto* tw = topk_weights.data_ptr<float>();
  auto* hp = h.data_ptr<c10::BFloat16>();
  auto* ap = act.data_ptr<c10::BFloat16>();
  auto* op = out.data_ptr<float>();
  for (int64_t t = 0; t < M; ++t) {
    for (int64_t s = 0; s < topk; ++s) {
      const int64_t e = ids[t * topk + s];
      const float wt = tw[t * topk + s];
      const c10::BFloat16* xe = xp + t * K;
      const c10::BFloat16* w13e = w13p + e * 2 * N * K;
      const c10::BFloat16* w2e = w2p + e * K * N;
#pragma omp parallel for schedule(static)
      for (int64_t j = 0; j < 2 * N; ++j) {
        hp[j] = mg_dot_product(xe, w13e + j * K, K);
      }
#pragma omp parallel for schedule(static)
      for (int64_t j = 0; j < N; ++j) {
        float g = static_cast<float>(hp[j]);
        float u = static_cast<float>(hp[N + j]);
        ap[j] = c10::BFloat16(g / (1.0f + std::exp(-g)) * u);
      }
#pragma omp parallel for schedule(static)
      for (int64_t i = 0; i < K; ++i) {
        op[t * K + i] +=
            wt * static_cast<float>(mg_dot_product(ap, w2e + i * N, N));
      }
    }
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moegen_gqa", &moegen_gqa);
  m.def("sglang_decode", &sglang_decode);
  m.def("moegen_style_expert_ffn", &moegen_style_expert_ffn);
}
