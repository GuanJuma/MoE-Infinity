// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#include "model/batchgen_moe.h"

#include <cuda.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

#include "utils/logger.h"

#if defined(MOE_WITH_BATCHGEN)
  #include "batchgen_moe_cubins.h"
#endif

namespace batchgen {
namespace {

std::atomic<int> g_kernel{-1};

ExpertKernel ParseKernel(std::string name) {
  std::transform(name.begin(), name.end(), name.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (name.empty() || name == "default") return ExpertKernel::kDefault;
  if (name == "batchgen") return ExpertKernel::kBatchGen;
  throw std::invalid_argument("unknown expert kernel '" + name +
                              "', expected 'default' or 'batchgen'");
}

struct Counters {
  std::atomic<std::uint64_t> expert_calls{0};
  std::atomic<std::uint64_t> packed_gate_up_calls{0};
  std::atomic<std::uint64_t> fallback_calls{0};
  std::mutex mutex;
  std::string last_fallback_reason;
  std::set<std::string> logged_reasons;
  std::set<int> loaded_archs;
};

Counters& GetCounters() {
  static Counters counters;
  return counters;
}

bool Fallback(const std::string& reason) {
  auto& counters = GetCounters();
  counters.fallback_calls.fetch_add(1, std::memory_order_relaxed);
  bool first = false;
  {
    std::lock_guard<std::mutex> lock(counters.mutex);
    counters.last_fallback_reason = reason;
    first = counters.logged_reasons.insert(reason).second;
  }
  if (first) {
    DLOG_WARN("BatchGen expert kernel not used, running default kernel:",
              reason);
  }
  return false;
}

#if defined(MOE_WITH_BATCHGEN)

using moe_batchgen_aot::CubinEntry;
using moe_batchgen_aot::kCubins;

static_assert(moe_batchgen_aot::kNumGemmParams == 14,
              "GEMM parameter packing below is out of sync with aot.py");
static_assert(moe_batchgen_aot::kNumSiluParams == 5,
              "silu parameter packing below is out of sync with aot.py");
static_assert(moe_batchgen_aot::kTritonScratchArgs <= 2,
              "unexpected number of Triton scratch arguments");

// Index = 2 * large + odd_k, matching aot.gemm_variants().
constexpr const char* kGemmVariants[] = {"gemm_small_evenk", "gemm_small_oddk",
                                         "gemm_large_evenk", "gemm_large_oddk"};
// BatchGen's _pick_config: the small M-tile up to 1024 token slots.
constexpr int64_t kSmallConfigMaxRows = 1024;
constexpr int64_t kMaxChunkRows = 8192;
constexpr int kAlign = 16;

struct Kernel {
  CUfunction fn = nullptr;
  unsigned shared_bytes = 0;
  int num_warps = 0;
  int block_m = 0;
  int block_n = 0;
  int block_k = 0;
};

struct DeviceKernels {
  std::string error;
  int arch = 0;
  Kernel gemm[4];
  Kernel silu;
  // Identity routing for a single expert: token slot i -> row i, every
  // M-tile maps to expert 0, and the padded-token bound never trims the grid.
  CUdeviceptr sorted_ids = 0;
  CUdeviceptr expert_ids = 0;
  CUdeviceptr num_post = 0;
};

std::string DriverError(const char* what, CUresult result) {
  const char* name = nullptr;
  cuGetErrorName(result, &name);
  return std::string(what) + " failed: " + (name ? name : "unknown");
}

const CubinEntry* FindCubin(const char* variant, int major, int minor) {
  const CubinEntry* best = nullptr;
  for (const auto& entry : kCubins) {
    if (std::strcmp(entry.variant, variant) != 0) continue;
    if (entry.arch / 10 != major || entry.arch % 10 > minor) continue;
    if (best == nullptr || entry.arch > best->arch) best = &entry;
  }
  return best;
}

std::string LoadKernel(const CubinEntry& entry, CUdevice device,
                       Kernel* kernel) {
  CUmodule module = nullptr;
  CUresult result = cuModuleLoadData(&module, entry.data);
  if (result != CUDA_SUCCESS) return DriverError("cuModuleLoadData", result);
  result = cuModuleGetFunction(&kernel->fn, module, entry.function);
  if (result != CUDA_SUCCESS) return DriverError("cuModuleGetFunction", result);
  if (entry.shared_bytes > 49152) {
    int shared_optin_bytes = 0;
    int static_bytes = 0;
    cuDeviceGetAttribute(&shared_optin_bytes,
                         CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                         device);
    cuFuncGetAttribute(&static_bytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                       kernel->fn);
    if (shared_optin_bytes < static_cast<int>(entry.shared_bytes)) {
      return "kernel needs more shared memory than the device offers";
    }
    cuFuncSetCacheConfig(kernel->fn, CU_FUNC_CACHE_PREFER_SHARED);
    result = cuFuncSetAttribute(kernel->fn,
                                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                shared_optin_bytes - static_bytes);
    if (result != CUDA_SUCCESS)
      return DriverError("cuFuncSetAttribute", result);
  }
  kernel->shared_bytes = entry.shared_bytes;
  kernel->num_warps = entry.num_warps;
  kernel->block_m = entry.block_m;
  kernel->block_n = entry.block_n;
  kernel->block_k = entry.block_k;
  return "";
}

std::string InitDevice(int device_index, DeviceKernels* state) {
  cudaError_t rt = cudaSetDevice(device_index);
  if (rt != cudaSuccess) return cudaGetErrorString(rt);
  // Make sure the primary context exists and is current for the driver API.
  rt = cudaFree(nullptr);
  if (rt != cudaSuccess) return cudaGetErrorString(rt);

  CUdevice device;
  CUresult result = cuDeviceGet(&device, device_index);
  if (result != CUDA_SUCCESS) return DriverError("cuDeviceGet", result);
  int major = 0;
  int minor = 0;
  cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                       device);
  cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                       device);

  std::vector<std::pair<const char*, Kernel*>> wanted;
  for (int i = 0; i < 4; ++i)
    wanted.emplace_back(kGemmVariants[i], &state->gemm[i]);
  wanted.emplace_back("silu_and_mul", &state->silu);
  for (auto& [variant, kernel] : wanted) {
    const CubinEntry* entry = FindCubin(variant, major, minor);
    if (entry == nullptr) {
      return "no BatchGen cubin built for sm_" + std::to_string(major) +
             std::to_string(minor);
    }
    std::string error = LoadKernel(*entry, device, kernel);
    if (!error.empty()) return std::string(variant) + ": " + error;
    state->arch = entry->arch;
  }

  const int64_t max_padded = kMaxChunkRows + 64;
  std::vector<int32_t> ids(max_padded);
  for (int64_t i = 0; i < max_padded; ++i) ids[i] = static_cast<int32_t>(i);
  std::vector<int32_t> zeros(max_padded, 0);
  const int32_t num_post = INT32_MAX;
  void* ptr = nullptr;
  for (auto [dst, src, bytes] :
       {std::tuple{&state->sorted_ids, static_cast<const void*>(ids.data()),
                   ids.size() * sizeof(int32_t)},
        std::tuple{&state->expert_ids, static_cast<const void*>(zeros.data()),
                   zeros.size() * sizeof(int32_t)},
        std::tuple{&state->num_post, static_cast<const void*>(&num_post),
                   sizeof(int32_t)}}) {
    rt = cudaMalloc(&ptr, bytes);
    if (rt != cudaSuccess) return cudaGetErrorString(rt);
    rt = cudaMemcpy(ptr, src, bytes, cudaMemcpyHostToDevice);
    if (rt != cudaSuccess) return cudaGetErrorString(rt);
    *dst = reinterpret_cast<CUdeviceptr>(ptr);
  }
  return "";
}

const DeviceKernels& GetDeviceKernels(int device_index) {
  static std::mutex mutex;
  static std::unordered_map<int, std::unique_ptr<DeviceKernels>> states;
  std::lock_guard<std::mutex> lock(mutex);
  auto& slot = states[device_index];
  if (!slot) {
    int previous = 0;
    cudaGetDevice(&previous);
    slot = std::make_unique<DeviceKernels>();
    slot->error = InitDevice(device_index, slot.get());
    cudaSetDevice(previous);
    if (slot->error.empty()) {
      auto& counters = GetCounters();
      std::lock_guard<std::mutex> stats_lock(counters.mutex);
      counters.loaded_archs.insert(slot->arch);
      DLOG_INFO("BatchGen expert kernels loaded on device", device_index,
                "from sm_", slot->arch, "cubins");
    }
  }
  return *slot;
}

int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

CUdeviceptr Ptr(const torch::Tensor& t, int64_t element_offset = 0) {
  return reinterpret_cast<CUdeviceptr>(t.data_ptr()) +
         element_offset * t.element_size();
}

// C[row] = A[row] @ B^T for rows [0, valid) of one expert.
void LaunchGemm(const DeviceKernels& state, const Kernel& kernel,
                CUstream stream, CUdeviceptr a, CUdeviceptr b, CUdeviceptr c,
                int n, int k, int rows, int stride_am, int stride_bn,
                int stride_cm) {
  CUdeviceptr sorted_ids = state.sorted_ids;
  CUdeviceptr expert_ids = state.expert_ids;
  CUdeviceptr num_post = state.num_post;
  int em = static_cast<int>(CeilDiv(rows, kernel.block_m) * kernel.block_m);
  int64_t stride_be = 0;
  CUdeviceptr global_scratch = 0;
  CUdeviceptr profile_scratch = 0;
  void* params[] = {&a,
                    &b,
                    &c,
                    &sorted_ids,
                    &expert_ids,
                    &num_post,
                    &n,
                    &k,
                    &em,
                    &rows,
                    &stride_am,
                    &stride_be,
                    &stride_bn,
                    &stride_cm,
                    &global_scratch,
                    &profile_scratch};
  unsigned grid = static_cast<unsigned>(CeilDiv(em, kernel.block_m) *
                                        CeilDiv(n, kernel.block_n));
  CUresult result =
      cuLaunchKernel(kernel.fn, grid, 1, 1, kernel.num_warps * 32, 1, 1,
                     kernel.shared_bytes, stream, params, nullptr);
  TORCH_CHECK(result == CUDA_SUCCESS,
              DriverError("BatchGen _fused_moe_gemm launch", result));
}

void LaunchSilu(const Kernel& kernel, CUstream stream, CUdeviceptr x,
                CUdeviceptr y, int n, int rows, int stride_xm, int stride_ym) {
  CUdeviceptr global_scratch = 0;
  CUdeviceptr profile_scratch = 0;
  void* params[] = {
      &x, &y, &n, &stride_xm, &stride_ym, &global_scratch, &profile_scratch};
  unsigned grid_y = static_cast<unsigned>(CeilDiv(n, kernel.block_n));
  CUresult result = cuLaunchKernel(
      kernel.fn, static_cast<unsigned>(rows), grid_y, 1, kernel.num_warps * 32,
      1, 1, kernel.shared_bytes, stream, params, nullptr);
  TORCH_CHECK(result == CUDA_SUCCESS,
              DriverError("BatchGen _silu_and_mul_kernel launch", result));
}

const Kernel& PickGemm(const DeviceKernels& state, int64_t rows, int64_t k) {
  const int large = rows > kSmallConfigMaxRows ? 1 : 0;
  const Kernel& even = state.gemm[2 * large];
  return (k % even.block_k == 0) ? even : state.gemm[2 * large + 1];
}

bool Aligned(const torch::Tensor& t) {
  return reinterpret_cast<std::uintptr_t>(t.data_ptr()) % kAlign == 0;
}

#endif  // MOE_WITH_BATCHGEN

}  // namespace

ExpertKernel GetExpertKernel() {
  int value = g_kernel.load(std::memory_order_acquire);
  if (value >= 0) return static_cast<ExpertKernel>(value);
  const char* env = std::getenv("MOE_EXPERT_KERNEL");
  ExpertKernel parsed = ExpertKernel::kDefault;
  try {
    parsed = ParseKernel(env ? env : "");
  } catch (const std::invalid_argument& e) {
    DLOG_WARN("MOE_EXPERT_KERNEL:", e.what(), "- using default");
  }
  int expected = -1;
  g_kernel.compare_exchange_strong(expected, static_cast<int>(parsed),
                                   std::memory_order_acq_rel);
  return static_cast<ExpertKernel>(g_kernel.load(std::memory_order_acquire));
}

void SetExpertKernel(const std::string& name) {
  g_kernel.store(static_cast<int>(ParseKernel(name)),
                 std::memory_order_release);
}

std::string ExpertKernelName(ExpertKernel kernel) {
  return kernel == ExpertKernel::kBatchGen ? "batchgen" : "default";
}

bool CompiledIn() {
#if defined(MOE_WITH_BATCHGEN)
  return true;
#else
  return false;
#endif
}

std::vector<int> CompiledArchs() {
#if defined(MOE_WITH_BATCHGEN)
  return {std::begin(moe_batchgen_aot::kArchs),
          std::end(moe_batchgen_aot::kArchs)};
#else
  return {};
#endif
}

bool ExpertFFN(const torch::Tensor& input, const torch::Tensor& gate,
               const torch::Tensor& up, const torch::Tensor& down,
               torch::Tensor& gate_up_buf, torch::Tensor& act_buf,
               torch::Tensor& output, cudaStream_t stream) {
#if !defined(MOE_WITH_BATCHGEN)
  return Fallback("built without BatchGen cubins (MOE_BUILD_BATCHGEN=0)");
#else
  for (const torch::Tensor* t : std::initializer_list<const torch::Tensor*>{
           &input, &gate, &up, &down, &gate_up_buf, &act_buf, &output}) {
    if (!t->is_cuda() || t->scalar_type() != torch::kBFloat16) {
      return Fallback("tensors must be BF16 on CUDA");
    }
    if (!t->is_contiguous()) return Fallback("tensors must be contiguous");
    if (!Aligned(*t)) return Fallback("tensors must be 16-byte aligned");
  }
  if (input.dim() != 2 || gate.dim() != 2 || up.dim() != 2 || down.dim() != 2 ||
      output.dim() != 2) {
    return Fallback("expected 2-D activations and weights");
  }
  const int64_t rows = input.size(0);
  const int64_t k = input.size(1);
  const int64_t n = gate.size(0);
  const int64_t h = down.size(0);
  if (gate.size(1) != k || up.size(0) != n || up.size(1) != k ||
      down.size(1) != n || output.size(0) != rows || output.size(1) != h) {
    return Fallback("expert weight shapes do not match the input");
  }
  if (rows <= 0) return Fallback("empty token batch");
  if (k % 16 != 0 || n % 16 != 0 || h % 16 != 0) {
    return Fallback("hidden and intermediate sizes must be multiples of 16");
  }
  if (2 * n > INT32_MAX || h > INT32_MAX || k > INT32_MAX) {
    return Fallback("dimension exceeds int32");
  }
  const int64_t chunk_rows = std::min({rows, gate_up_buf.numel() / (2 * n),
                                       act_buf.numel() / n, kMaxChunkRows});
  if (chunk_rows <= 0) return Fallback("scratch buffers are too small");

  const DeviceKernels& state = GetDeviceKernels(input.get_device());
  if (!state.error.empty()) return Fallback(state.error);

  const bool packed =
      up.data_ptr() == static_cast<const char*>(gate.data_ptr()) +
                           gate.numel() * gate.element_size();
  auto cu_stream = reinterpret_cast<CUstream>(stream);
  const CUdeviceptr c1 = Ptr(gate_up_buf);
  const CUdeviceptr act = Ptr(act_buf);
  for (int64_t r0 = 0; r0 < rows; r0 += chunk_rows) {
    const int chunk = static_cast<int>(std::min(chunk_rows, rows - r0));
    const Kernel& stage1 = PickGemm(state, chunk, k);
    const CUdeviceptr x = Ptr(input, r0 * k);
    if (packed) {
      LaunchGemm(state, stage1, cu_stream, x, Ptr(gate), c1,
                 static_cast<int>(2 * n), static_cast<int>(k), chunk,
                 static_cast<int>(k), static_cast<int>(k),
                 static_cast<int>(2 * n));
    } else {
      LaunchGemm(state, stage1, cu_stream, x, Ptr(gate), c1,
                 static_cast<int>(n), static_cast<int>(k), chunk,
                 static_cast<int>(k), static_cast<int>(k),
                 static_cast<int>(2 * n));
      LaunchGemm(state, stage1, cu_stream, x, Ptr(up),
                 c1 + n * gate_up_buf.element_size(), static_cast<int>(n),
                 static_cast<int>(k), chunk, static_cast<int>(k),
                 static_cast<int>(k), static_cast<int>(2 * n));
    }
    LaunchSilu(state.silu, cu_stream, c1, act, static_cast<int>(n), chunk,
               static_cast<int>(2 * n), static_cast<int>(n));
    const Kernel& stage2 = PickGemm(state, chunk, n);
    LaunchGemm(state, stage2, cu_stream, act, Ptr(down), Ptr(output, r0 * h),
               static_cast<int>(h), static_cast<int>(n), chunk,
               static_cast<int>(n), static_cast<int>(n), static_cast<int>(h));
  }

  auto& counters = GetCounters();
  counters.expert_calls.fetch_add(1, std::memory_order_relaxed);
  if (packed) {
    counters.packed_gate_up_calls.fetch_add(1, std::memory_order_relaxed);
  }
  return true;
#endif
}

Stats GetStats() {
  auto& counters = GetCounters();
  Stats stats;
  stats.expert_calls = counters.expert_calls.load();
  stats.packed_gate_up_calls = counters.packed_gate_up_calls.load();
  stats.fallback_calls = counters.fallback_calls.load();
  std::lock_guard<std::mutex> lock(counters.mutex);
  stats.last_fallback_reason = counters.last_fallback_reason;
  stats.loaded_archs.assign(counters.loaded_archs.begin(),
                            counters.loaded_archs.end());
  return stats;
}

void ResetStats() {
  auto& counters = GetCounters();
  counters.expert_calls = 0;
  counters.packed_gate_up_calls = 0;
  counters.fallback_calls = 0;
  std::lock_guard<std::mutex> lock(counters.mutex);
  counters.last_fallback_reason.clear();
}

}  // namespace batchgen
