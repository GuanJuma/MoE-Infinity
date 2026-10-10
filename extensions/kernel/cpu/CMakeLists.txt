# SGLang-derived CPU expert kernels -> moe_infinity._cpu_moe (x86_64 only).
# Builds as part of the top-level project or standalone (no CUDA needed):
#   cmake -S extensions/kernel/cpu -B build/cpu_moe && cmake --build build/cpu_moe
cmake_minimum_required(VERSION 3.18)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(moe_infinity_cpu_moe LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" "-c"
            "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE _TORCH_CMAKE_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
  list(APPEND CMAKE_PREFIX_PATH "${_TORCH_CMAKE_PREFIX}")
  find_package(Torch REQUIRED)
  find_package(OpenMP REQUIRED)
endif()

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  message(STATUS "cpu_moe: skipped (needs x86_64)")
  return()
endif()

# Keep in sync with CPU_MOE_SOURCES / CPU_MOE_CXX_FLAGS in
# moe_infinity/kernel/cpu/_ext.py.
set(CPU_MOE_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/cpu_moe_bindings.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/sglang/gemm.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/sglang/gemm_int8.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/sglang/gemm_fp8.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/sglang/gemm_int4.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/sglang/moe.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/sglang/moe_int8.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/sglang/moe_fp8.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/sglang/moe_int4.cpp)

Python3_add_library(cpu_moe MODULE WITH_SOABI ${CPU_MOE_SOURCES})
set_target_properties(cpu_moe PROPERTIES OUTPUT_NAME _cpu_moe)
target_include_directories(cpu_moe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/sglang
                                           ${TORCH_INCLUDE_DIRS})
target_compile_definitions(cpu_moe PRIVATE SGLANG_CPU_FP8_CVT_FTZ)
target_compile_options(cpu_moe PRIVATE
  -O3 -Wno-unknown-pragmas -march=x86-64-v4 -mavx512bf16 -mavx512vnni
  -mamx-tile -mamx-bf16 -mamx-int8)
target_link_libraries(cpu_moe PRIVATE ${TORCH_LIBRARIES} OpenMP::OpenMP_CXX)
