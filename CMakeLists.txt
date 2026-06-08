cmake_minimum_required(VERSION 3.24)

# CUDA arch: respect an explicit CMAKE_CUDA_ARCHITECTURES cache value, else autodetect.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures")
endif()

project(mpm_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Locate Python, nanobind, and the JAX FFI header from whichever Python is
# running scikit-build-core.
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
    COMMAND ${Python_EXECUTABLE} -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _nanobind_result
)
if(NOT _nanobind_result EQUAL 0 OR nanobind_ROOT STREQUAL "")
    message(FATAL_ERROR
        "python -m nanobind --cmake_dir failed. Add 'nanobind' to the Pixi "
        "environment and [build-system] requires.")
endif()
find_package(nanobind CONFIG REQUIRED)

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import jax.ffi; print(jax.ffi.include_dir())"
    OUTPUT_VARIABLE JAX_FFI_INCLUDE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _ffi_result
)
if(NOT _ffi_result EQUAL 0 OR JAX_FFI_INCLUDE STREQUAL "")
    message(FATAL_ERROR
        "jax.ffi.include_dir() failed. Add 'jax' to [build-system] requires "
        "or run from an env where jax is importable.")
endif()
message(STATUS "JAX FFI include: ${JAX_FFI_INCLUDE}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

set(KERNEL_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/mpm_jax/p2g/cuda/kernels)
set(P2G_FFI_SOURCES
    ${KERNEL_SRC_DIR}/p2g_ffi_module.cc
    ${KERNEL_SRC_DIR}/p2g_v1.cu
    ${KERNEL_SRC_DIR}/p2g_v2.cu
    ${KERNEL_SRC_DIR}/p2g_v3.cu
)

nanobind_add_module(_p2g_ffi NB_STATIC NOMINSIZE ${P2G_FFI_SOURCES})
target_include_directories(_p2g_ffi PRIVATE ${JAX_FFI_INCLUDE})
target_compile_options(_p2g_ffi PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    # -lineinfo: source-level attribution for ncu/Nsight (no codegen change,
    # unlike -G); this project's analysis method leans on per-kernel ncu.
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    # -Xptxas -v: print per-kernel registers / spills / shared-mem usage at
    # build time, feeding the occupancy/roofline narrative.
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-return-type>
    $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=940,2473>
)

install(TARGETS _p2g_ffi LIBRARY DESTINATION mpm_jax/p2g/cuda)
