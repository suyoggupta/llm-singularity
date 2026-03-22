cmake_minimum_required(VERSION 3.24)
project(llm-singularity LANGUAGES CXX CUDA)

# Allow FetchContent'd deps (e.g. sentencepiece) that declare older cmake_minimum
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

option(BUILD_TESTS "Build tests" ON)
set(MODEL "llama" CACHE STRING "Model to build")

# Find CUDA and cuBLAS
find_package(CUDAToolkit REQUIRED)

# Third-party deps
add_subdirectory(third_party)

# Core library
add_subdirectory(core)

# Kernels library
add_subdirectory(kernels)

# Model module
add_subdirectory(models/${MODEL})

# Main application
add_subdirectory(app)

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
