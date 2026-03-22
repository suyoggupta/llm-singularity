// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "kernels/residual.h"

__global__ void residual_add_kernel(const float* __restrict__ a,
                                     const float* __restrict__ b,
                                     float* __restrict__ output,
                                     int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = a[idx] + b[idx];
    }
}

void launch_residual_add(const float* a, const float* b, float* output,
                          int size, cudaStream_t stream) {
    if (size == 0) return;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    residual_add_kernel<<<blocks, threads, 0, stream>>>(a, b, output, size);
}
