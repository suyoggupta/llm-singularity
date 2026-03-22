// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "kernels/activations.h"

__global__ void silu_mul_kernel(const float* __restrict__ gate,
                                const float* __restrict__ up,
                                float* __restrict__ output,
                                int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = gate[idx];
        float silu_g = g / (1.0f + expf(-g));
        output[idx] = silu_g * up[idx];
    }
}

void launch_silu_mul(const float* gate, const float* up, float* output,
                     int size, cudaStream_t stream) {
    constexpr int threads = 256;
    int blocks = (size + threads - 1) / threads;
    silu_mul_kernel<<<blocks, threads, 0, stream>>>(gate, up, output, size);
}
