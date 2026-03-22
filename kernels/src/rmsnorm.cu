// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "kernels/rmsnorm.h"
#include <cmath>

// Warp-level reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction using shared memory
__device__ float block_reduce_sum(float val) {
    __shared__ float shared[32]; // max 32 warps per block
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // First warp reduces across warps
    int num_warps = (blockDim.x + 31) / 32;
    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : 0.0f;
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}

__global__ void rms_norm_kernel(const float* __restrict__ input,
                                const float* __restrict__ weight,
                                float* __restrict__ output,
                                int hidden_size, float eps) {
    int row = blockIdx.x;
    const float* row_in = input + row * hidden_size;
    float* row_out = output + row * hidden_size;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = row_in[i];
        sum_sq += v * v;
    }

    sum_sq = block_reduce_sum(sum_sq);

    // Broadcast rms via shared memory
    __shared__ float s_rms;
    if (threadIdx.x == 0) {
        s_rms = rsqrtf(sum_sq / static_cast<float>(hidden_size) + eps);
    }
    __syncthreads();

    float rms_inv = s_rms;

    // Normalize and apply weight
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        row_out[i] = row_in[i] * rms_inv * weight[i];
    }
}

void launch_rms_norm(const float* input, const float* weight, float* output,
                     int rows, int hidden_size, float eps, cudaStream_t stream) {
    int threads = (hidden_size < 1024) ? hidden_size : 1024;
    // Round up to nearest multiple of 32
    threads = ((threads + 31) / 32) * 32;
    if (threads < 32) threads = 32;

    rms_norm_kernel<<<rows, threads, 0, stream>>>(input, weight, output, hidden_size, eps);
}
