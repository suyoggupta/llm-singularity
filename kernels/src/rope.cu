// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "kernels/rope.h"
#include <cmath>

// Apply RoPE to a single head's dimension pair
__device__ void apply_rope_pair(float* x, int dim_pair, int head_dim,
                                int position, float theta) {
    float freq = 1.0f / powf(theta, (2.0f * dim_pair) / static_cast<float>(head_dim));
    float angle = static_cast<float>(position) * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    int idx0 = 2 * dim_pair;
    int idx1 = 2 * dim_pair + 1;
    float x0 = x[idx0];
    float x1 = x[idx1];
    x[idx0] = x0 * cos_val - x1 * sin_val;
    x[idx1] = x0 * sin_val + x1 * cos_val;
}

// Grid: (total_tokens, total_heads, half_head_dim)
// where total_heads = num_heads + num_kv_heads
__global__ void rope_kernel(float* __restrict__ q,
                            float* __restrict__ k,
                            int total_tokens, int num_heads, int num_kv_heads,
                            int head_dim, const int* __restrict__ positions,
                            float theta) {
    int token = blockIdx.x;
    int head_and_pair = blockIdx.y * blockDim.x + threadIdx.x;

    int total_heads = num_heads + num_kv_heads;
    int half_dim = head_dim / 2;
    int total_work = total_heads * half_dim;

    if (token >= total_tokens || head_and_pair >= total_work) return;

    int head = head_and_pair / half_dim;
    int dim_pair = head_and_pair % half_dim;
    int position = positions[token];

    if (head < num_heads) {
        // Q tensor: [total_tokens, num_heads, head_dim]
        float* q_head = q + token * num_heads * head_dim + head * head_dim;
        apply_rope_pair(q_head, dim_pair, head_dim, position, theta);
    } else {
        // K tensor: [total_tokens, num_kv_heads, head_dim]
        int kv_head = head - num_heads;
        float* k_head = k + token * num_kv_heads * head_dim + kv_head * head_dim;
        apply_rope_pair(k_head, dim_pair, head_dim, position, theta);
    }
}

void launch_rope(float* q, float* k,
                 int total_tokens, int num_heads, int num_kv_heads, int head_dim,
                 const int* positions, float theta, cudaStream_t stream) {
    int total_heads = num_heads + num_kv_heads;
    int half_dim = head_dim / 2;
    int total_work_per_token = total_heads * half_dim;

    constexpr int threads = 256;
    int blocks_y = (total_work_per_token + threads - 1) / threads;

    dim3 grid(total_tokens, blocks_y);
    rope_kernel<<<grid, threads, 0, stream>>>(q, k, total_tokens, num_heads,
                                               num_kv_heads, head_dim, positions,
                                               theta);
}
