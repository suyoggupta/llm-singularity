// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "kernels/kv_cache.h"

// Each block handles one token. Threads iterate over num_kv_heads * head_dim.
// Cache layout per block: [num_kv_heads, block_size, head_dim]
// K input layout: [num_tokens, num_kv_heads * head_dim]
__global__ void kv_cache_scatter_kernel(const float* __restrict__ K,
                                         const float* __restrict__ V,
                                         float* __restrict__ k_cache,
                                         float* __restrict__ v_cache,
                                         const int* __restrict__ block_table,
                                         const int* __restrict__ block_offsets,
                                         int num_tokens, int num_kv_heads,
                                         int head_dim, int block_size) {
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    int block_idx = block_table[token_idx];
    int offset_in_block = block_offsets[token_idx];
    int kv_dim = num_kv_heads * head_dim;

    // block_stride = num_kv_heads * block_size * head_dim (elements per block)
    int block_stride = num_kv_heads * block_size * head_dim;

    const float* k_src = K + static_cast<int64_t>(token_idx) * kv_dim;
    const float* v_src = V + static_cast<int64_t>(token_idx) * kv_dim;

    // For each kv_head h, write head_dim elements to:
    //   cache[block_idx * block_stride + h * block_size * head_dim + offset_in_block * head_dim + d]
    for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) {
        int h = i / head_dim;
        int d = i % head_dim;
        int cache_offset = block_idx * block_stride + h * block_size * head_dim +
                           offset_in_block * head_dim + d;
        k_cache[cache_offset] = k_src[i];
        v_cache[cache_offset] = v_src[i];
    }
}

void launch_kv_cache_scatter(const float* K, const float* V,
                              float* k_cache, float* v_cache,
                              const int* block_table, const int* block_offsets,
                              int num_tokens, int num_kv_heads, int head_dim,
                              int block_size,
                              cudaStream_t stream) {
    if (num_tokens == 0) return;
    int kv_dim = num_kv_heads * head_dim;
    int threads = (kv_dim < 256) ? kv_dim : 256;
    kv_cache_scatter_kernel<<<num_tokens, threads, 0, stream>>>(
        K, V, k_cache, v_cache, block_table, block_offsets,
        num_tokens, num_kv_heads, head_dim, block_size);
}
