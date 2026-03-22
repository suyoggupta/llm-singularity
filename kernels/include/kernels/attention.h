// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once
#include <cuda_runtime.h>

/// Launch paged attention for decode (single query token per request).
/// Q: [batch_size, num_heads, head_dim] — one query token per request
/// k_cache, v_cache: contiguous block pools.
///   Each block stores [num_kv_heads, block_size, head_dim] for K (or V).
/// block_table: [batch_size, max_blocks_per_seq] — maps to pool block indices
/// seq_lens: [batch_size] — number of KV tokens each request attends to
/// output: [batch_size, num_heads, head_dim]
void launch_paged_attention(
    const float* Q,
    const float* k_cache, const float* v_cache,
    float* output,
    const int* block_table, int max_blocks_per_seq,
    const int* seq_lens,
    int batch_size, int num_heads, int num_kv_heads, int head_dim,
    int block_size,
    cudaStream_t stream);
