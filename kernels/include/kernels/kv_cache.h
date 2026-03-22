// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once
#include <cuda_runtime.h>

/// Write K and V vectors into paged KV cache blocks.
/// K: [num_tokens, num_kv_heads * head_dim] — computed K for this batch
/// V: [num_tokens, num_kv_heads * head_dim] — computed V for this batch
/// k_cache, v_cache: contiguous block pool.
///   Each block stores [num_kv_heads, block_size, head_dim].
/// block_table: [num_tokens] — block index for each token
/// block_offsets: [num_tokens] — offset within the block for each token
void launch_kv_cache_scatter(const float* K, const float* V,
                              float* k_cache, float* v_cache,
                              const int* block_table, const int* block_offsets,
                              int num_tokens, int num_kv_heads, int head_dim,
                              int block_size,
                              cudaStream_t stream);
