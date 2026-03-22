// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "kernels/embedding.h"

// Each block handles one token. Threads iterate over hidden_size elements.
__global__ void embedding_lookup_kernel(const int32_t* __restrict__ token_ids,
                                         const float* __restrict__ embed_table,
                                         float* __restrict__ output,
                                         int num_tokens, int hidden_size) {
    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    int token_id = token_ids[token_idx];
    const float* src = embed_table + static_cast<int64_t>(token_id) * hidden_size;
    float* dst = output + static_cast<int64_t>(token_idx) * hidden_size;

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        dst[i] = src[i];
    }
}

void launch_embedding_lookup(const int32_t* token_ids, const float* embed_table,
                              float* output, int num_tokens, int hidden_size,
                              cudaStream_t stream) {
    if (num_tokens == 0) return;
    int threads = (hidden_size < 256) ? hidden_size : 256;
    embedding_lookup_kernel<<<num_tokens, threads, 0, stream>>>(
        token_ids, embed_table, output, num_tokens, hidden_size);
}
