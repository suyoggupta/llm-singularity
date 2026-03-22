// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once
#include <cuda_runtime.h>

/// Gather embedding vectors for given token IDs.
/// token_ids: [num_tokens] — indices into embed_table
/// embed_table: [vocab_size, hidden_size]
/// output: [num_tokens, hidden_size]
void launch_embedding_lookup(const int32_t* token_ids, const float* embed_table,
                              float* output, int num_tokens, int hidden_size,
                              cudaStream_t stream);
