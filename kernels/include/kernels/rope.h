// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once
#include <cuda_runtime.h>

/// Apply rotary position embeddings (RoPE) to Q and K tensors in-place.
/// q layout: [total_tokens, num_heads, head_dim]
/// k layout: [total_tokens, num_kv_heads, head_dim]
/// positions: [total_tokens]
/// For each position p and dimension pair (2i, 2i+1):
///   freq = 1.0 / (theta^(2i / head_dim))
///   angle = position * freq
///   (x[2i], x[2i+1]) = (x[2i]*cos - x[2i+1]*sin, x[2i]*sin + x[2i+1]*cos)
void launch_rope(float* q, float* k,
                 int total_tokens, int num_heads, int num_kv_heads, int head_dim,
                 const int* positions, float theta, cudaStream_t stream);
