// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once
#include <cuda_runtime.h>

/// Launch SwiGLU activation kernel (element-wise).
/// output[i] = silu(gate[i]) * up[i]
/// where silu(x) = x * sigmoid(x) = x / (1 + exp(-x)).
void launch_silu_mul(const float* gate, const float* up, float* output,
                     int size, cudaStream_t stream);
