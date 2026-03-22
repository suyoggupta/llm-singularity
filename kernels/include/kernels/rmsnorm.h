// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once
#include <cuda_runtime.h>

/// Launch fused RMSNorm kernel.
/// output[r][c] = input[r][c] / rms(input[r]) * weight[c]
/// where rms(x) = sqrt(mean(x^2) + eps).
/// One thread block per row.
void launch_rms_norm(const float* input, const float* weight, float* output,
                     int rows, int hidden_size, float eps, cudaStream_t stream);
