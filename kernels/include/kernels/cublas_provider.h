// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once
#include "kernels/kernel_provider.h"
#include <cublas_v2.h>

class CublasProvider : public KernelProvider {
public:
    CublasProvider();
    ~CublasProvider();

    void gemm(GemmDescriptor desc, const void* A, const void* B, void* C,
              cudaStream_t stream) override;

    void fused_attention(AttentionDescriptor desc,
                         const void* Q, const void* K, const void* V,
                         void* output, const int* block_table,
                         int block_size, const int* seq_lens,
                         cudaStream_t stream) override;

    void rms_norm(const void* input, const void* weight, void* output,
                  int rows, int hidden_size, float eps,
                  DataType dtype, cudaStream_t stream) override;

    void silu_mul(const void* input, const void* gate, void* output,
                  int size, DataType dtype, cudaStream_t stream) override;

    void rope(void* q, void* k,
              int batch_size, int seq_len, int num_heads,
              int num_kv_heads, int head_dim,
              const int* positions, float theta,
              DataType dtype, cudaStream_t stream) override;

private:
    cublasHandle_t handle_;
};
