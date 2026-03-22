// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "kernels/cublas_provider.h"
#include "kernels/rmsnorm.h"
#include "kernels/activations.h"
#include "kernels/rope.h"

#include <stdexcept>

CublasProvider::CublasProvider() {
    cublasCreate(&handle_);
}

CublasProvider::~CublasProvider() {
    cublasDestroy(handle_);
}

void CublasProvider::gemm(GemmDescriptor desc, const void* A, const void* B, void* C,
                           cudaStream_t stream) {
    if (desc.input_dtype != DataType::kFloat32 || desc.output_dtype != DataType::kFloat32) {
        throw std::runtime_error("CublasProvider::gemm only supports float32");
    }

    cublasSetStream(handle_, stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Our data is row-major: A[M,K], B[K,N], C[M,N].
    // cuBLAS operates column-major. The trick to compute C = A*B in row-major:
    // Interpret everything as column-major transposed, then compute:
    //   C^T = B^T * A^T
    // In cuBLAS column-major terms (with no-op transpose since ^T of a
    // row-major matrix is just reading it as column-major):
    //   cublasSgemm(handle, N, N, M, K, &alpha, B, N, A, K, &beta, C, N)
    cublasSgemm(handle_,
                CUBLAS_OP_N, CUBLAS_OP_N,
                desc.N, desc.M, desc.K,
                &alpha,
                static_cast<const float*>(B), desc.N,
                static_cast<const float*>(A), desc.K,
                &beta,
                static_cast<float*>(C), desc.N);
}

void CublasProvider::fused_attention(AttentionDescriptor desc,
                                      const void* Q, const void* K, const void* V,
                                      void* output, const int* block_table,
                                      int block_size, const int* seq_lens,
                                      cudaStream_t stream) {
    (void)desc; (void)Q; (void)K; (void)V; (void)output;
    (void)block_table; (void)block_size; (void)seq_lens; (void)stream;
    throw std::runtime_error("CublasProvider::fused_attention not yet implemented — use Task 9");
}

void CublasProvider::rms_norm(const void* input, const void* weight, void* output,
                               int rows, int hidden_size, float eps,
                               DataType dtype, cudaStream_t stream) {
    (void)dtype;  // cast to float for now
    launch_rms_norm(static_cast<const float*>(input),
                    static_cast<const float*>(weight),
                    static_cast<float*>(output),
                    rows, hidden_size, eps, stream);
}

void CublasProvider::silu_mul(const void* input, const void* gate, void* output,
                               int size, DataType dtype, cudaStream_t stream) {
    (void)dtype;  // cast to float for now
    // KernelProvider interface: input=gate operand, gate=up operand
    launch_silu_mul(static_cast<const float*>(input),
                    static_cast<const float*>(gate),
                    static_cast<float*>(output),
                    size, stream);
}

void CublasProvider::rope(void* q, void* k,
                           int batch_size, int seq_len, int num_heads,
                           int num_kv_heads, int head_dim,
                           const int* positions, float theta,
                           DataType dtype, cudaStream_t stream) {
    (void)dtype;  // cast to float for now
    int total_tokens = batch_size * seq_len;
    launch_rope(static_cast<float*>(q),
                static_cast<float*>(k),
                total_tokens, num_heads, num_kv_heads, head_dim,
                positions, theta, stream);
}
