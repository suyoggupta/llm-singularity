#pragma once
#include <cuda_runtime.h>
#include "core/types.h"

struct GemmDescriptor {
    int M, N, K;
    DataType input_dtype;
    DataType output_dtype;
    bool transA = false;
    bool transB = false;
};

struct AttentionDescriptor {
    int batch_size;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int max_seq_len;
    DataType dtype;
};

class KernelProvider {
public:
    virtual ~KernelProvider() = default;

    virtual void gemm(GemmDescriptor desc,
                      const void* A, const void* B, void* C,
                      cudaStream_t stream) = 0;

    virtual void fused_attention(AttentionDescriptor desc,
                                 const void* Q, const void* K, const void* V,
                                 void* output,
                                 const int* block_table,
                                 int block_size,
                                 const int* seq_lens,
                                 cudaStream_t stream) = 0;

    virtual void rms_norm(const void* input, const void* weight, void* output,
                          int rows, int hidden_size, float eps,
                          DataType dtype, cudaStream_t stream) = 0;

    virtual void silu_mul(const void* input, const void* gate, void* output,
                          int size, DataType dtype, cudaStream_t stream) = 0;

    virtual void rope(void* q, void* k,
                      int batch_size, int seq_len, int num_heads,
                      int num_kv_heads, int head_dim,
                      const int* positions, float theta,
                      DataType dtype, cudaStream_t stream) = 0;
};
