// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "kernels/attention.h"
#include <cfloat>
#include <cmath>
#include <cstdio>

// Paged attention decode kernel.
// Grid: (num_heads, batch_size)
// Block: up to 256 threads; each thread handles a subset of KV positions.
//
// Algorithm per (batch, head):
//   1. Load query vector Q[batch, head, :] into shared memory.
//   2. Each thread iterates over its assigned KV positions, computing QK dot products.
//   3. Two-pass online softmax: first find max and sum, then accumulate weighted V.
//      Actually we do a single-pass online softmax (numerically stable streaming).
//   4. Final reduction across threads to produce output[batch, head, :].

static constexpr int kBlockThreads = 256;

__global__ void paged_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ output,
    const int* __restrict__ block_table,
    int max_blocks_per_seq,
    const int* __restrict__ seq_lens,
    int num_heads, int num_kv_heads, int head_dim,
    int block_size, float scale)
{
    const int head_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    const int tid = threadIdx.x;

    const int seq_len = seq_lens[batch_idx];
    if (seq_len == 0) return;

    // GQA: map query head to kv head
    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);

    // Pointer to this request's query: Q[batch_idx, head_idx, :]
    const float* q_ptr = Q + (batch_idx * num_heads + head_idx) * head_dim;

    // Pointer to this request's block table
    const int* bt = block_table + batch_idx * max_blocks_per_seq;

    // Block stride for the cache: each block has [num_kv_heads, block_size, head_dim]
    const int block_stride = num_kv_heads * block_size * head_dim;

    // Shared memory for the query vector
    extern __shared__ float smem[];
    float* s_query = smem;  // [head_dim]

    // Load query into shared memory
    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_query[d] = q_ptr[d];
    }
    __syncthreads();

    // Each thread computes partial attention over its assigned KV positions.
    // We use online softmax: track running max, running sum of exp, and weighted V accumulator.
    float thread_max = -FLT_MAX;
    float thread_sum = 0.0f;
    // Per-thread V accumulator (stored in registers, head_dim is small for prototype)
    // For large head_dim, we'd need a different approach, but head_dim <= 128 typically.
    // We'll store in local array; for head_dim=4 this is fine.
    float v_acc[128];  // max supported head_dim
    for (int d = 0; d < head_dim; d++) {
        v_acc[d] = 0.0f;
    }

    for (int pos = tid; pos < seq_len; pos += blockDim.x) {
        // Find which block and offset within block
        int cache_block_idx = bt[pos / block_size];
        int offset_in_block = pos % block_size;

        // K address: k_cache[cache_block_idx][kv_head_idx][offset_in_block][d]
        const float* k_ptr = k_cache
            + cache_block_idx * block_stride
            + kv_head_idx * block_size * head_dim
            + offset_in_block * head_dim;

        // Compute Q dot K
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += s_query[d] * k_ptr[d];
        }
        score *= scale;

        // Online softmax update
        // V address
        const float* v_ptr = v_cache
            + cache_block_idx * block_stride
            + kv_head_idx * block_size * head_dim
            + offset_in_block * head_dim;

        if (score > thread_max) {
            float correction = expf(thread_max - score);
            thread_sum = thread_sum * correction;
            for (int d = 0; d < head_dim; d++) {
                v_acc[d] = v_acc[d] * correction;
            }
            thread_max = score;
        }
        float w = expf(score - thread_max);
        thread_sum += w;
        for (int d = 0; d < head_dim; d++) {
            v_acc[d] += w * v_ptr[d];
        }
    }

    // Now we need to reduce across threads: merge (thread_max, thread_sum, v_acc).
    // Use shared memory for the reduction.
    // Layout: [kBlockThreads] for max, [kBlockThreads] for sum,
    //         [kBlockThreads * head_dim] for v_acc
    // But that's a lot of smem for large head_dim. For the prototype with small head_dim, it's fine.
    // We'll reuse smem after the query is no longer needed.
    __syncthreads();

    // smem layout for reduction:
    // s_max: [kBlockThreads]
    // s_sum: [kBlockThreads]
    // s_vacc: [kBlockThreads * head_dim]
    float* s_max = smem;
    float* s_sum = smem + blockDim.x;
    float* s_vacc = smem + 2 * blockDim.x;

    s_max[tid] = thread_max;
    s_sum[tid] = thread_sum;
    for (int d = 0; d < head_dim; d++) {
        s_vacc[tid * head_dim + d] = v_acc[d];
    }
    __syncthreads();

    // Tree reduction
    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float my_max = s_max[tid];
            float other_max = s_max[tid + stride];
            float new_max = fmaxf(my_max, other_max);

            float my_correction = expf(my_max - new_max);
            float other_correction = expf(other_max - new_max);

            s_sum[tid] = s_sum[tid] * my_correction + s_sum[tid + stride] * other_correction;
            s_max[tid] = new_max;

            for (int d = 0; d < head_dim; d++) {
                s_vacc[tid * head_dim + d] =
                    s_vacc[tid * head_dim + d] * my_correction +
                    s_vacc[(tid + stride) * head_dim + d] * other_correction;
            }
        }
        __syncthreads();
    }

    // Thread 0 writes the output
    if (tid == 0) {
        float* out_ptr = output + (batch_idx * num_heads + head_idx) * head_dim;
        float inv_sum = (s_sum[0] > 0.0f) ? (1.0f / s_sum[0]) : 0.0f;
        for (int d = 0; d < head_dim; d++) {
            out_ptr[d] = s_vacc[d] * inv_sum;
        }
    }
}

void launch_paged_attention(
    const float* Q,
    const float* k_cache, const float* v_cache,
    float* output,
    const int* block_table, int max_blocks_per_seq,
    const int* seq_lens,
    int batch_size, int num_heads, int num_kv_heads, int head_dim,
    int block_size,
    cudaStream_t stream)
{
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    // Choose thread count to keep shared memory under 48KB.
    // Reduction needs: (2*threads + threads*head_dim) * 4 bytes.
    // For head_dim=128: threads * (2 + 128) * 4 = threads * 520.
    // 48KB / 520 ≈ 94 threads → round down to power of 2 = 64.
    // Use a single warp (32 threads) for simplicity and to avoid shared memory issues.
    // For production, a more sophisticated approach is needed.
    int threads = 32;

    dim3 grid(num_heads, batch_size);
    dim3 block(threads);

    // Shared memory: query [head_dim] during load phase,
    // then reused for reduction: max[threads] + sum[threads] + vacc[threads * head_dim]
    size_t smem_query = head_dim * sizeof(float);
    size_t smem_reduce = (2 * threads + threads * head_dim) * sizeof(float);
    size_t smem_size = (smem_query > smem_reduce) ? smem_query : smem_reduce;

    paged_attention_kernel<<<grid, block, smem_size, stream>>>(
        Q, k_cache, v_cache, output,
        block_table, max_blocks_per_seq, seq_lens,
        num_heads, num_kv_heads, head_dim,
        block_size, scale);

    // Debug: check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ATTENTION KERNEL LAUNCH ERROR: %s (threads=%d, smem=%zu)\n",
                cudaGetErrorString(err), threads, smem_size);
    }
}
