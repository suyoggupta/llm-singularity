// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include <gtest/gtest.h>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#include "kernels/cublas_provider.h"
#include "kernels/embedding.h"
#include "kernels/kv_cache.h"
#include "kernels/residual.h"
#include "models/model_interface.h"

// Forward declaration — defined in llama_model.h
class LlamaModel;
std::unique_ptr<ModelModule> create_llama_model(KernelProvider* kernels);

// ---------------------------------------------------------------------------
// Helper kernel unit tests (always run, no model needed)
// ---------------------------------------------------------------------------

TEST(ResidualKernelTest, BasicAdd) {
    const int N = 1024;
    std::vector<float> h_a(N), h_b(N), h_out(N);
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));

    cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    launch_residual_add(d_a, d_b, d_out, N, nullptr);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(h_out[i], h_a[i] + h_b[i], 1e-5f);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
}

TEST(EmbeddingKernelTest, Lookup) {
    const int vocab = 10;
    const int hidden = 4;
    const int num_tokens = 3;

    // Build embedding table: row i = [i*10, i*10+1, i*10+2, i*10+3]
    std::vector<float> h_table(vocab * hidden);
    for (int i = 0; i < vocab * hidden; ++i) {
        h_table[i] = static_cast<float>(i);
    }
    std::vector<int32_t> h_ids = {0, 5, 9};
    std::vector<float> h_out(num_tokens * hidden);

    float* d_table;
    int32_t* d_ids;
    float* d_out;
    cudaMalloc(&d_table, h_table.size() * sizeof(float));
    cudaMalloc(&d_ids, h_ids.size() * sizeof(int32_t));
    cudaMalloc(&d_out, h_out.size() * sizeof(float));

    cudaMemcpy(d_table, h_table.data(), h_table.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_ids, h_ids.data(), h_ids.size() * sizeof(int32_t),
               cudaMemcpyHostToDevice);

    launch_embedding_lookup(d_ids, d_table, d_out, num_tokens, hidden, nullptr);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Token 0 → row 0: [0, 1, 2, 3]
    EXPECT_NEAR(h_out[0], 0.0f, 1e-5f);
    EXPECT_NEAR(h_out[3], 3.0f, 1e-5f);
    // Token 5 → row 5: [20, 21, 22, 23]
    EXPECT_NEAR(h_out[4], 20.0f, 1e-5f);
    // Token 9 → row 9: [36, 37, 38, 39]
    EXPECT_NEAR(h_out[8], 36.0f, 1e-5f);

    cudaFree(d_table);
    cudaFree(d_ids);
    cudaFree(d_out);
}

TEST(KVCacheScatterTest, BasicScatter) {
    const int num_tokens = 2;
    const int num_kv_heads = 1;
    const int head_dim = 4;
    const int block_size = 4;
    const int num_blocks = 2;

    int kv_dim = num_kv_heads * head_dim;
    int block_elems = num_kv_heads * block_size * head_dim;

    std::vector<float> h_k(num_tokens * kv_dim);
    std::vector<float> h_v(num_tokens * kv_dim);
    for (int i = 0; i < num_tokens * kv_dim; ++i) {
        h_k[i] = static_cast<float>(i + 1);
        h_v[i] = static_cast<float>((i + 1) * 10);
    }

    // Token 0 → block 0, offset 2; Token 1 → block 1, offset 0
    std::vector<int> h_bt = {0, 1};
    std::vector<int> h_bo = {2, 0};

    float *d_k, *d_v, *d_kc, *d_vc;
    int *d_bt, *d_bo;
    cudaMalloc(&d_k, h_k.size() * sizeof(float));
    cudaMalloc(&d_v, h_v.size() * sizeof(float));
    cudaMalloc(&d_kc, num_blocks * block_elems * sizeof(float));
    cudaMalloc(&d_vc, num_blocks * block_elems * sizeof(float));
    cudaMalloc(&d_bt, h_bt.size() * sizeof(int));
    cudaMalloc(&d_bo, h_bo.size() * sizeof(int));

    cudaMemset(d_kc, 0, num_blocks * block_elems * sizeof(float));
    cudaMemset(d_vc, 0, num_blocks * block_elems * sizeof(float));
    cudaMemcpy(d_k, h_k.data(), h_k.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), h_v.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_bt, h_bt.data(), h_bt.size() * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_bo, h_bo.data(), h_bo.size() * sizeof(int),
               cudaMemcpyHostToDevice);

    launch_kv_cache_scatter(d_k, d_v, d_kc, d_vc, d_bt, d_bo,
                            num_tokens, num_kv_heads, head_dim,
                            block_size, nullptr);
    cudaDeviceSynchronize();

    std::vector<float> h_kc(num_blocks * block_elems, 0.0f);
    std::vector<float> h_vc(num_blocks * block_elems, 0.0f);
    cudaMemcpy(h_kc.data(), d_kc, h_kc.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vc.data(), d_vc, h_vc.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Token 0 goes to block 0, offset 2, head 0:
    // cache[0 * block_elems + 0 * block_size * head_dim + 2 * head_dim + d]
    // = cache[0 + 0 + 8 + d] = cache[8..11]
    EXPECT_NEAR(h_kc[8], 1.0f, 1e-5f);
    EXPECT_NEAR(h_kc[9], 2.0f, 1e-5f);
    EXPECT_NEAR(h_kc[10], 3.0f, 1e-5f);
    EXPECT_NEAR(h_kc[11], 4.0f, 1e-5f);

    // Token 1 goes to block 1, offset 0:
    // cache[1 * block_elems + 0] = cache[16..19]
    EXPECT_NEAR(h_kc[16], 5.0f, 1e-5f);
    EXPECT_NEAR(h_vc[16], 50.0f, 1e-5f);

    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_kc);
    cudaFree(d_vc);
    cudaFree(d_bt);
    cudaFree(d_bo);
}

// ---------------------------------------------------------------------------
// Model-level tests (require TEST_MODEL_PATH)
// ---------------------------------------------------------------------------

TEST(LlamaModelTest, ConfigParsing) {
    const char* model_path = std::getenv("TEST_MODEL_PATH");
    if (!model_path) {
        GTEST_SKIP() << "TEST_MODEL_PATH not set — skipping model test";
    }

    auto provider = std::make_unique<CublasProvider>();
    auto model = create_llama_model(provider.get());
    model->load_weights(model_path);

    auto cfg = model->config();
    EXPECT_GT(cfg.num_layers, 0);
    EXPECT_GT(cfg.hidden_size, 0);
    EXPECT_GT(cfg.vocab_size, 0);
    EXPECT_GT(cfg.num_attention_heads, 0);
    EXPECT_GT(cfg.num_kv_heads, 0);
    EXPECT_EQ(cfg.dtype, DataType::kFloat32);
    // hidden_size should be divisible by num_attention_heads
    EXPECT_EQ(cfg.hidden_size % cfg.num_attention_heads, 0);
}

TEST(LlamaModelTest, ForwardPassShape) {
    const char* model_path = std::getenv("TEST_MODEL_PATH");
    if (!model_path) {
        GTEST_SKIP() << "TEST_MODEL_PATH not set — skipping model test";
    }

    auto provider = std::make_unique<CublasProvider>();
    auto model = create_llama_model(provider.get());
    model->load_weights(model_path);

    auto cfg = model->config();
    auto kvc = model->kv_cache_config();

    // Create a simple decode request
    RequestContext req;
    req.request_id = 1;
    req.token_ids = {1, 2, 3};
    req.seq_len = 3;
    req.prefill_start_pos = 0;
    req.prefill_chunk_len = 3;
    // Allocate blocks for this request
    int num_blocks_needed = (req.seq_len + kvc.block_size - 1) / kvc.block_size;
    for (int i = 0; i < num_blocks_needed; ++i) {
        req.block_table.push_back(i);
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Run prefill
    auto result = model->prefill({req}, stream);
    cudaStreamSynchronize(stream);

    // Should have 1 request's logits (last token only)
    EXPECT_EQ(result.vocab_size, cfg.vocab_size);
    EXPECT_EQ(static_cast<int>(result.logits.size()), cfg.vocab_size);

    cudaStreamDestroy(stream);
}
