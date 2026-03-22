// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once
#include "models/model_interface.h"
#include "kernels/kernel_provider.h"
#include "core/safetensors.h"
#include <memory>
#include <string>
#include <vector>

class LlamaModel : public ModelModule {
public:
    explicit LlamaModel(KernelProvider* kernels);
    ~LlamaModel() override;

    ModelConfig config() const override;
    void load_weights(const std::string& weight_path) override;
    KVCacheConfig kv_cache_config() const override;
    ForwardResult prefill(const std::vector<RequestContext>& requests,
                          cudaStream_t stream) override;
    ForwardResult decode(const std::vector<RequestContext>& requests,
                         cudaStream_t stream) override;
    int max_batch_size() const override;
    int max_tokens_per_batch() const override;

private:
    KernelProvider* kernels_;
    ModelConfig config_{};
    bool loaded_ = false;

    // GPU weight pointers (float32 for prototype)
    float* embed_tokens_ = nullptr;       // [vocab_size, hidden_size]
    float* lm_head_ = nullptr;            // [vocab_size, hidden_size]
    float* final_norm_weight_ = nullptr;  // [hidden_size]
    bool lm_head_tied_ = false;           // true if lm_head shares embed_tokens

    struct LayerWeights {
        float* input_norm = nullptr;
        float* post_attn_norm = nullptr;
        float* q_proj = nullptr;
        float* k_proj = nullptr;
        float* v_proj = nullptr;
        float* o_proj = nullptr;
        float* gate_proj = nullptr;
        float* up_proj = nullptr;
        float* down_proj = nullptr;
    };
    std::vector<LayerWeights> layers_;

    // KV cache pools — per-layer. Each layer has its own k_cache and v_cache
    // region. Total pool: num_layers * num_blocks * block_elems.
    float* k_cache_ = nullptr;
    float* v_cache_ = nullptr;
    size_t kv_cache_num_blocks_ = 0;
    size_t kv_block_elems_ = 0;  // num_kv_heads * block_size * head_dim

    // Get per-layer cache pointer
    float* k_cache_layer(int layer) const {
        return k_cache_ + static_cast<size_t>(layer) * kv_cache_num_blocks_ * kv_block_elems_;
    }
    float* v_cache_layer(int layer) const {
        return v_cache_ + static_cast<size_t>(layer) * kv_cache_num_blocks_ * kv_block_elems_;
    }

    // Workspace
    float* workspace_ = nullptr;
    size_t workspace_size_ = 0;

    float rope_theta_ = 10000.0f;

    int head_dim() const;

    // Forward pass implementation (shared between prefill and decode)
    void forward_impl(const std::vector<RequestContext>& requests,
                      bool is_prefill, ForwardResult& result,
                      cudaStream_t stream);

    // Allocate a GPU float buffer and copy from host, with optional dtype conversion
    float* alloc_and_upload(const void* host_data, const TensorInfo& info);

    void free_weights();
};

std::unique_ptr<ModelModule> create_llama_model(KernelProvider* kernels);
