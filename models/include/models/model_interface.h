#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <cuda_runtime.h>
#include "core/types.h"
#include "kernels/kernel_provider.h"

struct ModelConfig {
    int num_layers;
    int hidden_size;
    int num_attention_heads;
    int num_kv_heads;
    int intermediate_size;
    int vocab_size;
    int max_seq_len;
    int eos_token_id;
    std::vector<int> stop_token_ids;
    DataType dtype;
    std::unordered_map<std::string, std::string> extra;
};

struct KVCacheConfig {
    int block_size;
    int max_blocks;
    DataType cache_dtype;
};

struct RequestContext {
    int64_t request_id;
    std::vector<int32_t> token_ids;
    int seq_len;
    int prefill_start_pos;
    int prefill_chunk_len;
    std::vector<int> block_table;
    int max_new_tokens;
};

struct ForwardResult {
    std::vector<float> logits;
    int vocab_size;
};

class ModelModule {
public:
    virtual ~ModelModule() = default;
    virtual ModelConfig config() const = 0;
    virtual void load_weights(const std::string& weight_path) = 0;
    virtual KVCacheConfig kv_cache_config() const = 0;
    virtual ForwardResult prefill(const std::vector<RequestContext>& requests, cudaStream_t stream) = 0;
    virtual ForwardResult decode(const std::vector<RequestContext>& requests, cudaStream_t stream) = 0;
    virtual int max_batch_size() const = 0;
    virtual int max_tokens_per_batch() const = 0;
    virtual bool supports_in_flight_batching() const { return true; }
};

using ModelModuleFactory = std::unique_ptr<ModelModule>(*)(KernelProvider*);
