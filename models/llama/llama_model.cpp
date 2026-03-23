// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include "llama_model.h"
#include "kernels/attention.h"
#include "kernels/embedding.h"
#include "kernels/kv_cache.h"
#include "kernels/residual.h"

#include <algorithm>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include <regex>
#include <stdexcept>

#include <cuda_fp16.h>
#include <nlohmann/json.hpp>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert fp16 array to fp32 on CPU.
static void convert_f16_to_f32(const void* src, float* dst, int64_t n) {
    const __half* h = static_cast<const __half*>(src);
    for (int64_t i = 0; i < n; ++i) {
        dst[i] = __half2float(h[i]);
    }
}

/// Convert bf16 array to fp32 on CPU. BF16 is stored as uint16_t.
static void convert_bf16_to_f32(const void* src, float* dst, int64_t n) {
    const uint16_t* bf = static_cast<const uint16_t*>(src);
    for (int64_t i = 0; i < n; ++i) {
        // BF16 → F32: shift 16 bits left (upper 16 bits of float32)
        uint32_t bits = static_cast<uint32_t>(bf[i]) << 16;
        float val;
        std::memcpy(&val, &bits, sizeof(float));
        dst[i] = val;
    }
}

/// List files in a directory matching a regex pattern.
static std::vector<std::string> glob_files(const std::string& dir,
                                            const std::string& pattern) {
    std::vector<std::string> result;
    std::regex re(pattern);
    DIR* d = opendir(dir.c_str());
    if (!d) return result;
    struct dirent* entry;
    while ((entry = readdir(d)) != nullptr) {
        std::string name = entry->d_name;
        if (std::regex_match(name, re)) {
            result.push_back(dir + "/" + name);
        }
    }
    closedir(d);
    std::sort(result.begin(), result.end());
    return result;
}

// ---------------------------------------------------------------------------
// LlamaModel
// ---------------------------------------------------------------------------

LlamaModel::LlamaModel(KernelProvider* kernels) : kernels_(kernels) {}

LlamaModel::~LlamaModel() {
    free_weights();
    if (k_cache_) cudaFree(k_cache_);
    if (v_cache_) cudaFree(v_cache_);
    if (workspace_) cudaFree(workspace_);
}

void LlamaModel::free_weights() {
    if (embed_tokens_) cudaFree(embed_tokens_);
    if (lm_head_ && !lm_head_tied_) cudaFree(lm_head_);
    if (final_norm_weight_) cudaFree(final_norm_weight_);
    for (auto& l : layers_) {
        if (l.input_norm) cudaFree(l.input_norm);
        if (l.post_attn_norm) cudaFree(l.post_attn_norm);
        if (l.q_proj) cudaFree(l.q_proj);
        if (l.k_proj) cudaFree(l.k_proj);
        if (l.v_proj) cudaFree(l.v_proj);
        if (l.o_proj) cudaFree(l.o_proj);
        if (l.gate_proj) cudaFree(l.gate_proj);
        if (l.up_proj) cudaFree(l.up_proj);
        if (l.down_proj) cudaFree(l.down_proj);
    }
    embed_tokens_ = nullptr;
    lm_head_ = nullptr;
    final_norm_weight_ = nullptr;
    layers_.clear();
    loaded_ = false;
}

int LlamaModel::head_dim() const {
    return config_.hidden_size / config_.num_attention_heads;
}

ModelConfig LlamaModel::config() const { return config_; }

KVCacheConfig LlamaModel::kv_cache_config() const {
    KVCacheConfig kvc;
    kvc.block_size = 16;
    kvc.max_blocks = 1024;
    kvc.cache_dtype = DataType::kFloat32;
    return kvc;
}

int LlamaModel::max_batch_size() const { return 256; }
int LlamaModel::max_tokens_per_batch() const { return 2048; }

float* LlamaModel::alloc_and_upload(const void* host_data,
                                     const TensorInfo& info) {
    int64_t num_elems = info.num_elements();
    size_t gpu_bytes = num_elems * sizeof(float);
    float* gpu_ptr = nullptr;
    cudaMalloc(&gpu_ptr, gpu_bytes);

    if (info.dtype == DataType::kFloat32) {
        cudaMemcpy(gpu_ptr, host_data, gpu_bytes, cudaMemcpyHostToDevice);
    } else {
        // Convert to f32 on CPU then upload
        std::vector<float> f32_buf(num_elems);
        if (info.dtype == DataType::kFloat16) {
            convert_f16_to_f32(host_data, f32_buf.data(), num_elems);
        } else if (info.dtype == DataType::kBFloat16) {
            convert_bf16_to_f32(host_data, f32_buf.data(), num_elems);
        }
        cudaMemcpy(gpu_ptr, f32_buf.data(), gpu_bytes, cudaMemcpyHostToDevice);
    }
    return gpu_ptr;
}

// ---------------------------------------------------------------------------
// load_weights
// ---------------------------------------------------------------------------

void LlamaModel::load_weights(const std::string& weight_path) {
    free_weights();

    // 1. Read config.json
    std::string config_path = weight_path + "/config.json";
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        throw std::runtime_error("LlamaModel: cannot open " + config_path);
    }
    nlohmann::json cfg = nlohmann::json::parse(config_file);

    config_.num_layers = cfg.at("num_hidden_layers").get<int>();
    config_.hidden_size = cfg.at("hidden_size").get<int>();
    config_.num_attention_heads = cfg.at("num_attention_heads").get<int>();
    config_.num_kv_heads = cfg.value("num_key_value_heads",
                                      config_.num_attention_heads);
    config_.intermediate_size = cfg.at("intermediate_size").get<int>();
    config_.vocab_size = cfg.at("vocab_size").get<int>();
    config_.max_seq_len = cfg.value("max_position_embeddings", 2048);
    config_.dtype = DataType::kFloat32;

    // eos_token_id may be int or array
    if (cfg.contains("eos_token_id")) {
        auto& eos = cfg["eos_token_id"];
        if (eos.is_array()) {
            config_.eos_token_id = eos[0].get<int>();
            for (auto& v : eos) {
                config_.stop_token_ids.push_back(v.get<int>());
            }
        } else {
            config_.eos_token_id = eos.get<int>();
            config_.stop_token_ids.push_back(config_.eos_token_id);
        }
    }

    rope_theta_ = cfg.value("rope_theta", 10000.0f);

    // 2. Allocate layer weights vector
    layers_.resize(config_.num_layers);

    // 3. Find and load safetensors files
    auto st_files = glob_files(weight_path, "model.*\\.safetensors");
    if (st_files.empty()) {
        // Try single file without shard index
        st_files = glob_files(weight_path, "model\\.safetensors");
    }
    if (st_files.empty()) {
        throw std::runtime_error(
            "LlamaModel: no safetensors files found in " + weight_path);
    }

    for (const auto& st_path : st_files) {
        SafetensorsFile sf;
        sf.open(st_path);

        for (const auto& name : sf.tensor_names()) {
            TensorInfo info = sf.tensor_info(name);
            const void* data = sf.tensor_data(name);
            float* gpu = alloc_and_upload(data, info);

            // Map HF tensor names to member pointers
            if (name == "model.embed_tokens.weight") {
                embed_tokens_ = gpu;
            } else if (name == "model.norm.weight") {
                final_norm_weight_ = gpu;
            } else if (name == "lm_head.weight") {
                lm_head_ = gpu;
            } else if (name.find("model.layers.") == 0) {
                // Parse layer index: model.layers.{i}.xxx
                size_t dot1 = name.find('.', 13);  // after "model.layers."
                if (dot1 == std::string::npos) continue;
                int layer_idx = std::stoi(name.substr(13, dot1 - 13));
                if (layer_idx < 0 ||
                    layer_idx >= static_cast<int>(layers_.size())) {
                    cudaFree(gpu);
                    continue;
                }
                std::string suffix = name.substr(dot1 + 1);
                auto& lw = layers_[layer_idx];

                if (suffix == "input_layernorm.weight") {
                    lw.input_norm = gpu;
                } else if (suffix == "post_attention_layernorm.weight") {
                    lw.post_attn_norm = gpu;
                } else if (suffix == "self_attn.q_proj.weight") {
                    lw.q_proj = gpu;
                } else if (suffix == "self_attn.k_proj.weight") {
                    lw.k_proj = gpu;
                } else if (suffix == "self_attn.v_proj.weight") {
                    lw.v_proj = gpu;
                } else if (suffix == "self_attn.o_proj.weight") {
                    lw.o_proj = gpu;
                } else if (suffix == "mlp.gate_proj.weight") {
                    lw.gate_proj = gpu;
                } else if (suffix == "mlp.up_proj.weight") {
                    lw.up_proj = gpu;
                } else if (suffix == "mlp.down_proj.weight") {
                    lw.down_proj = gpu;
                } else {
                    // Unknown layer tensor, free it
                    cudaFree(gpu);
                }
            } else {
                // Unknown top-level tensor
                cudaFree(gpu);
            }
        }
    }

    // If lm_head not found, tie to embed_tokens
    if (!lm_head_ && embed_tokens_) {
        lm_head_ = embed_tokens_;
        lm_head_tied_ = true;
    }

    // 4. Allocate workspace
    int max_dim = std::max({config_.hidden_size, config_.intermediate_size,
                            config_.vocab_size});
    // Enough for multiple intermediate buffers with max 2048 tokens
    workspace_size_ = static_cast<size_t>(max_tokens_per_batch()) *
                      static_cast<size_t>(max_dim) * sizeof(float) * 10;
    cudaMalloc(&workspace_, workspace_size_);
    cudaMemset(workspace_, 0, workspace_size_);

    // 5. Allocate KV cache — per-layer pools
    auto kvc = kv_cache_config();
    kv_cache_num_blocks_ = kvc.max_blocks;
    kv_block_elems_ = static_cast<size_t>(config_.num_kv_heads) *
                      kvc.block_size * head_dim();
    // Total: num_layers * num_blocks * block_elems
    size_t total_cache_elems = static_cast<size_t>(config_.num_layers) *
                               kv_cache_num_blocks_ * kv_block_elems_;
    size_t cache_bytes = total_cache_elems * sizeof(float);
    cudaMalloc(&k_cache_, cache_bytes);
    cudaMalloc(&v_cache_, cache_bytes);
    cudaMemset(k_cache_, 0, cache_bytes);
    cudaMemset(v_cache_, 0, cache_bytes);

    loaded_ = true;
    std::cout << "LlamaModel: loaded " << config_.num_layers << " layers, "
              << "hidden=" << config_.hidden_size
              << ", heads=" << config_.num_attention_heads
              << ", kv_heads=" << config_.num_kv_heads
              << ", vocab=" << config_.vocab_size << std::endl;
}

// ---------------------------------------------------------------------------
// forward_impl
// ---------------------------------------------------------------------------

void LlamaModel::forward_impl(const std::vector<RequestContext>& requests,
                                bool is_prefill, ForwardResult& result,
                                cudaStream_t stream) {
    if (!loaded_) {
        throw std::runtime_error("LlamaModel: weights not loaded");
    }

    // 1. Collect all tokens, positions, and block info
    int total_tokens = 0;
    for (const auto& req : requests) {
        total_tokens += is_prefill
                            ? static_cast<int>(req.token_ids.size())
                            : 1;
    }

    // Build flattened arrays on CPU
    std::vector<int32_t> all_token_ids;
    std::vector<int> all_positions;
    std::vector<int> all_block_indices;   // per-token block index
    std::vector<int> all_block_offsets;   // per-token offset within block
    std::vector<int> seq_lens;            // per-request total sequence length
    std::vector<int> tokens_per_request;

    auto kvc = kv_cache_config();

    for (const auto& req : requests) {
        int num_tok = is_prefill
                          ? static_cast<int>(req.token_ids.size())
                          : 1;
        tokens_per_request.push_back(num_tok);
        seq_lens.push_back(req.seq_len);

        for (int t = 0; t < num_tok; ++t) {
            if (is_prefill) {
                all_token_ids.push_back(req.token_ids[t]);
                int pos = req.prefill_start_pos + t;
                all_positions.push_back(pos);
                // Determine block for this position
                int abs_pos = pos;
                int blk_idx_in_seq = abs_pos / kvc.block_size;
                int blk_offset = abs_pos % kvc.block_size;
                if (blk_idx_in_seq < static_cast<int>(req.block_table.size())) {
                    all_block_indices.push_back(
                        req.block_table[blk_idx_in_seq]);
                } else {
                    all_block_indices.push_back(0);  // fallback
                }
                all_block_offsets.push_back(blk_offset);
            } else {
                // Decode: single new token
                all_token_ids.push_back(
                    req.token_ids.back());
                int pos = req.seq_len - 1;  // current position
                all_positions.push_back(pos);
                int blk_idx_in_seq = pos / kvc.block_size;
                int blk_offset = pos % kvc.block_size;
                if (blk_idx_in_seq < static_cast<int>(req.block_table.size())) {
                    all_block_indices.push_back(
                        req.block_table[blk_idx_in_seq]);
                } else {
                    all_block_indices.push_back(0);
                }
                all_block_offsets.push_back(blk_offset);
            }
        }
    }

    int H = config_.hidden_size;
    int num_heads = config_.num_attention_heads;
    int num_kv_h = config_.num_kv_heads;
    int hdim = head_dim();
    int inter = config_.intermediate_size;
    int V = config_.vocab_size;

    // 2. Upload positions, block info to GPU
    int* d_positions = nullptr;
    int* d_block_indices = nullptr;
    int* d_block_offsets = nullptr;
    int32_t* d_token_ids = nullptr;
    int* d_seq_lens = nullptr;

    cudaMalloc(&d_token_ids,
               total_tokens * sizeof(int32_t));
    cudaMalloc(&d_positions, total_tokens * sizeof(int));
    cudaMalloc(&d_block_indices, total_tokens * sizeof(int));
    cudaMalloc(&d_block_offsets, total_tokens * sizeof(int));
    cudaMalloc(&d_seq_lens, requests.size() * sizeof(int));

    cudaMemcpyAsync(d_token_ids, all_token_ids.data(),
                    total_tokens * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_positions, all_positions.data(),
                    total_tokens * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_block_indices, all_block_indices.data(),
                    total_tokens * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_block_offsets, all_block_offsets.data(),
                    total_tokens * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_seq_lens, seq_lens.data(),
                    requests.size() * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    // 3. Workspace layout (all float)
    // Divide workspace into named regions
    float* ws = workspace_;
    size_t ws_offset = 0;
    auto alloc_ws = [&](size_t elems) -> float* {
        float* p = ws + ws_offset;
        ws_offset += elems;
        return p;
    };

    float* hidden = alloc_ws(total_tokens * H);
    float* normed = alloc_ws(total_tokens * H);
    float* q_buf = alloc_ws(total_tokens * num_heads * hdim);
    float* k_buf = alloc_ws(total_tokens * num_kv_h * hdim);
    float* v_buf = alloc_ws(total_tokens * num_kv_h * hdim);
    float* attn_out = alloc_ws(total_tokens * H);
    float* gate_buf = alloc_ws(total_tokens * inter);
    float* up_buf = alloc_ws(total_tokens * inter);
    float* down_buf = alloc_ws(total_tokens * inter);
    float* logits_buf = alloc_ws(total_tokens * V);

    // 4. Embedding lookup
    launch_embedding_lookup(d_token_ids, embed_tokens_, hidden,
                            total_tokens, H, stream);

    // 5. Transformer layers
    GemmDescriptor gemm_desc{};
    gemm_desc.input_dtype = DataType::kFloat32;
    gemm_desc.output_dtype = DataType::kFloat32;

    // Build block_table for paged attention (per-request, padded)
    // Use the actual max block table size, not config_.max_seq_len
    int max_blocks_per_seq = 0;
    for (const auto& req : requests) {
        max_blocks_per_seq = std::max(max_blocks_per_seq,
                                       static_cast<int>(req.block_table.size()));
    }
    if (max_blocks_per_seq == 0) max_blocks_per_seq = 1;

    std::vector<int> flat_block_table(
        requests.size() * max_blocks_per_seq, 0);
    for (size_t r = 0; r < requests.size(); ++r) {
        for (size_t b = 0; b < requests[r].block_table.size() &&
                           b < static_cast<size_t>(max_blocks_per_seq);
             ++b) {
            flat_block_table[r * max_blocks_per_seq + b] =
                requests[r].block_table[b];
        }
    }
    int* d_attn_block_table = nullptr;
    cudaMalloc(&d_attn_block_table,
               flat_block_table.size() * sizeof(int));
    cudaMemcpyAsync(d_attn_block_table, flat_block_table.data(),
                    flat_block_table.size() * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    // Pre-allocate per-request prefill block tables and seq_lens.
    // These are identical across all layers, so we upload them once here
    // and reuse them every layer, avoiding use-after-free races.
    struct PrefillRequestBuffers {
        int* d_bt = nullptr;       // [num_tok * req_max_blocks]
        int* d_sl = nullptr;       // [num_tok]
        int  num_tok = 0;
        int  req_max_blocks = 0;
    };
    std::vector<PrefillRequestBuffers> prefill_bufs;
    if (is_prefill) {
        prefill_bufs.resize(requests.size());
        for (size_t r = 0; r < requests.size(); ++r) {
            int num_tok = tokens_per_request[r];
            int req_max_blocks = std::max(
                1, static_cast<int>(requests[r].block_table.size()));

            std::vector<int> req_bt(num_tok * req_max_blocks, 0);
            for (int t = 0; t < num_tok; ++t) {
                for (size_t b = 0; b < requests[r].block_table.size(); ++b) {
                    req_bt[t * req_max_blocks + b] =
                        requests[r].block_table[b];
                }
            }
            std::vector<int> tok_seq_lens(num_tok);
            for (int t = 0; t < num_tok; ++t) {
                tok_seq_lens[t] = requests[r].prefill_start_pos + t + 1;
            }

            cudaMalloc(&prefill_bufs[r].d_bt,
                       req_bt.size() * sizeof(int));
            cudaMalloc(&prefill_bufs[r].d_sl,
                       num_tok * sizeof(int));
            cudaMemcpyAsync(prefill_bufs[r].d_bt, req_bt.data(),
                            req_bt.size() * sizeof(int),
                            cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(prefill_bufs[r].d_sl, tok_seq_lens.data(),
                            num_tok * sizeof(int),
                            cudaMemcpyHostToDevice, stream);
            prefill_bufs[r].num_tok = num_tok;
            prefill_bufs[r].req_max_blocks = req_max_blocks;
        }
    }

    for (int layer = 0; layer < config_.num_layers; ++layer) {
        const auto& lw = layers_[layer];

        // a. RMSNorm(hidden) → normed
        kernels_->rms_norm(hidden, lw.input_norm, normed,
                           total_tokens, H, 1e-5f,
                           DataType::kFloat32, stream);

        // b. Q = normed @ q_proj.T → [total_tokens, num_heads * head_dim]
        //    GEMM: C[M,N] = A[M,K] * B[K,N]
        //    A = normed [total_tokens, H], B = q_proj [num_heads*hdim, H]^T
        //    But weights stored as [out_features, in_features] (row-major),
        //    so q_proj is [num_heads*hdim, H].
        //    C = A @ B^T means: C[M, N] where M=total_tokens, N=num_heads*hdim, K=H
        //    With row-major gemm: C = A * B^T. The CublasProvider gemm does
        //    C = A * B (no transpose). We need to pass B^T = weight^T...
        //    Actually the cublas_provider gemm computes C = A * B where
        //    A[M,K], B[K,N], C[M,N]. The weight is [N, K] = [out, in].
        //    So we need: C = normed @ weight^T, which is standard linear.
        //    But the gemm interface expects B[K,N]. Weight is [N,K].
        //    We have two options: transpose on load, or adjust gemm.
        //    The simplest: treat as C[M,K] = A[M,K] * B^T[K,N]
        //    But the interface has no transB support for us to use directly...
        //    Let's think differently. cublas_provider does: C = A * B with
        //    A[M,K], B[K,N]. We want output = input @ weight^T.
        //    input is [M, K_in], weight is [K_out, K_in].
        //    So we set: A = input [M, K_in], B = weight^T [K_in, K_out].
        //    But we don't have weight^T stored. However, the cublas trick
        //    computes C^T = B^T * A^T in column-major. Let's just call gemm
        //    with M=total_tokens, K=H, N=out_dim and pass weight as B.
        //    The cublasSgemm call in cublas_provider.cpp is:
        //      cublasSgemm(handle_, N, N, desc.N, desc.M, desc.K,
        //                  &alpha, B, desc.N, A, desc.K, &beta, C, desc.N)
        //    This computes (in column-major view): C_col = B_col * A_col
        //    Since row-major A[M,K] == column-major A^T[K,M], this becomes:
        //      C^T[N,M] = B^T[N,K] * A^T[K,M] → C[M,N] = A[M,K] * B[K,N]
        //    So we need B to be [K, N] in row-major = weight^T.
        //    Since weight is [N, K], and we want B = weight^T [K, N]...
        //    Actually: let's flip it. We want: out = input * weight^T.
        //    This is the same as: out^T = weight * input^T.
        //    In cublas column-major: out^T[N,M] = weight[N,K] * input^T[K,M]
        //    cublasSgemm(N, N, N, M, K, alpha, weight, N, input, K, beta, out, N)
        //    But the GemmDescriptor expects M, N, K corresponding to:
        //    C[M,N] = A[M,K] * B[K,N] with A passed first.
        //    So if I set desc.M = total_tokens, desc.N = out_dim, desc.K = H,
        //    and pass A = input, B = weight, the cublas code will do:
        //    cublasSgemm(N, N, out_dim, total_tokens, H, alpha,
        //                weight, out_dim, input, H, beta, out, out_dim)
        //    Which computes: out_col = weight_col * input_col
        //    = weight^T[out_dim, H] * input^T[H, M] (row→col interpretation)
        //    Actually in pure col-major: out[out_dim, M] = weight[out_dim, H] * input[H, M]
        //    Reading as row-major: out^T[M, out_dim] = (weight * input)^T = ...
        //    Let me just verify: the cublas call does:
        //    C = B^T_rowmajor * A^T_rowmajor (as col major)
        //    = B[K,N]^T_as_col * A[M,K]^T_as_col = [N,K]*[K,M] = [N,M]
        //    which in row-major is [M,N]. But B here is weight[N_out, K_in]
        //    and A is input[M, K_in].
        //    C_col[N_out, M] = weight_col[N_out, K_in] * input_col[K_in, M]
        //    In row-major C[M, N_out] = ... this is exactly input @ weight^T!
        //    So: desc = {M=total_tokens, N=out_dim, K=H}, A=input, B=weight.
        //    The weight is passed AS-IS (no transpose needed).

        gemm_desc.M = total_tokens;
        gemm_desc.K = H;
        gemm_desc.N = num_heads * hdim;
        kernels_->gemm(gemm_desc, normed, lw.q_proj, q_buf, stream);

        gemm_desc.N = num_kv_h * hdim;
        kernels_->gemm(gemm_desc, normed, lw.k_proj, k_buf, stream);
        kernels_->gemm(gemm_desc, normed, lw.v_proj, v_buf, stream);

        // c. RoPE on Q and K
        kernels_->rope(q_buf, k_buf,
                       total_tokens, 1,  // batch=total_tokens, seq_len=1
                       num_heads, num_kv_h, hdim,
                       d_positions, rope_theta_,
                       DataType::kFloat32, stream);

        // d. KV cache scatter — write to THIS layer's cache
        float* k_cache_l = k_cache_layer(layer);
        float* v_cache_l = v_cache_layer(layer);

        launch_kv_cache_scatter(k_buf, v_buf, k_cache_l, v_cache_l,
                                d_block_indices, d_block_offsets,
                                total_tokens, num_kv_h, hdim,
                                kvc.block_size, stream);

        // e. Attention — read from THIS layer's cache
        if (!is_prefill) {
            // Decode: paged attention (one query per request)
            launch_paged_attention(
                q_buf, k_cache_l, v_cache_l, attn_out,
                d_attn_block_table, max_blocks_per_seq,
                d_seq_lens,
                static_cast<int>(requests.size()),
                num_heads, num_kv_h, hdim,
                kvc.block_size, stream);
        } else {
            // Prefill: each token attends to its causal prefix.
            // Use pre-allocated block tables and seq_lens (same every layer).
            int tok_offset = 0;
            for (size_t r = 0; r < requests.size(); ++r) {
                const auto& pb = prefill_bufs[r];
                launch_paged_attention(
                    q_buf + tok_offset * num_heads * hdim,
                    k_cache_l, v_cache_l,
                    attn_out + tok_offset * H,
                    pb.d_bt, pb.req_max_blocks,
                    pb.d_sl,
                    pb.num_tok, num_heads, num_kv_h, hdim,
                    kvc.block_size, stream);
                tok_offset += pb.num_tok;
            }
        }

        // f. O projection: proj = attn_out @ o_proj.T → [total_tokens, H]
        gemm_desc.M = total_tokens;
        gemm_desc.K = H;
        gemm_desc.N = H;
        // We'll store result in normed (reuse buffer)
        kernels_->gemm(gemm_desc, attn_out, lw.o_proj, normed, stream);

        // g. Residual: hidden = hidden + normed
        launch_residual_add(hidden, normed, hidden,
                            total_tokens * H, stream);

        // h. RMSNorm(hidden) → normed
        kernels_->rms_norm(hidden, lw.post_attn_norm, normed,
                           total_tokens, H, 1e-5f,
                           DataType::kFloat32, stream);

        // i. gate = normed @ gate_proj.T, up = normed @ up_proj.T
        gemm_desc.M = total_tokens;
        gemm_desc.K = H;
        gemm_desc.N = inter;
        kernels_->gemm(gemm_desc, normed, lw.gate_proj, gate_buf, stream);
        kernels_->gemm(gemm_desc, normed, lw.up_proj, up_buf, stream);

        // j. SwiGLU: fused = silu(gate) * up
        kernels_->silu_mul(gate_buf, up_buf, gate_buf,
                           total_tokens * inter,
                           DataType::kFloat32, stream);

        // k. down = fused @ down_proj.T → [total_tokens, H]
        gemm_desc.M = total_tokens;
        gemm_desc.K = inter;
        gemm_desc.N = H;
        kernels_->gemm(gemm_desc, gate_buf, lw.down_proj, normed, stream);

        // l. Residual: hidden = hidden + normed
        launch_residual_add(hidden, normed, hidden,
                            total_tokens * H, stream);

    }

    // 6. Final RMSNorm
    kernels_->rms_norm(hidden, final_norm_weight_, normed,
                       total_tokens, H, 1e-5f,
                       DataType::kFloat32, stream);

    // 7. LM head: logits = normed @ lm_head.T → [total_tokens, vocab_size]
    gemm_desc.M = total_tokens;
    gemm_desc.K = H;
    gemm_desc.N = V;
    kernels_->gemm(gemm_desc, normed, lm_head_, logits_buf, stream);

    // 8. Extract logits and copy to CPU
    int num_output_tokens = 0;
    if (is_prefill) {
        // Extract last token logits per request
        num_output_tokens = static_cast<int>(requests.size());
    } else {
        // One token per request
        num_output_tokens = static_cast<int>(requests.size());
    }

    result.vocab_size = V;
    result.logits.resize(num_output_tokens * V);

    if (is_prefill) {
        // Copy last token's logits per request
        int tok_offset = 0;
        for (size_t r = 0; r < requests.size(); ++r) {
            int num_tok = tokens_per_request[r];
            int last_tok_idx = tok_offset + num_tok - 1;
            cudaMemcpyAsync(result.logits.data() + r * V,
                            logits_buf + static_cast<int64_t>(last_tok_idx) * V,
                            V * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
            tok_offset += num_tok;
        }
    } else {
        cudaMemcpyAsync(result.logits.data(), logits_buf,
                        static_cast<size_t>(num_output_tokens) * V *
                            sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
    }

    cudaStreamSynchronize(stream);

    // DEBUG: print top-5 logits for multi-token prefill
    if (is_prefill && total_tokens > 1) {
        const float* logits = result.logits.data();
        std::vector<int> idx(V);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                          [&](int a, int b){ return logits[a] > logits[b]; });
        std::cerr << "DEBUG top-5 logits (multi-tok prefill, " << total_tokens << " tokens): ";
        for (int i = 0; i < 5; i++)
            std::cerr << "tok" << idx[i] << "=" << logits[idx[i]] << " ";
        std::cerr << "\n";
    }

    // Cleanup temp GPU allocations
    cudaFree(d_token_ids);
    cudaFree(d_positions);
    cudaFree(d_block_indices);
    cudaFree(d_block_offsets);
    cudaFree(d_seq_lens);
    cudaFree(d_attn_block_table);
    for (auto& pb : prefill_bufs) {
        cudaFree(pb.d_bt);
        cudaFree(pb.d_sl);
    }
}

// ---------------------------------------------------------------------------
// prefill / decode
// ---------------------------------------------------------------------------

ForwardResult LlamaModel::prefill(const std::vector<RequestContext>& requests,
                                   cudaStream_t stream) {
    ForwardResult result;
    forward_impl(requests, /*is_prefill=*/true, result, stream);
    return result;
}

ForwardResult LlamaModel::decode(const std::vector<RequestContext>& requests,
                                  cudaStream_t stream) {
    ForwardResult result;
    forward_impl(requests, /*is_prefill=*/false, result, stream);
    return result;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<ModelModule> create_llama_model(KernelProvider* kernels) {
    return std::make_unique<LlamaModel>(kernels);
}
