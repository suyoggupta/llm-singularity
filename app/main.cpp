// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION &
// AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

#include "core/memory.h"
#include "core/sampling.h"
#include "core/scheduler.h"
#include "core/server.h"
#include "core/tokenizer.h"
#include "core/types.h"
#include "kernels/cublas_provider.h"
#include "models/model_interface.h"

// Model factory from the llama model library
extern std::unique_ptr<ModelModule> create_llama_model(KernelProvider* kernels);

// ---------------------------------------------------------------------------
// CLI argument parser
// ---------------------------------------------------------------------------

struct Args {
    std::string model_dir;
    std::string host = "0.0.0.0";
    int port = 8000;

    static void print_usage() {
        std::cerr << "Usage: llm-serve --model-dir <path> [--host <addr>] "
                     "[--port <port>]\n";
    }
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model-dir" && i + 1 < argc) {
            args.model_dir = argv[++i];
        } else if (arg == "--host" && i + 1 < argc) {
            args.host = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            args.port = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            Args::print_usage();
            std::exit(0);
        }
    }
    return args;
}

// ---------------------------------------------------------------------------
// Per-request tracking state shared between server callback and inference loop
// ---------------------------------------------------------------------------

struct InFlightRequest {
    std::function<void(const std::string& token, bool done)> stream_cb;
    std::mutex mtx;
    std::condition_variable cv;
    bool finished = false;
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    // 1. Parse CLI args
    Args args = parse_args(argc, argv);
    if (args.model_dir.empty()) {
        std::cerr << "Error: --model-dir is required\n";
        Args::print_usage();
        return 1;
    }

    std::cout << "[init] model-dir: " << args.model_dir << "\n";
    std::cout << "[init] host: " << args.host << ":" << args.port << "\n";

    // 2. Create kernel provider
    std::cout << "[init] Creating cuBLAS kernel provider...\n";
    auto kernels = std::make_unique<CublasProvider>();

    // 3. Create model and load weights
    std::cout << "[init] Creating LLaMA model...\n";
    auto model = create_llama_model(kernels.get());
    std::cout << "[init] Loading weights from " << args.model_dir << "...\n";
    model->load_weights(args.model_dir);

    ModelConfig model_cfg = model->config();
    KVCacheConfig kv_cfg = model->kv_cache_config();
    std::cout << "[init] Model loaded: vocab_size=" << model_cfg.vocab_size
              << " hidden_size=" << model_cfg.hidden_size
              << " layers=" << model_cfg.num_layers << "\n";

    // 4. Create tokenizer
    std::cout << "[init] Loading tokenizer...\n";
    Tokenizer tokenizer;
    tokenizer.load(args.model_dir);
    std::cout << "[init] Tokenizer loaded: vocab_size=" << tokenizer.vocab_size()
              << " eos=" << tokenizer.eos_token_id() << "\n";

    // 5. Create BlockPool for KV cache
    size_t block_size_bytes = static_cast<size_t>(kv_cfg.block_size)
        * model_cfg.num_kv_heads * (model_cfg.hidden_size / model_cfg.num_attention_heads)
        * dtype_size(kv_cfg.cache_dtype) * 2  // K + V
        * model_cfg.num_layers;
    std::cout << "[init] Creating BlockPool: " << kv_cfg.max_blocks
              << " blocks, " << block_size_bytes << " bytes/block\n";
    BlockPool block_pool(kv_cfg.max_blocks, block_size_bytes);

    // 6. Create PrefixCache
    PrefixCache prefix_cache(kv_cfg.block_size);

    // 7. Create Scheduler
    SchedulerConfig sched_cfg;
    sched_cfg.max_batch_size = model->max_batch_size();
    sched_cfg.max_tokens_per_batch = model->max_tokens_per_batch();
    sched_cfg.eos_token_id = model_cfg.eos_token_id;
    sched_cfg.stop_token_ids = model_cfg.stop_token_ids;
    sched_cfg.kv_block_size = kv_cfg.block_size;
    Scheduler scheduler(sched_cfg, block_pool, prefix_cache);

    // 8. Create Sampler
    Sampler sampler;

    // 9. Create CUDA stream for inference
    cudaStream_t cuda_stream;
    cudaStreamCreate(&cuda_stream);

    // 10. Request tracking
    std::mutex request_map_mutex;
    std::unordered_map<int64_t, std::shared_ptr<InFlightRequest>> request_map;
    std::atomic<int64_t> next_request_id{1};
    std::atomic<bool> running{true};

    // Helper: stream a token to the registered callback
    auto stream_token = [&](int64_t request_id, const std::string& text) {
        std::shared_ptr<InFlightRequest> inflight;
        {
            std::lock_guard<std::mutex> lk(request_map_mutex);
            auto it = request_map.find(request_id);
            if (it == request_map.end()) return;
            inflight = it->second;
        }
        if (inflight->stream_cb) {
            inflight->stream_cb(text, false);
        }
    };

    // Helper: notify request completion
    auto notify_done = [&](int64_t request_id) {
        std::shared_ptr<InFlightRequest> inflight;
        {
            std::lock_guard<std::mutex> lk(request_map_mutex);
            auto it = request_map.find(request_id);
            if (it == request_map.end()) return;
            inflight = it->second;
            request_map.erase(it);
        }
        if (inflight->stream_cb) {
            inflight->stream_cb("", true);
        }
        {
            std::lock_guard<std::mutex> lk(inflight->mtx);
            inflight->finished = true;
        }
        inflight->cv.notify_all();
    };

    // 11. Start inference loop on background thread
    std::thread inference_thread([&]() {
        while (running.load()) {
            ScheduledBatch batch = scheduler.form_batch();

            if (batch.prefill_requests.empty()
                && batch.decode_requests.empty()) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                continue;
            }

            // Process prefill requests
            if (!batch.prefill_requests.empty()) {
                ForwardResult result =
                    model->prefill(batch.prefill_requests, cuda_stream);
                std::vector<std::pair<int64_t, int32_t>> sampled;
                for (size_t i = 0; i < batch.prefill_requests.size(); i++) {
                    auto& req = batch.prefill_requests[i];
                    bool is_last_chunk =
                        (req.prefill_start_pos + req.prefill_chunk_len
                         >= req.seq_len);
                    if (is_last_chunk) {
                        const float* logits_ptr =
                            result.logits.data()
                            + static_cast<ptrdiff_t>(i) * result.vocab_size;
                        int32_t token = sampler.sample(
                            logits_ptr, result.vocab_size, SamplingParams{});
                        sampled.push_back({req.request_id, token});
                        stream_token(req.request_id,
                                     tokenizer.decode({token}));
                    }
                }
                scheduler.update_after_step(batch, sampled);
            }

            // Process decode requests
            if (!batch.decode_requests.empty()) {
                ForwardResult result =
                    model->decode(batch.decode_requests, cuda_stream);
                std::vector<std::pair<int64_t, int32_t>> sampled;
                for (size_t i = 0; i < batch.decode_requests.size(); i++) {
                    const float* logits_ptr =
                        result.logits.data()
                        + static_cast<ptrdiff_t>(i) * result.vocab_size;
                    int32_t token = sampler.sample(
                        logits_ptr, result.vocab_size, SamplingParams{});
                    sampled.push_back(
                        {batch.decode_requests[i].request_id, token});
                    stream_token(batch.decode_requests[i].request_id,
                                 tokenizer.decode({token}));
                }
                scheduler.update_after_step(batch, sampled);
            }

            // Notify completed requests
            for (auto& completed : scheduler.get_completed()) {
                notify_done(completed.request_id);
            }
        }
    });

    // 12. Create and start HTTP server
    std::string model_name = "llama";
    if (model_cfg.extra.count("model_type")) {
        model_name = model_cfg.extra.at("model_type");
    }

    GenerateCallback generate_cb = [&](const std::string& prompt_text,
                                       const SamplingParams& params,
                                       int max_tokens, bool stream,
                                       std::function<void(
                                           const std::string& token,
                                           bool done)> stream_cb) {
        // Tokenize prompt
        std::vector<int32_t> tokens = tokenizer.encode(prompt_text);

        // Create generation request
        int64_t req_id = next_request_id.fetch_add(1);
        GenerationRequest gen_req;
        gen_req.request_id = req_id;
        gen_req.prompt_tokens = tokens;
        gen_req.max_new_tokens = max_tokens;
        gen_req.sampling_params = params;

        // Register in-flight tracking
        auto inflight = std::make_shared<InFlightRequest>();
        inflight->stream_cb = stream_cb;
        {
            std::lock_guard<std::mutex> lk(request_map_mutex);
            request_map[req_id] = inflight;
        }

        // Enqueue to scheduler (thread-safe)
        scheduler.enqueue(std::move(gen_req));

        // Block until request completes
        {
            std::unique_lock<std::mutex> lk(inflight->mtx);
            inflight->cv.wait(lk, [&] { return inflight->finished; });
        }
    };

    std::cout << "[init] Starting server on " << args.host << ":"
              << args.port << "...\n";
    Server server(args.host, args.port, model_name, generate_cb);

    // Install signal handler for clean shutdown
    // (simplified: just use atexit-style cleanup)
    std::cout << "[init] Ready. Listening for requests.\n";
    server.start();  // blocking

    // Cleanup on server exit
    running.store(false);
    if (inference_thread.joinable()) {
        inference_thread.join();
    }
    cudaStreamDestroy(cuda_stream);

    std::cout << "[shutdown] Done.\n";
    return 0;
}
