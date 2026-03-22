// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION &
// AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0

#pragma once
#include <deque>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "core/memory.h"
#include "core/sampling.h"
#include "core/types.h"
#include "models/model_interface.h"

struct GenerationRequest {
    int64_t request_id;
    std::vector<int32_t> prompt_tokens;
    int max_new_tokens;
    SamplingParams sampling_params;
};

struct SchedulerConfig {
    int max_batch_size = 64;
    int max_tokens_per_batch = 512;
    int eos_token_id = -1;
    std::vector<int> stop_token_ids;
    int kv_block_size = 16;
};

struct ScheduledBatch {
    std::vector<RequestContext> prefill_requests;
    std::vector<RequestContext> decode_requests;
};

struct CompletedRequest {
    int64_t request_id;
    std::vector<int32_t> output_tokens;
};

class Scheduler {
public:
    Scheduler(SchedulerConfig config, BlockPool& block_pool,
              PrefixCache& prefix_cache);

    void enqueue(GenerationRequest req);  // thread-safe
    ScheduledBatch form_batch();          // main loop only
    void update_after_step(
        const ScheduledBatch& batch,
        const std::vector<std::pair<int64_t, int32_t>>& sampled_tokens);
    std::vector<CompletedRequest> get_completed();

private:
    SchedulerConfig config_;
    BlockPool& block_pool_;
    PrefixCache& prefix_cache_;
    std::mutex pending_mutex_;
    std::deque<GenerationRequest> pending_;

    struct ActiveRequest {
        int64_t request_id;
        std::vector<int32_t> all_tokens;
        int prompt_len;  // length of original prompt (for prefix cache key)
        int prefill_pos;
        bool prefill_done;
        int generated_count;
        int max_new_tokens;
        std::vector<int> block_table;
        SamplingParams sampling_params;
    };
    std::unordered_map<int64_t, ActiveRequest> active_;
    std::vector<CompletedRequest> completed_;

    bool allocate_blocks_for_request(ActiveRequest& req);
    void release_blocks(ActiveRequest& req);
};
