// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION &
// AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0

#include "core/scheduler.h"

#include <algorithm>
#include <cmath>

Scheduler::Scheduler(SchedulerConfig config, BlockPool& block_pool,
                     PrefixCache& prefix_cache)
    : config_(std::move(config)),
      block_pool_(block_pool),
      prefix_cache_(prefix_cache) {}

void Scheduler::enqueue(GenerationRequest req) {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    pending_.push_back(std::move(req));
}

bool Scheduler::allocate_blocks_for_request(ActiveRequest& req) {
    int total_tokens = static_cast<int>(req.all_tokens.size());
    int blocks_needed =
        (total_tokens + config_.kv_block_size - 1) / config_.kv_block_size;
    int blocks_have = static_cast<int>(req.block_table.size());
    int blocks_to_alloc = blocks_needed - blocks_have;

    for (int i = 0; i < blocks_to_alloc; ++i) {
        auto blk = block_pool_.allocate();
        if (!blk.has_value()) {
            // Can't allocate — free what we just allocated this round
            // (blocks beyond blocks_have)
            while (static_cast<int>(req.block_table.size()) > blocks_have) {
                block_pool_.free(req.block_table.back());
                req.block_table.pop_back();
            }
            return false;
        }
        req.block_table.push_back(blk.value());
    }
    return true;
}

void Scheduler::release_blocks(ActiveRequest& req) {
    // Use prompt tokens as the cache key (that's what future requests lookup)
    std::vector<int> prompt_int(req.all_tokens.begin(),
                                req.all_tokens.begin() + req.prompt_len);
    // Only donate the blocks covering the prompt tokens
    int prompt_blocks =
        (req.prompt_len + config_.kv_block_size - 1) / config_.kv_block_size;
    std::vector<int> cache_blocks(
        req.block_table.begin(),
        req.block_table.begin() +
            std::min(prompt_blocks,
                     static_cast<int>(req.block_table.size())));

    // Donate prompt blocks to prefix cache
    prefix_cache_.insert(prompt_int, cache_blocks);

    // Free any remaining blocks (for generated tokens beyond prompt)
    for (int i = prompt_blocks;
         i < static_cast<int>(req.block_table.size()); ++i) {
        block_pool_.free(req.block_table[i]);
    }
    req.block_table.clear();
}

ScheduledBatch Scheduler::form_batch() {
    // Move pending requests into local queue
    std::deque<GenerationRequest> local_pending;
    {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        local_pending.swap(pending_);
    }

    // Activate new requests
    std::deque<GenerationRequest> deferred;
    for (auto& gen_req : local_pending) {
        ActiveRequest areq;
        areq.request_id = gen_req.request_id;
        areq.all_tokens.assign(gen_req.prompt_tokens.begin(),
                               gen_req.prompt_tokens.end());
        areq.prompt_len = static_cast<int>(areq.all_tokens.size());
        areq.prefill_pos = 0;
        areq.prefill_done = false;
        areq.generated_count = 0;
        areq.max_new_tokens = gen_req.max_new_tokens;
        areq.sampling_params = gen_req.sampling_params;

        // Check prefix cache
        std::vector<int> tokens_int(areq.all_tokens.begin(),
                                    areq.all_tokens.end());
        auto cache_result = prefix_cache_.lookup(tokens_int);
        if (cache_result.matched_tokens > 0) {
            areq.block_table = cache_result.blocks;
            areq.prefill_pos = cache_result.matched_tokens;
            // Add ref so the cached blocks won't be evicted
            // Use the matched prefix as key
            std::vector<int> prefix_key(tokens_int.begin(),
                                        tokens_int.begin() +
                                            cache_result.matched_tokens);
            prefix_cache_.add_ref(prefix_key);
        }

        // Allocate remaining blocks
        if (!allocate_blocks_for_request(areq)) {
            // Put back in pending
            deferred.push_back(std::move(gen_req));
            // Release any cached block refs we took
            if (cache_result.matched_tokens > 0) {
                std::vector<int> prefix_key(
                    tokens_int.begin(),
                    tokens_int.begin() + cache_result.matched_tokens);
                prefix_cache_.release(prefix_key);
            }
            continue;
        }

        active_[areq.request_id] = std::move(areq);
    }

    // Put deferred requests back
    if (!deferred.empty()) {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        for (auto it = deferred.rbegin(); it != deferred.rend(); ++it) {
            pending_.push_front(std::move(*it));
        }
    }

    // Build the scheduled batch
    ScheduledBatch batch;
    int tokens_used = 0;
    int batch_size = 0;

    // Schedule prefill requests first
    for (auto& [id, areq] : active_) {
        if (areq.prefill_done) continue;
        if (batch_size >= config_.max_batch_size) break;
        if (tokens_used >= config_.max_tokens_per_batch) break;

        int remaining_prefill =
            static_cast<int>(areq.all_tokens.size()) - areq.prefill_pos;
        int budget = config_.max_tokens_per_batch - tokens_used;
        if (remaining_prefill > 0 && budget <= 0) continue;
        int chunk_len = std::min(remaining_prefill, budget);
        if (chunk_len < 0) chunk_len = 0;
        // chunk_len == 0 is valid for fully-cached prefill requests

        RequestContext ctx;
        ctx.request_id = areq.request_id;
        ctx.token_ids.assign(
            areq.all_tokens.begin() + areq.prefill_pos,
            areq.all_tokens.begin() + areq.prefill_pos + chunk_len);
        ctx.seq_len = areq.prefill_pos + chunk_len;
        ctx.prefill_start_pos = areq.prefill_pos;
        ctx.prefill_chunk_len = chunk_len;
        ctx.block_table = areq.block_table;
        ctx.max_new_tokens = areq.max_new_tokens;

        batch.prefill_requests.push_back(std::move(ctx));
        tokens_used += chunk_len;
        batch_size++;
    }

    // Schedule decode requests
    for (auto& [id, areq] : active_) {
        if (!areq.prefill_done) continue;
        if (batch_size >= config_.max_batch_size) break;
        if (tokens_used >= config_.max_tokens_per_batch) break;

        RequestContext ctx;
        ctx.request_id = areq.request_id;
        // For decode, token_ids is the last token
        if (!areq.all_tokens.empty()) {
            ctx.token_ids = {areq.all_tokens.back()};
        }
        ctx.seq_len = static_cast<int>(areq.all_tokens.size());
        ctx.prefill_start_pos = 0;
        ctx.prefill_chunk_len = 0;
        ctx.block_table = areq.block_table;
        ctx.max_new_tokens = areq.max_new_tokens;

        batch.decode_requests.push_back(std::move(ctx));
        tokens_used += 1;  // decode uses 1 token
        batch_size++;
    }

    return batch;
}

void Scheduler::update_after_step(
    const ScheduledBatch& batch,
    const std::vector<std::pair<int64_t, int32_t>>& sampled_tokens) {
    // Build a map from request_id to sampled token for quick lookup
    std::unordered_map<int64_t, int32_t> sampled_map;
    for (const auto& [rid, tok] : sampled_tokens) {
        sampled_map[rid] = tok;
    }

    // Process prefill requests: advance prefill_pos
    for (const auto& ctx : batch.prefill_requests) {
        auto it = active_.find(ctx.request_id);
        if (it == active_.end()) continue;
        auto& areq = it->second;

        areq.prefill_pos += ctx.prefill_chunk_len;
        if (areq.prefill_pos >=
            static_cast<int>(areq.all_tokens.size())) {
            areq.prefill_done = true;
        }
    }

    // Process sampled tokens
    std::vector<int64_t> to_complete;
    for (const auto& [rid, tok] : sampled_tokens) {
        auto it = active_.find(rid);
        if (it == active_.end()) continue;
        auto& areq = it->second;

        areq.all_tokens.push_back(static_cast<int32_t>(tok));
        areq.generated_count++;

        // Check if we need more blocks for the new token
        int total_tokens = static_cast<int>(areq.all_tokens.size());
        int blocks_needed =
            (total_tokens + config_.kv_block_size - 1) / config_.kv_block_size;
        while (static_cast<int>(areq.block_table.size()) < blocks_needed) {
            auto blk = block_pool_.allocate();
            if (blk.has_value()) {
                areq.block_table.push_back(blk.value());
            } else {
                break;  // can't allocate more — will handle in next iteration
            }
        }

        // Check stop conditions
        bool should_stop = false;
        if (tok == config_.eos_token_id) should_stop = true;
        for (int stop_id : config_.stop_token_ids) {
            if (tok == stop_id) {
                should_stop = true;
                break;
            }
        }
        if (areq.generated_count >= areq.max_new_tokens) should_stop = true;

        if (should_stop) {
            to_complete.push_back(rid);
        }
    }

    // Complete requests
    for (int64_t rid : to_complete) {
        auto it = active_.find(rid);
        if (it == active_.end()) continue;
        auto& areq = it->second;

        CompletedRequest cr;
        cr.request_id = rid;
        // Output tokens = generated tokens (after prompt)
        cr.output_tokens.assign(
            areq.all_tokens.begin() +
                static_cast<int>(areq.all_tokens.size()) -
                areq.generated_count,
            areq.all_tokens.end());

        release_blocks(areq);
        completed_.push_back(std::move(cr));
        active_.erase(it);
    }
}

std::vector<CompletedRequest> Scheduler::get_completed() {
    std::vector<CompletedRequest> result;
    result.swap(completed_);
    return result;
}
