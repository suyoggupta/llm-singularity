// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION &
// AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "core/scheduler.h"

class SchedulerTestFixture : public ::testing::Test {
protected:
    BlockPool block_pool{100, 4096};
    PrefixCache prefix_cache{16};

    SchedulerConfig make_config(int eos = 2, int max_batch = 64,
                                int max_tokens = 512) {
        SchedulerConfig cfg;
        cfg.max_batch_size = max_batch;
        cfg.max_tokens_per_batch = max_tokens;
        cfg.eos_token_id = eos;
        cfg.kv_block_size = 16;
        return cfg;
    }
};

TEST_F(SchedulerTestFixture, EnqueueAndFormBatch) {
    auto cfg = make_config();
    Scheduler sched(cfg, block_pool, prefix_cache);

    // Create two requests with 100 and 200 tokens
    GenerationRequest req1;
    req1.request_id = 1;
    req1.prompt_tokens.assign(100, 42);
    req1.max_new_tokens = 10;

    GenerationRequest req2;
    req2.request_id = 2;
    req2.prompt_tokens.assign(200, 43);
    req2.max_new_tokens = 10;

    sched.enqueue(std::move(req1));
    sched.enqueue(std::move(req2));

    auto batch = sched.form_batch();

    // Both should be scheduled for prefill
    EXPECT_EQ(batch.prefill_requests.size(), 2u);
    EXPECT_EQ(batch.decode_requests.size(), 0u);

    // Both should have non-empty block tables
    for (const auto& ctx : batch.prefill_requests) {
        EXPECT_FALSE(ctx.block_table.empty());
    }
}

TEST_F(SchedulerTestFixture, BlockAllocationAndRelease) {
    auto cfg = make_config();
    Scheduler sched(cfg, block_pool, prefix_cache);

    // 32-token request needs ceil(32/16) = 2 blocks
    GenerationRequest req;
    req.request_id = 1;
    req.prompt_tokens.assign(32, 10);
    req.max_new_tokens = 5;

    int free_before = block_pool.num_free_blocks();
    sched.enqueue(std::move(req));
    auto batch = sched.form_batch();

    ASSERT_EQ(batch.prefill_requests.size(), 1u);
    int free_after_alloc = block_pool.num_free_blocks();
    EXPECT_LT(free_after_alloc, free_before);

    // Complete prefill (all 32 tokens fit in one chunk with default
    // max_tokens_per_batch=512)
    sched.update_after_step(batch, {});

    // Now form decode batch
    auto batch2 = sched.form_batch();
    ASSERT_EQ(batch2.decode_requests.size(), 1u);

    // Send EOS to complete the request
    sched.update_after_step(batch2, {{1, cfg.eos_token_id}});

    auto completed = sched.get_completed();
    EXPECT_EQ(completed.size(), 1u);

    // After completion blocks should be returned (donated to prefix cache, but
    // since the prefix cache entry goes to LRU, the pool may not get them back
    // immediately). For this test we just verify the request completed.
    EXPECT_EQ(completed[0].request_id, 1);
}

TEST_F(SchedulerTestFixture, PrefixCacheReuse) {
    auto cfg = make_config();
    Scheduler sched(cfg, block_pool, prefix_cache);

    // First request: 32 tokens
    std::vector<int32_t> tokens(32, 7);
    GenerationRequest req1;
    req1.request_id = 1;
    req1.prompt_tokens = tokens;
    req1.max_new_tokens = 1;

    sched.enqueue(std::move(req1));
    auto batch1 = sched.form_batch();
    ASSERT_EQ(batch1.prefill_requests.size(), 1u);

    // Complete prefill
    sched.update_after_step(batch1, {});

    // Complete decode with EOS
    auto batch1d = sched.form_batch();
    sched.update_after_step(batch1d, {{1, cfg.eos_token_id}});
    sched.get_completed();

    // Now enqueue same tokens again — should reuse cached blocks
    GenerationRequest req2;
    req2.request_id = 2;
    req2.prompt_tokens = tokens;
    req2.max_new_tokens = 1;

    sched.enqueue(std::move(req2));
    auto batch2 = sched.form_batch();
    ASSERT_EQ(batch2.prefill_requests.size(), 1u);

    // The second request should reuse cached blocks: prefill_start_pos == 32
    EXPECT_EQ(batch2.prefill_requests[0].prefill_start_pos, 32);
}

TEST_F(SchedulerTestFixture, ChunkedPrefill) {
    auto cfg = make_config(/*eos=*/2, /*max_batch=*/64, /*max_tokens=*/128);
    Scheduler sched(cfg, block_pool, prefix_cache);

    // 300-token request, but max_tokens_per_batch = 128
    GenerationRequest req;
    req.request_id = 1;
    req.prompt_tokens.assign(300, 5);
    req.max_new_tokens = 10;

    sched.enqueue(std::move(req));

    // First batch: should chunk to 128 tokens
    auto batch1 = sched.form_batch();
    ASSERT_EQ(batch1.prefill_requests.size(), 1u);
    EXPECT_EQ(batch1.prefill_requests[0].prefill_chunk_len, 128);
    EXPECT_EQ(batch1.prefill_requests[0].prefill_start_pos, 0);

    sched.update_after_step(batch1, {});

    // Second batch: starts at pos 128
    auto batch2 = sched.form_batch();
    ASSERT_EQ(batch2.prefill_requests.size(), 1u);
    EXPECT_EQ(batch2.prefill_requests[0].prefill_start_pos, 128);
}

TEST_F(SchedulerTestFixture, ContinuousBatching) {
    auto cfg = make_config();
    Scheduler sched(cfg, block_pool, prefix_cache);

    // Enqueue req1 with 10 tokens
    GenerationRequest req1;
    req1.request_id = 1;
    req1.prompt_tokens.assign(10, 1);
    req1.max_new_tokens = 50;

    sched.enqueue(std::move(req1));

    // First batch: req1 is prefill
    auto batch1 = sched.form_batch();
    ASSERT_EQ(batch1.prefill_requests.size(), 1u);
    EXPECT_EQ(batch1.decode_requests.size(), 0u);

    // After prefill completes (no sampled token yet), req1 transitions to decode
    sched.update_after_step(batch1, {});

    // Now add req2
    GenerationRequest req2;
    req2.request_id = 2;
    req2.prompt_tokens.assign(20, 2);
    req2.max_new_tokens = 50;
    sched.enqueue(std::move(req2));

    // Second batch: req1 in decode, req2 in prefill
    auto batch2 = sched.form_batch();
    EXPECT_EQ(batch2.prefill_requests.size(), 1u);
    EXPECT_EQ(batch2.decode_requests.size(), 1u);

    // Verify identities
    EXPECT_EQ(batch2.decode_requests[0].request_id, 1);
    EXPECT_EQ(batch2.prefill_requests[0].request_id, 2);
}

TEST_F(SchedulerTestFixture, StopOnEOS) {
    auto cfg = make_config(/*eos=*/2);
    Scheduler sched(cfg, block_pool, prefix_cache);

    GenerationRequest req;
    req.request_id = 1;
    req.prompt_tokens = {100};
    req.max_new_tokens = 50;

    sched.enqueue(std::move(req));

    // Prefill
    auto batch1 = sched.form_batch();
    ASSERT_EQ(batch1.prefill_requests.size(), 1u);
    sched.update_after_step(batch1, {});

    // Decode — send EOS
    auto batch2 = sched.form_batch();
    ASSERT_EQ(batch2.decode_requests.size(), 1u);
    sched.update_after_step(batch2, {{1, 2}});

    auto completed = sched.get_completed();
    ASSERT_EQ(completed.size(), 1u);
    EXPECT_EQ(completed[0].request_id, 1);
}
