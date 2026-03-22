#include <gtest/gtest.h>
#include "core/memory.h"

TEST(BlockPoolTest, AllocateAndFree) {
    // CPU-side block pool logic (no GPU needed for pool management)
    BlockPool pool(/*num_blocks=*/100, /*block_size_bytes=*/4096);
    EXPECT_EQ(pool.num_free_blocks(), 100);

    auto block = pool.allocate();
    ASSERT_TRUE(block.has_value());
    EXPECT_EQ(pool.num_free_blocks(), 99);

    pool.free(block.value());
    EXPECT_EQ(pool.num_free_blocks(), 100);
}

TEST(BlockPoolTest, ExhaustPool) {
    BlockPool pool(/*num_blocks=*/2, /*block_size_bytes=*/4096);
    auto b1 = pool.allocate();
    auto b2 = pool.allocate();
    auto b3 = pool.allocate();
    ASSERT_TRUE(b1.has_value());
    ASSERT_TRUE(b2.has_value());
    ASSERT_FALSE(b3.has_value());
}

TEST(PrefixCacheTest, InsertAndLookup) {
    PrefixCache cache;
    std::vector<int> tokens = {1, 2, 3, 4, 5};
    std::vector<int> blocks = {10, 11};

    cache.insert(tokens, blocks);

    // Full match
    auto result = cache.lookup({1, 2, 3, 4, 5});
    EXPECT_EQ(result.matched_tokens, 5);
    EXPECT_EQ(result.blocks, (std::vector<int>{10, 11}));

    // Prefix match
    auto partial = cache.lookup({1, 2, 3, 4, 5, 6, 7});
    EXPECT_EQ(partial.matched_tokens, 5);

    // No match
    auto none = cache.lookup({9, 8, 7});
    EXPECT_EQ(none.matched_tokens, 0);
}

TEST(PrefixCacheTest, SharedPrefixRefCounting) {
    PrefixCache cache;
    cache.insert({1, 2, 3}, {10, 11});
    cache.add_ref({1, 2, 3});

    EXPECT_EQ(cache.ref_count({1, 2, 3}), 2);
    cache.release({1, 2, 3});
    EXPECT_EQ(cache.ref_count({1, 2, 3}), 1);
}

TEST(PrefixCacheTest, LRUEviction) {
    PrefixCache cache;
    cache.insert({1, 2}, {10});
    cache.insert({3, 4}, {11});

    // Both unreferenced — evict returns LRU entry
    auto evicted = cache.evict_lru();
    ASSERT_TRUE(evicted.has_value());
    EXPECT_EQ(evicted->blocks, (std::vector<int>{10}));  // first inserted = LRU
}
