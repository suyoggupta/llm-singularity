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
