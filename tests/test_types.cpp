#include <gtest/gtest.h>
#include "core/types.h"

TEST(TypesTest, DataTypeSizes) {
    EXPECT_EQ(dtype_size(DataType::kFloat16), 2);
    EXPECT_EQ(dtype_size(DataType::kBFloat16), 2);
    EXPECT_EQ(dtype_size(DataType::kFloat32), 4);
}

TEST(TypesTest, DataTypeNames) {
    EXPECT_EQ(dtype_name(DataType::kFloat16), "float16");
    EXPECT_EQ(dtype_name(DataType::kBFloat16), "bfloat16");
    EXPECT_EQ(dtype_name(DataType::kFloat32), "float32");
}
