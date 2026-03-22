#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include "core/safetensors.h"

TEST(SafetensorsTest, ParseSmallFile) {
    const char* model_path = std::getenv("TEST_MODEL_PATH");
    if (!model_path) GTEST_SKIP() << "TEST_MODEL_PATH not set";

    SafetensorsFile sf;
    sf.open(std::string(model_path) + "/model.safetensors");

    auto names = sf.tensor_names();
    EXPECT_GT(names.size(), 0u);

    auto info = sf.tensor_info(names[0]);
    EXPECT_GT(info.num_elements(), 0u);
}

TEST(SafetensorsTest, ReadSyntheticFile) {
    // Create a minimal safetensors file:
    // Format: 8 bytes (header_size as uint64 LE) + JSON header + raw tensor data
    // The JSON header maps tensor names to {dtype, shape, data_offsets}

    std::string path = "/tmp/test_synthetic.safetensors";

    // Create a file with one tensor: "weight" of shape [2,3] with F32 dtype
    // 6 floats = 24 bytes of data
    nlohmann::json header;
    header["weight"] = {
        {"dtype", "F32"},
        {"shape", {2, 3}},
        {"data_offsets", {0, 24}}
    };
    std::string header_str = header.dump();
    uint64_t header_size = header_str.size();

    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(&header_size), 8);
    f.write(header_str.data(), header_str.size());
    // Write 6 floats: 1.0, 2.0, 3.0, 4.0, 5.0, 6.0
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    f.write(reinterpret_cast<const char*>(data.data()), 24);
    f.close();

    SafetensorsFile sf;
    sf.open(path);

    auto names = sf.tensor_names();
    ASSERT_EQ(names.size(), 1u);
    EXPECT_EQ(names[0], "weight");

    auto info = sf.tensor_info("weight");
    EXPECT_EQ(info.dtype, DataType::kFloat32);
    EXPECT_EQ(info.shape, (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(info.num_elements(), 6);

    const float* ptr = static_cast<const float*>(sf.tensor_data("weight"));
    EXPECT_NEAR(ptr[0], 1.0f, 1e-6);
    EXPECT_NEAR(ptr[5], 6.0f, 1e-6);

    std::remove(path.c_str());
}
