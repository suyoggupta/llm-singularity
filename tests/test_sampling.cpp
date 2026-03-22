#include <gtest/gtest.h>
#include "core/sampling.h"
#include <numeric>
#include <cmath>

TEST(SamplingTest, GreedySampling) {
    std::vector<float> logits = {0.1f, 0.2f, 0.05f, 0.9f, 0.3f};
    SamplingParams params;
    params.temperature = 0.0f;  // greedy

    Sampler sampler;
    auto token = sampler.sample(logits.data(), logits.size(), params);
    EXPECT_EQ(token, 3);
}

TEST(SamplingTest, TemperatureScaling) {
    std::vector<float> logits = {1.0f, 2.0f, 3.0f};
    SamplingParams params;
    params.temperature = 0.001f;  // near-greedy

    Sampler sampler;
    auto token = sampler.sample(logits.data(), logits.size(), params);
    EXPECT_EQ(token, 2);
}

TEST(SamplingTest, TopKFiltering) {
    std::vector<float> logits = {0.1f, 5.0f, 0.2f, 4.0f, 0.3f};
    SamplingParams params;
    params.temperature = 0.001f;
    params.top_k = 2;

    Sampler sampler;
    auto token = sampler.sample(logits.data(), logits.size(), params);
    EXPECT_EQ(token, 1);
}

TEST(SamplingTest, RepetitionPenalty) {
    std::vector<float> logits = {1.0f, 1.0f, 5.0f, 1.0f};
    SamplingParams params;
    params.temperature = 0.001f;
    params.repetition_penalty = 100.0f;

    Sampler sampler;
    std::vector<int32_t> prev_tokens = {2};
    auto token = sampler.sample(logits.data(), logits.size(), params, prev_tokens);
    EXPECT_NE(token, 2);
}
