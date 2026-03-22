#pragma once
#include <cstdint>
#include <random>
#include <vector>

struct SamplingParams {
    float temperature = 1.0f;
    int top_k = 0;          // 0 = disabled
    float top_p = 1.0f;     // 1.0 = disabled
    float repetition_penalty = 1.0f;
    uint64_t seed = 0;
};

class Sampler {
public:
    Sampler();
    int32_t sample(const float* logits, int vocab_size,
                   const SamplingParams& params,
                   const std::vector<int32_t>& prev_tokens = {});
private:
    std::mt19937 rng_;
};
