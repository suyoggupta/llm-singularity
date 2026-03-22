#include "core/sampling.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>

Sampler::Sampler() : rng_(std::mt19937{std::random_device{}()}) {}

int32_t Sampler::sample(const float* logits, int vocab_size,
                        const SamplingParams& params,
                        const std::vector<int32_t>& prev_tokens) {
    assert(vocab_size > 0);

    // Step 1: Copy logits into a mutable working buffer.
    std::vector<float> work(logits, logits + vocab_size);

    // Step 2: Apply repetition penalty.
    // For each previously generated token, penalise its logit:
    //   positive logit -> divide by penalty  (reduces probability)
    //   negative logit -> multiply by penalty (further reduces probability)
    if (params.repetition_penalty != 1.0f) {
        for (int32_t tok : prev_tokens) {
            if (tok >= 0 && tok < vocab_size) {
                if (work[tok] > 0.0f)
                    work[tok] /= params.repetition_penalty;
                else
                    work[tok] *= params.repetition_penalty;
            }
        }
    }

    // Step 3: Greedy (temperature == 0) -> return argmax immediately.
    if (params.temperature == 0.0f) {
        return static_cast<int32_t>(
            std::max_element(work.begin(), work.end()) - work.begin());
    }

    // Step 4: Temperature scaling.
    for (float& v : work) v /= params.temperature;

    // Step 5: Top-k filtering.
    // Keep only the top-k logits; set the rest to -infinity.
    if (params.top_k > 0 && params.top_k < vocab_size) {
        // Find the k-th largest value via partial sort on a copy of indices.
        std::vector<int> indices(vocab_size);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + params.top_k,
                          indices.end(),
                          [&](int a, int b) { return work[a] > work[b]; });

        // Build a mask: mark the top-k positions as kept.
        std::vector<bool> keep(vocab_size, false);
        for (int i = 0; i < params.top_k; ++i) keep[indices[i]] = true;

        const float neg_inf = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < vocab_size; ++i)
            if (!keep[i]) work[i] = neg_inf;
    }

    // Step 6: Top-p (nucleus) filtering.
    // Compute softmax probabilities, sort descending, keep the smallest prefix
    // whose cumulative probability exceeds top_p, set the rest to -infinity.
    if (params.top_p < 1.0f) {
        // Stable softmax.
        float max_val = *std::max_element(work.begin(), work.end());
        std::vector<float> probs(vocab_size);
        float sum = 0.0f;
        for (int i = 0; i < vocab_size; ++i) {
            probs[i] = std::exp(work[i] - max_val);
            sum += probs[i];
        }
        for (float& p : probs) p /= sum;

        // Sort indices by descending probability.
        std::vector<int> idx(vocab_size);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b) { return probs[a] > probs[b]; });

        float cumsum = 0.0f;
        const float neg_inf = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < vocab_size; ++i) {
            cumsum += probs[idx[i]];
            // Keep this token; once we have exceeded top_p, mask the rest.
            if (cumsum > params.top_p) {
                for (int j = i + 1; j < vocab_size; ++j)
                    work[idx[j]] = neg_inf;
                break;
            }
        }
    }

    // Step 7: Softmax over the remaining (non-masked) logits.
    float max_val = *std::max_element(work.begin(), work.end());
    std::vector<float> probs(vocab_size);
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        probs[i] = std::exp(work[i] - max_val);
        sum += probs[i];
    }
    for (float& p : probs) p /= sum;

    // Step 8: Sample from the categorical distribution.
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float u = dist(rng_);
    float cumsum = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        cumsum += probs[i];
        if (u <= cumsum) return static_cast<int32_t>(i);
    }
    // Fallback: return last valid token (handles floating-point rounding).
    for (int i = vocab_size - 1; i >= 0; --i) {
        if (probs[i] > 0.0f) return static_cast<int32_t>(i);
    }
    return vocab_size - 1;
}
