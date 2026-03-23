// Debug tool: loads model, runs a single forward pass, dumps intermediate values.
// Compile: link against core, kernels, llama_model, add to tests or build manually.
// Usage: ./debug_dump <model_dir>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>

#include "core/memory.h"
#include "core/types.h"
#include "kernels/activations.h"
#include "kernels/cublas_provider.h"
#include "kernels/embedding.h"
#include "kernels/rmsnorm.h"
#include "kernels/rope.h"
#include "models/model_interface.h"

extern std::unique_ptr<ModelModule> create_llama_model(KernelProvider* kernels);

// Dump first N float values from GPU pointer
void dump_gpu(const char* label, const float* d_ptr, int n, int total = 0) {
    std::vector<float> buf(n);
    cudaMemcpy(buf.data(), d_ptr, n * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << label << " (first " << n << " of " << (total > 0 ? total : n)
              << "): [";
    for (int i = 0; i < n; i++) {
        if (i > 0) std::cout << ", ";
        std::cout << buf[i];
    }
    std::cout << "]" << std::endl;

    // Stats
    if (total > 0) {
        std::vector<float> full(total);
        cudaMemcpy(full.data(), d_ptr, total * sizeof(float),
                   cudaMemcpyDeviceToHost);
        float mn = full[0], mx = full[0], sum = 0;
        for (float v : full) {
            mn = std::min(mn, v);
            mx = std::max(mx, v);
            sum += v;
        }
        float mean = sum / total;
        float var = 0;
        for (float v : full) var += (v - mean) * (v - mean);
        float sd = std::sqrt(var / total);
        std::cout << "  stats: min=" << mn << " max=" << mx << " mean=" << mean
                  << " std=" << sd << std::endl;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_dir>" << std::endl;
        return 1;
    }
    std::string model_dir = argv[1];

    // Create provider and model
    CublasProvider kernels;
    auto model = create_llama_model(&kernels);
    model->load_weights(model_dir);
    auto cfg = model->config();

    std::cout << "\nModel: layers=" << cfg.num_layers
              << " hidden=" << cfg.hidden_size
              << " heads=" << cfg.num_attention_heads
              << " kv_heads=" << cfg.num_kv_heads
              << " vocab=" << cfg.vocab_size << std::endl;

    // Helper lambda: run prefill on a token sequence and print top-5 logits
    auto run_prefill_test = [&](const char* label,
                                const std::vector<int32_t>& tokens,
                                const std::vector<std::pair<int,float>>& hf_top5) {
        int N = static_cast<int>(tokens.size());
        std::cout << "\n=== " << label << " (" << N << " tokens) ===" << std::endl;

        BlockPool pool(200, 4096);
        // Allocate enough blocks (block_size=16, so 1 block covers up to 16 tokens)
        int num_blocks = (N + 15) / 16;
        std::vector<int> btable;
        for (int i = 0; i < num_blocks; i++) {
            auto blk = pool.allocate();
            btable.push_back(blk.value());
        }

        RequestContext req;
        req.request_id = 1;
        req.token_ids = tokens;
        req.seq_len = N;
        req.prefill_start_pos = 0;
        req.prefill_chunk_len = N;
        req.block_table = btable;
        req.max_new_tokens = 1;

        auto result = model->prefill({req}, nullptr);

        // Top 5
        std::vector<int> indices(result.logits.size());
        for (size_t i = 0; i < indices.size(); i++) indices[i] = i;
        std::partial_sort(indices.begin(), indices.begin() + 5, indices.end(),
                          [&](int a, int b) {
                              return result.logits[a] > result.logits[b];
                          });
        std::cout << "Our top-5:" << std::endl;
        for (int i = 0; i < 5; i++) {
            std::cout << "  tok" << indices[i] << " = " << result.logits[indices[i]] << std::endl;
        }
        std::cout << "HF top-5:" << std::endl;
        for (auto& [tok, val] : hf_top5) {
            std::cout << "  tok" << tok << " = " << val
                      << "  (ours=" << result.logits[tok] << ")" << std::endl;
        }
    };

    // 1-token: BOS=128000 → HF top: 14924=11.74, 128006=11.02, 17297=10.53
    run_prefill_test("1-token [BOS]",
                     {128000},
                     {{14924, 11.7413f}, {128006, 11.0166f}, {17297, 10.5269f},
                      {755, 10.1547f}, {121648, 9.6178f}});

    // 2-token: [BOS, <|start_header_id|>=128006]
    // HF reference: top tok128006=24.62
    run_prefill_test("2-token [BOS, 128006]",
                     {128000, 128006},
                     {{128006, 24.62f}, {128009, 0.0f}, {882, 0.0f}});

    // 3-token: [BOS, <|start_header_id|>, system]
    // HF reference: top tok198=15.40 ('\n')
    run_prefill_test("3-token [BOS, 128006, 9125]",
                     {128000, 128006, 9125},
                     {{198, 15.40f}, {271, 0.0f}, {29 , 0.0f}});

    return 0;
}
