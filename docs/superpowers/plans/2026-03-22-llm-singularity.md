# LLM-Singularity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone C++ LLM serving system where each model is a self-contained bundle (forward pass + runtime policy) that plugs into a shared core library, starting with a LLaMA prototype.

**Architecture:** Shared core library (HTTP server, scheduler, tokenizer, sampling, memory pool) with a narrow `ModelModule` interface. Per-model generated modules implement forward pass and KV cache policy. Kernel abstraction layer with cuBLAS + FlashAttention for prototype. Single binary per model.

**Tech Stack:** C++17, CUDA 12.x, cuBLAS, FlashAttention-2, CMake 3.24+, cpp-httplib, nlohmann/json, sentencepiece

**Spec:** `docs/superpowers/specs/2026-03-22-llm-serve-design.md`

---

## File Structure

```
llm-singularity/
├── CMakeLists.txt                              # top-level CMake
├── core/
│   ├── CMakeLists.txt                          # core library build
│   ├── include/
│   │   ├── core/types.h                        # shared types (DataType, enums)
│   │   ├── core/server.h                       # HTTP server + OpenAI endpoints
│   │   ├── core/scheduler.h                    # continuous batching scheduler
│   │   ├── core/tokenizer.h                    # tokenizer interface
│   │   ├── core/sampling.h                     # sampling strategies
│   │   └── core/memory.h                       # GPU memory pool + prefix cache
│   └── src/
│       ├── server.cpp
│       ├── scheduler.cpp
│       ├── tokenizer.cpp
│       ├── sampling.cpp
│       └── memory.cpp
├── kernels/
│   ├── CMakeLists.txt
│   ├── include/
│   │   └── kernels/kernel_provider.h           # abstract kernel API
│   └── src/
│       ├── cublas_provider.h / cublas_provider.cpp  # cuBLAS GEMM implementation
│       ├── attention.h / attention.cu          # FlashAttention wrapper + paged KV
│       ├── rmsnorm.h / rmsnorm.cu              # fused RMSNorm kernel
│       ├── activations.h / activations.cu      # SwiGLU, SiLU fused kernels
│       └── rope.h / rope.cu                    # RoPE kernel
├── models/
│   ├── include/
│   │   └── models/model_interface.h            # ModelModule ABC + RequestContext + ForwardResult
│   └── llama/
│       ├── CMakeLists.txt
│       ├── llama_model.h / llama_model.cpp     # LLaMA forward pass
│       └── llama_runtime.h / llama_runtime.cpp # LLaMA KV cache policy + scheduling hints
├── app/
│   ├── CMakeLists.txt
│   └── main.cpp                                # CLI entry point, wires everything together
├── tests/
│   ├── CMakeLists.txt
│   ├── test_types.cpp                          # types tests
│   ├── test_memory.cpp                         # memory pool + prefix cache tests
│   ├── test_sampling.cpp                       # sampling tests
│   ├── test_scheduler.cpp                      # scheduler tests
│   ├── test_tokenizer.cpp                      # tokenizer tests
│   ├── test_kernels.cpp                        # kernel correctness tests (GPU)
│   ├── test_llama_model.cpp                    # LLaMA forward pass tests (GPU)
│   └── test_server.cpp                         # server endpoint tests
└── third_party/
    ├── CMakeLists.txt
    └── README.md                               # instructions for fetching deps
```

---

## Task 1: Project Scaffolding + Build System

**Files:**
- Create: `CMakeLists.txt`
- Create: `core/CMakeLists.txt`
- Create: `kernels/CMakeLists.txt`
- Create: `models/llama/CMakeLists.txt`
- Create: `app/CMakeLists.txt`
- Create: `tests/CMakeLists.txt`
- Create: `third_party/CMakeLists.txt`
- Create: `third_party/README.md`
- Create: `.gitignore`

- [ ] **Step 1: Create top-level CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.24)
project(llm-singularity LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

option(BUILD_TESTS "Build tests" ON)
set(MODEL "llama" CACHE STRING "Model to build")

# Find CUDA and cuBLAS
find_package(CUDAToolkit REQUIRED)

# Third-party deps
add_subdirectory(third_party)

# Core library
add_subdirectory(core)

# Kernels library
add_subdirectory(kernels)

# Model module
add_subdirectory(models/${MODEL})

# Main application
add_subdirectory(app)

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
```

- [ ] **Step 2: Create third_party/CMakeLists.txt with FetchContent for header-only deps**

```cmake
include(FetchContent)

# cpp-httplib (header-only HTTP server)
FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG v0.18.3
)
FetchContent_MakeAvailable(httplib)

# nlohmann/json (header-only JSON)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

# Google Test
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)
FetchContent_MakeAvailable(googletest)

# sentencepiece (tokenizer)
FetchContent_Declare(
    sentencepiece
    GIT_REPOSITORY https://github.com/google/sentencepiece.git
    GIT_TAG v0.2.0
)
set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(sentencepiece)
```

- [ ] **Step 3: Create stub CMakeLists.txt for core, kernels, models/llama, app, tests**

Each sub-CMakeLists.txt should define its library/executable target with placeholder source files. Create minimal placeholder `.cpp` files so the build succeeds.

core/CMakeLists.txt:
```cmake
add_library(core STATIC
    src/server.cpp
    src/scheduler.cpp
    src/tokenizer.cpp
    src/sampling.cpp
    src/memory.cpp
)
target_include_directories(core PUBLIC include)
target_link_libraries(core PUBLIC httplib::httplib nlohmann_json::nlohmann_json sentencepiece-static CUDA::cudart)
```

kernels/CMakeLists.txt:
```cmake
add_library(kernels STATIC
    src/cublas_provider.cpp
    src/attention.cu
    src/rmsnorm.cu
    src/activations.cu
    src/rope.cu
)
target_include_directories(kernels PUBLIC include)
target_link_libraries(kernels PUBLIC CUDA::cublas CUDA::cudart)
```

models/llama/CMakeLists.txt:
```cmake
add_library(llama_model STATIC
    llama_model.cpp
    llama_runtime.cpp
)
target_include_directories(llama_model PUBLIC ${CMAKE_SOURCE_DIR}/models/include)
target_link_libraries(llama_model PUBLIC core kernels)
```

app/CMakeLists.txt:
```cmake
add_executable(llm-serve-${MODEL} main.cpp)
target_link_libraries(llm-serve-${MODEL} PRIVATE core kernels llama_model)
```

tests/CMakeLists.txt:
```cmake
add_executable(tests
    test_types.cpp
)
target_link_libraries(tests PRIVATE core kernels GTest::gtest_main)
add_test(NAME unit_tests COMMAND tests)
```

- [ ] **Step 4: Create .gitignore**

```
build/
.cache/
compile_commands.json
*.o
*.a
*.so
```

- [ ] **Step 5: Create placeholder source files so the build compiles**

Create minimal stub files:
- `core/include/core/types.h` — empty header with include guard
- `core/src/server.cpp`, `core/src/scheduler.cpp`, `core/src/tokenizer.cpp`, `core/src/sampling.cpp`, `core/src/memory.cpp` — empty files
- `kernels/include/kernels/kernel_provider.h` — empty header
- `kernels/src/cublas_provider.cpp`, `kernels/src/attention.cu`, `kernels/src/rmsnorm.cu`, `kernels/src/activations.cu`, `kernels/src/rope.cu` — empty files
- `models/include/models/model_interface.h` — empty header
- `models/llama/llama_model.h`, `models/llama/llama_model.cpp`, `models/llama/llama_runtime.h`, `models/llama/llama_runtime.cpp` — empty files
- `app/main.cpp` — `int main() { return 0; }`
- `tests/test_types.cpp` — `#include <gtest/gtest.h>` + one placeholder test

- [ ] **Step 6: Verify build compiles**

Run:
```bash
cd /home/suyogg/dev/llm-singularity
mkdir -p build && cd build
cmake .. -DMODEL=llama
make -j$(nproc)
```
Expected: Clean build, `llm-serve-llama` binary produced, test binary produced.

- [ ] **Step 7: Run placeholder test**

Run:
```bash
cd build && ctest --output-on-failure
```
Expected: 1 test passes.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -s -m "feat: project scaffolding with CMake build system"
```

---

## Task 2: Shared Types + Model Module Interface

**Files:**
- Create: `core/include/core/types.h`
- Create: `models/include/models/model_interface.h`
- Create: `kernels/include/kernels/kernel_provider.h`
- Modify: `tests/test_types.cpp`

- [ ] **Step 1: Write test for shared types**

```cpp
// tests/test_types.cpp
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd build && cmake .. && make -j && ctest --output-on-failure`
Expected: FAIL — `DataType` not defined.

- [ ] **Step 3: Implement core/types.h**

```cpp
// core/include/core/types.h
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum class DataType {
    kFloat16,
    kBFloat16,
    kFloat32,
};

inline size_t dtype_size(DataType dt) {
    switch (dt) {
        case DataType::kFloat16:  return 2;
        case DataType::kBFloat16: return 2;
        case DataType::kFloat32:  return 4;
    }
    return 0;
}

inline std::string dtype_name(DataType dt) {
    switch (dt) {
        case DataType::kFloat16:  return "float16";
        case DataType::kBFloat16: return "bfloat16";
        case DataType::kFloat32:  return "float32";
    }
    return "unknown";
}
```

- [ ] **Step 4: Implement KernelProvider interface**

```cpp
// kernels/include/kernels/kernel_provider.h
#pragma once
#include <cuda_runtime.h>
#include "core/types.h"

struct GemmDescriptor {
    int M, N, K;
    DataType input_dtype;
    DataType output_dtype;
    bool transA = false;
    bool transB = false;
};

struct AttentionDescriptor {
    int batch_size;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int max_seq_len;
    DataType dtype;
};

class KernelProvider {
public:
    virtual ~KernelProvider() = default;

    virtual void gemm(GemmDescriptor desc,
                      const void* A, const void* B, void* C,
                      cudaStream_t stream) = 0;

    virtual void fused_attention(AttentionDescriptor desc,
                                 const void* Q, const void* K, const void* V,
                                 void* output,
                                 const int* block_table,
                                 int block_size,
                                 const int* seq_lens,
                                 cudaStream_t stream) = 0;

    virtual void rms_norm(const void* input, const void* weight, void* output,
                          int rows, int hidden_size, float eps,
                          DataType dtype, cudaStream_t stream) = 0;

    virtual void silu_mul(const void* input, const void* gate, void* output,
                          int size, DataType dtype, cudaStream_t stream) = 0;

    virtual void rope(void* q, void* k,
                      int batch_size, int seq_len, int num_heads,
                      int num_kv_heads, int head_dim,
                      const int* positions, float theta,
                      DataType dtype, cudaStream_t stream) = 0;
};
```

- [ ] **Step 5: Implement ModelModule interface**

```cpp
// models/include/models/model_interface.h
#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <cuda_runtime.h>
#include "core/types.h"
#include "kernels/kernel_provider.h"

struct ModelConfig {
    int num_layers;
    int hidden_size;
    int num_attention_heads;
    int num_kv_heads;
    int intermediate_size;
    int vocab_size;
    int max_seq_len;
    int eos_token_id;
    std::vector<int> stop_token_ids;
    DataType dtype;
    std::unordered_map<std::string, std::string> extra;
};

struct KVCacheConfig {
    int block_size;       // tokens per block
    int max_blocks;       // total blocks in pool
    DataType cache_dtype;
};

struct RequestContext {
    int64_t request_id;
    std::vector<int32_t> token_ids;
    int seq_len;
    int prefill_start_pos;
    int prefill_chunk_len;
    std::vector<int> block_table;
    int max_new_tokens;
};

struct ForwardResult {
    // logits for each request: [batch_size * vocab_size]
    std::vector<float> logits;
    int vocab_size;
};

class ModelModule {
public:
    virtual ~ModelModule() = default;

    virtual ModelConfig config() const = 0;
    virtual void load_weights(const std::string& weight_path) = 0;

    virtual KVCacheConfig kv_cache_config() const = 0;

    virtual ForwardResult prefill(const std::vector<RequestContext>& requests,
                                  cudaStream_t stream) = 0;
    virtual ForwardResult decode(const std::vector<RequestContext>& requests,
                                 cudaStream_t stream) = 0;

    virtual int max_batch_size() const = 0;
    virtual int max_tokens_per_batch() const = 0;
    virtual bool supports_in_flight_batching() const { return true; }
};

// Factory — each generated model module provides this.
using ModelModuleFactory = std::unique_ptr<ModelModule>(*)(KernelProvider*);
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd build && cmake .. && make -j && ctest --output-on-failure`
Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -s -m "feat: define shared types, KernelProvider, and ModelModule interfaces"
```

---

## Task 3: GPU Memory Pool

**Files:**
- Create: `core/include/core/memory.h`
- Modify: `core/src/memory.cpp`
- Modify: `tests/test_memory.cpp`

- [ ] **Step 1: Write tests for block pool**

```cpp
// tests/test_memory.cpp
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd build && cmake .. && make -j && ctest --output-on-failure`
Expected: FAIL — `BlockPool` not defined.

- [ ] **Step 3: Implement BlockPool**

```cpp
// core/include/core/memory.h
#pragma once
#include <cstddef>
#include <optional>
#include <vector>
#include <stack>
#include <cuda_runtime.h>

class BlockPool {
public:
    BlockPool(int num_blocks, size_t block_size_bytes);
    ~BlockPool();

    std::optional<int> allocate();          // returns block index
    void free(int block_idx);
    int num_free_blocks() const;
    int num_total_blocks() const;
    size_t block_size_bytes() const;
    void* block_ptr(int block_idx) const;   // GPU pointer for block

private:
    int num_blocks_;
    size_t block_size_bytes_;
    void* gpu_buffer_ = nullptr;            // single GPU allocation
    std::stack<int> free_blocks_;
};
```

```cpp
// core/src/memory.cpp
#include "core/memory.h"
#include <stdexcept>

BlockPool::BlockPool(int num_blocks, size_t block_size_bytes)
    : num_blocks_(num_blocks), block_size_bytes_(block_size_bytes) {
    if (num_blocks > 0 && block_size_bytes > 0) {
        cudaMalloc(&gpu_buffer_, static_cast<size_t>(num_blocks) * block_size_bytes);
    }
    for (int i = num_blocks - 1; i >= 0; --i) {
        free_blocks_.push(i);
    }
}

BlockPool::~BlockPool() {
    if (gpu_buffer_) {
        cudaFree(gpu_buffer_);
    }
}

std::optional<int> BlockPool::allocate() {
    if (free_blocks_.empty()) return std::nullopt;
    int idx = free_blocks_.top();
    free_blocks_.pop();
    return idx;
}

void BlockPool::free(int block_idx) {
    free_blocks_.push(block_idx);
}

int BlockPool::num_free_blocks() const { return static_cast<int>(free_blocks_.size()); }
int BlockPool::num_total_blocks() const { return num_blocks_; }
size_t BlockPool::block_size_bytes() const { return block_size_bytes_; }

void* BlockPool::block_ptr(int block_idx) const {
    return static_cast<char*>(gpu_buffer_) + static_cast<size_t>(block_idx) * block_size_bytes_;
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd build && cmake .. && make -j && ctest --output-on-failure`
Expected: All memory tests pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -s -m "feat: implement GPU block pool for KV cache memory management"
```

---

## Task 4: Prefix Cache (Radix Tree)

**Files:**
- Modify: `core/include/core/memory.h` — add `PrefixCache` class
- Modify: `core/src/memory.cpp` — implement `PrefixCache`
- Modify: `tests/test_memory.cpp` — add prefix cache tests

- [ ] **Step 1: Write tests for prefix cache**

```cpp
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
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — `PrefixCache` not defined.

- [ ] **Step 3: Implement PrefixCache**

Implement a radix-tree-based prefix cache with:
- `insert(tokens, block_indices)` — store mapping
- `lookup(tokens)` — return longest prefix match with block indices and matched token count
- `add_ref(tokens)` / `release(tokens)` — reference counting
- `ref_count(tokens)` — query ref count
- `evict_lru()` — evict least recently used entry with ref_count == 0, return its blocks

**Block-alignment constraint**: Each block holds `block_size` tokens (e.g., 16). The prefix cache only stores *complete* blocks. A 50-token prefix with block_size=16 stores 3 blocks (48 tokens) — the remaining 2 tokens must be recomputed. The `PrefixCacheResult::matched_tokens` is always a multiple of `block_size`. The constructor takes `block_size` as a parameter to enforce this:

```cpp
struct PrefixCacheResult {
    int matched_tokens;            // always a multiple of block_size
    std::vector<int> blocks;       // block indices for the matched prefix
};

class PrefixCache {
public:
    explicit PrefixCache(int block_size = 16);
    // ...
};
```

Use a trie (radix tree) where each edge represents a block-sized chunk of tokens, and nodes store the corresponding block index. LRU ordering via an intrusive doubly-linked list on unreferenced entries.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd build && cmake .. && make -j && ctest --output-on-failure`
Expected: All prefix cache tests pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -s -m "feat: implement prefix cache with radix tree and LRU eviction"
```

---

## Task 5: Sampling

**Files:**
- Create: `core/include/core/sampling.h`
- Modify: `core/src/sampling.cpp`
- Create: `tests/test_sampling.cpp`

- [ ] **Step 1: Write tests for sampling**

```cpp
// tests/test_sampling.cpp
#include <gtest/gtest.h>
#include "core/sampling.h"
#include <numeric>
#include <cmath>

TEST(SamplingTest, GreedySampling) {
    // Logits: token 3 has highest logit
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
    // With very low temperature, should pick highest
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
    // Only tokens 1 and 3 considered (top-2), token 1 has highest
    EXPECT_EQ(token, 1);
}

TEST(SamplingTest, RepetitionPenalty) {
    // Token 2 was previously generated — its logit should be penalized
    std::vector<float> logits = {1.0f, 1.0f, 5.0f, 1.0f};
    SamplingParams params;
    params.temperature = 0.001f;
    params.repetition_penalty = 100.0f;  // heavy penalty

    Sampler sampler;
    std::vector<int32_t> prev_tokens = {2};  // token 2 was already generated
    auto token = sampler.sample(logits.data(), logits.size(), params, prev_tokens);
    // Token 2 penalized heavily — should pick something else
    EXPECT_NE(token, 2);
}
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — `Sampler` not defined.

- [ ] **Step 3: Implement Sampler**

```cpp
// core/include/core/sampling.h
#pragma once
#include <cstdint>
#include <random>

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
```

Implementation in `core/src/sampling.cpp`:
- Apply repetition penalty: for each token in `prev_tokens`, divide its logit by `repetition_penalty` if positive, multiply if negative
- Apply temperature scaling: `logits[i] /= temperature`
- Apply top-k: keep only top-k logits, set rest to -inf
- Apply top-p (nucleus): sort by probability, keep cumulative sum <= top_p
- Softmax over remaining logits
- Sample from distribution (or argmax if temperature == 0)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd build && cmake .. && make -j && ctest --output-on-failure`
Expected: All sampling tests pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -s -m "feat: implement CPU sampler with temperature, top-k, top-p"
```

---

## Task 6: Tokenizer Interface

**Files:**
- Create: `core/include/core/tokenizer.h`
- Modify: `core/src/tokenizer.cpp`
- Create: `tests/test_tokenizer.cpp`

- [ ] **Step 1: Write tests for tokenizer**

```cpp
// tests/test_tokenizer.cpp
#include <gtest/gtest.h>
#include "core/tokenizer.h"

TEST(TokenizerTest, LoadAndEncodeDecode) {
    // This test requires a sentencepiece model file.
    // Skip if not available.
    const char* model_path = std::getenv("TEST_TOKENIZER_PATH");
    if (!model_path) GTEST_SKIP() << "TEST_TOKENIZER_PATH not set";

    Tokenizer tok;
    tok.load(model_path);

    auto ids = tok.encode("Hello, world!");
    EXPECT_GT(ids.size(), 0u);

    auto text = tok.decode(ids);
    EXPECT_EQ(text, "Hello, world!");
}

TEST(TokenizerTest, SpecialTokens) {
    const char* model_path = std::getenv("TEST_TOKENIZER_PATH");
    if (!model_path) GTEST_SKIP() << "TEST_TOKENIZER_PATH not set";

    Tokenizer tok;
    tok.load(model_path);

    EXPECT_GE(tok.eos_token_id(), 0);
    EXPECT_GE(tok.bos_token_id(), 0);
}
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — `Tokenizer` not defined.

- [ ] **Step 3: Implement Tokenizer wrapping sentencepiece**

```cpp
// core/include/core/tokenizer.h
#pragma once
#include <string>
#include <vector>

class Tokenizer {
public:
    Tokenizer() = default;
    void load(const std::string& model_path);

    std::vector<int32_t> encode(const std::string& text) const;
    std::string decode(const std::vector<int32_t>& ids) const;

    int32_t eos_token_id() const;
    int32_t bos_token_id() const;
    int vocab_size() const;

    // Simple hardcoded chat template for prototype
    std::string apply_chat_template(
        const std::vector<std::pair<std::string, std::string>>& messages) const;

private:
    // sentencepiece processor (pimpl to avoid header dep)
    struct Impl;
    std::shared_ptr<Impl> impl_;
};
```

Implementation wraps `sentencepiece::SentencePieceProcessor`. The `apply_chat_template` method uses a hardcoded LLaMA chat format for the prototype.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd build && cmake .. && make -j && ctest --output-on-failure`
Expected: Tests pass (or skip if no tokenizer model available).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -s -m "feat: implement tokenizer interface wrapping sentencepiece"
```

---

## Task 7: CUDA Kernels — RMSNorm, SwiGLU, RoPE

**Files:**
- Modify: `kernels/src/rmsnorm.h`, `kernels/src/rmsnorm.cu`
- Modify: `kernels/src/activations.h`, `kernels/src/activations.cu`
- Modify: `kernels/src/rope.h`, `kernels/src/rope.cu`
- Create: `tests/test_kernels.cpp`

- [ ] **Step 1: Write GPU tests for RMSNorm**

```cpp
// tests/test_kernels.cpp
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include "kernels/rmsnorm.h"

TEST(KernelTest, RMSNorm) {
    const int rows = 2, cols = 4;
    std::vector<float> h_input = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> h_weight = {1, 1, 1, 1};
    std::vector<float> h_output(rows * cols);

    float *d_input, *d_weight, *d_output;
    cudaMalloc(&d_input, rows * cols * sizeof(float));
    cudaMalloc(&d_weight, cols * sizeof(float));
    cudaMalloc(&d_output, rows * cols * sizeof(float));
    cudaMemcpy(d_input, h_input.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight.data(), cols * sizeof(float), cudaMemcpyHostToDevice);

    launch_rms_norm(d_input, d_weight, d_output, rows, cols, 1e-6f, nullptr);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output.data(), d_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify: RMSNorm(x) = x / rms(x) * weight
    // Row 0: rms = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
    float rms0 = std::sqrt((1+4+9+16) / 4.0f);
    EXPECT_NEAR(h_output[0], 1.0f / rms0, 1e-4);
    EXPECT_NEAR(h_output[1], 2.0f / rms0, 1e-4);

    cudaFree(d_input); cudaFree(d_weight); cudaFree(d_output);
}
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — `launch_rms_norm` not defined.

- [ ] **Step 3: Implement RMSNorm CUDA kernel**

Standard fused RMSNorm: compute RMS per row via parallel reduction, normalize and scale in one pass. Support float32 compute with fp16/bf16 I/O.

- [ ] **Step 4: Write test for SwiGLU, implement SwiGLU kernel**

SwiGLU: `output = SiLU(gate) * up` fused in a single kernel. Test with known values.

- [ ] **Step 5: Write test for RoPE, implement RoPE kernel**

RoPE: apply rotary position embeddings to Q and K tensors. Test with known positions and verify against a reference implementation.

- [ ] **Step 6: Run all kernel tests**

Run: `cd build && cmake .. && make -j && ctest --output-on-failure`
Expected: All kernel tests pass.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -s -m "feat: implement RMSNorm, SwiGLU, and RoPE CUDA kernels"
```

---

## Task 8: cuBLAS GEMM Provider

**Files:**
- Modify: `kernels/src/cublas_provider.h`, `kernels/src/cublas_provider.cpp`
- Modify: `tests/test_kernels.cpp`

- [ ] **Step 1: Write GPU test for GEMM**

```cpp
TEST(KernelTest, CublasGemm) {
    // Simple 2x3 * 3x4 = 2x4 GEMM in float32
    const int M = 2, N = 4, K = 3;
    std::vector<float> h_A = {1, 2, 3, 4, 5, 6};
    std::vector<float> h_B = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<float> h_C(M * N);

    // ... allocate GPU memory, copy, call provider->gemm(), copy back ...

    // Expected: standard matmul result
    EXPECT_NEAR(h_C[0], 38.0f, 1e-3);  // row0*col0: 1*1+2*5+3*9
}
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — `CublasProvider` not defined.

- [ ] **Step 3: Implement CublasProvider**

```cpp
// kernels/src/cublas_provider.h
#pragma once
#include "kernels/kernel_provider.h"
#include <cublas_v2.h>

class CublasProvider : public KernelProvider {
public:
    CublasProvider();
    ~CublasProvider();

    void gemm(GemmDescriptor desc,
              const void* A, const void* B, void* C,
              cudaStream_t stream) override;

    // Other methods delegate to standalone CUDA kernels
    void fused_attention(...) override;
    void rms_norm(...) override;
    void silu_mul(...) override;
    void rope(...) override;

private:
    cublasHandle_t handle_;
};
```

The `gemm` method calls `cublasGemmEx` with appropriate dtype handling. Other methods call the standalone kernels from Task 7.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd build && cmake .. && make -j && ctest --output-on-failure`
Expected: GEMM test passes.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -s -m "feat: implement CublasProvider wrapping cuBLAS GEMM and CUDA kernels"
```

---

## Task 9: Paged Attention Kernel

**Files:**
- Modify: `kernels/src/attention.h`, `kernels/src/attention.cu`
- Modify: `tests/test_kernels.cpp`

**KV Cache Layout** (used by all attention implementations):
```
KV cache is stored as a pool of blocks allocated by core/memory/BlockPool.
Each block stores block_size tokens of KV data for all layers and all KV heads.

Per-block layout: [num_kv_heads, block_size, head_dim] for K and V separately.
Total per-block bytes: 2 * num_kv_heads * block_size * head_dim * dtype_size
  (factor of 2 for K and V)

A request's KV cache is described by its block_table: a list of block indices.
To read K for token position t:
  block_idx = block_table[t / block_size]
  offset_in_block = t % block_size
  K_ptr = BlockPool::block_ptr(block_idx) + kv_head * block_size * head_dim + offset_in_block * head_dim
```

- [ ] **Step 1: Write GPU test for paged attention**

```cpp
TEST(KernelTest, PagedAttention) {
    // Small test: batch_size=1, num_heads=2, num_kv_heads=2, head_dim=8
    // 2 blocks of 4 tokens each (block_size=4), total seq_len=6
    const int batch_size = 1, num_heads = 2, num_kv_heads = 2;
    const int head_dim = 8, block_size = 4, seq_len = 6;

    // Allocate Q: [batch_size, num_heads, 1, head_dim] (single query token for decode)
    // Allocate K cache: 2 blocks * [num_kv_heads, block_size, head_dim]
    // Allocate V cache: 2 blocks * [num_kv_heads, block_size, head_dim]
    // Block table: [0, 1] (2 blocks for this request)
    // Seq lens: [6]

    // Fill K, V with known values. Fill Q with known values.
    // Compute reference attention on CPU:
    //   scores = Q @ K^T / sqrt(head_dim)  (causal mask: only attend to positions 0..5)
    //   probs = softmax(scores)
    //   output = probs @ V

    // Launch paged attention kernel
    // Compare GPU output to CPU reference within tolerance

    // ... allocate GPU memory, set up block table, launch kernel ...

    float *d_Q, *d_output;
    float *d_k_cache, *d_v_cache;  // contiguous pool, blocks indexed by block_table
    int *d_block_table, *d_seq_lens;

    // ... cudaMalloc, cudaMemcpy setup ...

    launch_paged_attention(
        d_Q, d_k_cache, d_v_cache, d_output,
        d_block_table, d_seq_lens,
        batch_size, num_heads, num_kv_heads, head_dim,
        block_size, /*max_seq_len=*/8,
        nullptr /*stream*/
    );
    cudaDeviceSynchronize();

    // ... copy back and compare against CPU reference ...
    // Use EXPECT_NEAR with tolerance 1e-3 for each element
}
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — `launch_paged_attention` not defined.

- [ ] **Step 3: Implement paged attention kernel**

Per the spec's risk mitigation, use a **straightforward naive CUDA kernel** for the prototype (not FlashAttention-level tiling). The `KernelProvider` abstraction allows drop-in replacement with FlashAttention-2/3 later.

```cpp
// kernels/src/attention.h
#pragma once
#include <cuda_runtime.h>

// Paged attention: Q attends to K,V stored in paged blocks.
// Q shape: [batch_size, num_heads, 1, head_dim] (decode) or [batch_size, num_heads, q_len, head_dim] (prefill)
// K/V cache: contiguous pool of blocks, each [num_kv_heads, block_size, head_dim]
// block_table: [batch_size, max_blocks_per_seq] — maps (request, block_idx) to pool block
// seq_lens: [batch_size] — actual sequence length per request
void launch_paged_attention(
    const float* Q, const float* k_cache, const float* v_cache,
    float* output,
    const int* block_table, const int* seq_lens,
    int batch_size, int num_heads, int num_kv_heads, int head_dim,
    int block_size, int max_seq_len,
    cudaStream_t stream);
```

Implementation approach (naive but correct):
1. One CUDA thread block per (batch, head) pair
2. Each thread block iterates over all K blocks in the block table for this request
3. Computes Q @ K^T scores with causal masking (positions > current seq_len masked to -inf)
4. Online softmax (numerically stable: track running max and sum)
5. Accumulates softmax(scores) @ V
6. Writes output for this (batch, head) pair

For GQA: map query heads to KV heads via `kv_head = q_head / (num_heads / num_kv_heads)`.

**Fallback plan**: If the custom kernel proves too buggy, wrap FlashAttention-2's `flash_attn_with_kvcache` C API behind `KernelProvider::fused_attention`. Add FlashAttention-2 to `third_party/CMakeLists.txt` as a FetchContent dependency.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd build && cmake .. && make -j && ctest --output-on-failure`
Expected: Attention test passes — output matches CPU reference within tolerance.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -s -m "feat: implement paged attention CUDA kernel with block table support"
```

---

## Task 10: Safetensors Reader

**Files:**
- Create: `core/include/core/safetensors.h`
- Create: `core/src/safetensors.cpp`
- Modify: `tests/CMakeLists.txt`
- Create: `tests/test_safetensors.cpp`

- [ ] **Step 1: Write test for safetensors reader**

```cpp
TEST(SafetensorsTest, ParseSmallFile) {
    // Create a minimal safetensors file in-memory or from a test fixture
    const char* model_path = std::getenv("TEST_MODEL_PATH");
    if (!model_path) GTEST_SKIP() << "TEST_MODEL_PATH not set";

    SafetensorsFile sf;
    sf.open(std::string(model_path) + "/model.safetensors");

    auto names = sf.tensor_names();
    EXPECT_GT(names.size(), 0u);

    auto info = sf.tensor_info(names[0]);
    EXPECT_GT(info.num_elements(), 0u);
}
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — `SafetensorsFile` not defined.

- [ ] **Step 3: Implement safetensors reader**

Safetensors format: 8 bytes (header size as uint64 LE) + JSON header + raw tensor data. The JSON header maps tensor names to `{dtype, shape, data_offsets}`.

```cpp
// core/include/core/safetensors.h
#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include "core/types.h"

struct TensorInfo {
    std::string name;
    DataType dtype;
    std::vector<int64_t> shape;
    size_t offset;      // byte offset into data section
    size_t byte_size;

    int64_t num_elements() const;
};

class SafetensorsFile {
public:
    void open(const std::string& path);
    std::vector<std::string> tensor_names() const;
    TensorInfo tensor_info(const std::string& name) const;
    const void* tensor_data(const std::string& name) const;  // memory-mapped pointer

private:
    int fd_ = -1;
    void* mmap_ptr_ = nullptr;
    size_t file_size_ = 0;
    size_t data_offset_ = 0;
    std::unordered_map<std::string, TensorInfo> tensors_;
};
```

Implementation uses `mmap` for zero-copy tensor access. Parse the JSON header with nlohmann/json.

- [ ] **Step 4: Run tests to verify they pass**

Expected: Tests pass (or skip if no model available).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -s -m "feat: implement safetensors reader with mmap for zero-copy weight loading"
```

---

## Task 11: Scheduler (Continuous Batching + Chunked Prefill)

**Files:**
- Create: `core/include/core/scheduler.h`
- Modify: `core/src/scheduler.cpp`
- Create: `tests/test_scheduler.cpp`

- [ ] **Step 1: Write tests for scheduler**

```cpp
// tests/test_scheduler.cpp
#include <gtest/gtest.h>
#include "core/scheduler.h"

// Helper: create a scheduler with a block pool and prefix cache for testing
struct SchedulerTestFixture : public ::testing::Test {
    BlockPool block_pool{100, 4096};  // 100 blocks, 4096 bytes each
    PrefixCache prefix_cache{16};     // block_size = 16 tokens
    SchedulerConfig cfg;

    std::unique_ptr<Scheduler> make_scheduler() {
        return std::make_unique<Scheduler>(cfg, block_pool, prefix_cache);
    }
};

TEST_F(SchedulerTestFixture, EnqueueAndFormBatch) {
    cfg.max_batch_size = 4;
    cfg.max_tokens_per_batch = 512;
    auto sched = make_scheduler();

    // Enqueue 2 requests
    GenerationRequest req1;
    req1.request_id = 1;
    req1.prompt_tokens = std::vector<int32_t>(100, 1);
    req1.max_new_tokens = 50;

    GenerationRequest req2;
    req2.request_id = 2;
    req2.prompt_tokens = std::vector<int32_t>(200, 2);
    req2.max_new_tokens = 50;

    sched->enqueue(req1);
    sched->enqueue(req2);

    auto batch = sched->form_batch();
    // Both should be scheduled for prefill
    EXPECT_EQ(batch.prefill_requests.size(), 2u);
    EXPECT_EQ(batch.decode_requests.size(), 0u);
    // Block tables should be populated
    EXPECT_FALSE(batch.prefill_requests[0].block_table.empty());
    EXPECT_FALSE(batch.prefill_requests[1].block_table.empty());
}

TEST_F(SchedulerTestFixture, BlockAllocationAndRelease) {
    cfg.max_batch_size = 4;
    cfg.max_tokens_per_batch = 512;
    cfg.eos_token_id = 2;
    auto sched = make_scheduler();

    int initial_free = block_pool.num_free_blocks();

    GenerationRequest req;
    req.request_id = 1;
    req.prompt_tokens = std::vector<int32_t>(32, 1);  // 32 tokens = 2 blocks (block_size=16)
    req.max_new_tokens = 10;
    sched->enqueue(req);

    auto batch = sched->form_batch();
    // Should have allocated at least 2 blocks
    EXPECT_LT(block_pool.num_free_blocks(), initial_free);

    // Complete the request with EOS
    sched->update_after_step(batch, {{1, 2}});
    sched->get_completed();

    // Blocks should be returned (or donated to prefix cache)
    // Either way, they are accounted for
    EXPECT_GE(block_pool.num_free_blocks(), initial_free - 2);
}

TEST_F(SchedulerTestFixture, PrefixCacheReuse) {
    cfg.max_batch_size = 4;
    cfg.max_tokens_per_batch = 512;
    cfg.eos_token_id = 2;
    auto sched = make_scheduler();

    // Send first request with a specific prefix
    std::vector<int32_t> shared_prefix(32, 42);  // 32 tokens = 2 blocks
    GenerationRequest req1;
    req1.request_id = 1;
    req1.prompt_tokens = shared_prefix;
    req1.max_new_tokens = 5;
    sched->enqueue(req1);

    // Process and complete req1
    auto batch = sched->form_batch();
    sched->update_after_step(batch, {{1, 2}});
    sched->get_completed();  // req1 completes, blocks donated to prefix cache

    int free_before_req2 = block_pool.num_free_blocks();

    // Send second request with same prefix
    GenerationRequest req2;
    req2.request_id = 2;
    req2.prompt_tokens = shared_prefix;
    req2.max_new_tokens = 5;
    sched->enqueue(req2);

    auto batch2 = sched->form_batch();
    // req2 should reuse cached blocks — no new allocation for prefix
    EXPECT_EQ(block_pool.num_free_blocks(), free_before_req2);
    // prefill_start_pos should skip the cached prefix
    EXPECT_EQ(batch2.prefill_requests[0].prefill_start_pos, 32);
}

TEST_F(SchedulerTestFixture, ChunkedPrefill) {
    cfg.max_batch_size = 4;
    cfg.max_tokens_per_batch = 128;  // small budget
    auto sched = make_scheduler();

    GenerationRequest req;
    req.request_id = 1;
    req.prompt_tokens = std::vector<int32_t>(300, 1);  // needs chunking
    req.max_new_tokens = 50;
    sched->enqueue(req);

    auto batch1 = sched->form_batch();
    EXPECT_EQ(batch1.prefill_requests.size(), 1u);
    EXPECT_EQ(batch1.prefill_requests[0].prefill_chunk_len, 128);

    // Simulate completion of chunk, request still prefilling
    sched->update_after_step(batch1, {});

    auto batch2 = sched->form_batch();
    EXPECT_EQ(batch2.prefill_requests.size(), 1u);
    EXPECT_EQ(batch2.prefill_requests[0].prefill_start_pos, 128);
}

TEST_F(SchedulerTestFixture, ContinuousBatching) {
    cfg.max_batch_size = 4;
    cfg.max_tokens_per_batch = 512;
    auto sched = make_scheduler();

    // Start one request, let it enter decode phase
    GenerationRequest req1;
    req1.request_id = 1;
    req1.prompt_tokens = std::vector<int32_t>(10, 1);
    req1.max_new_tokens = 50;
    sched->enqueue(req1);

    auto batch1 = sched->form_batch();
    sched->update_after_step(batch1, {/*sampled token for req1*/});

    // Now add a second request — it should join the batch
    GenerationRequest req2;
    req2.request_id = 2;
    req2.prompt_tokens = std::vector<int32_t>(10, 2);
    req2.max_new_tokens = 50;
    sched->enqueue(req2);

    auto batch2 = sched->form_batch();
    EXPECT_EQ(batch2.prefill_requests.size(), 1u);   // req2 prefill
    EXPECT_EQ(batch2.decode_requests.size(), 1u);     // req1 decode
}

TEST_F(SchedulerTestFixture, StopOnEOS) {
    cfg.max_batch_size = 4;
    cfg.max_tokens_per_batch = 512;
    cfg.eos_token_id = 2;
    auto sched = make_scheduler();

    GenerationRequest req;
    req.request_id = 1;
    req.prompt_tokens = {1};
    req.max_new_tokens = 100;
    sched->enqueue(req);

    auto batch = sched->form_batch();
    sched->update_after_step(batch, {/*sampled token*/});  // not EOS

    batch = sched->form_batch();
    // Simulate EOS token
    sched->update_after_step(batch, {{1, 2}});  // request 1 got EOS token 2

    auto completed = sched->get_completed();
    EXPECT_EQ(completed.size(), 1u);
    EXPECT_EQ(completed[0].request_id, 1);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Expected: FAIL — `Scheduler` not defined.

- [ ] **Step 3: Implement Scheduler**

```cpp
// core/include/core/scheduler.h
#pragma once
#include <mutex>
#include <vector>
#include <deque>
#include <unordered_map>
#include "core/types.h"
#include "core/memory.h"
#include "core/sampling.h"
#include "models/model_interface.h"

struct GenerationRequest {
    int64_t request_id;
    std::vector<int32_t> prompt_tokens;
    int max_new_tokens;
    SamplingParams sampling_params;
};

struct SchedulerConfig {
    int max_batch_size;
    int max_tokens_per_batch;
    int eos_token_id = -1;
    std::vector<int> stop_token_ids;
    int kv_block_size = 16;         // tokens per KV cache block
};

struct ScheduledBatch {
    std::vector<RequestContext> prefill_requests;
    std::vector<RequestContext> decode_requests;
};

struct CompletedRequest {
    int64_t request_id;
    std::vector<int32_t> output_tokens;
};

class Scheduler {
public:
    // Scheduler owns references to BlockPool and PrefixCache for block management.
    // It allocates blocks when admitting requests and frees them on completion.
    Scheduler(SchedulerConfig config, BlockPool& block_pool, PrefixCache& prefix_cache);

    // Thread-safe: called from HTTP server threads
    void enqueue(GenerationRequest req);

    // Called from main loop thread only
    ScheduledBatch form_batch();
    void update_after_step(const ScheduledBatch& batch,
                           const std::vector<std::pair<int64_t, int32_t>>& sampled_tokens);
    std::vector<CompletedRequest> get_completed();

private:
    SchedulerConfig config_;
    BlockPool& block_pool_;
    PrefixCache& prefix_cache_;

    // Thread-safe pending queue (HTTP threads write, main loop reads)
    std::mutex pending_mutex_;
    std::deque<GenerationRequest> pending_;

    struct ActiveRequest {
        int64_t request_id;
        std::vector<int32_t> all_tokens;   // prompt + generated
        int prefill_pos;                    // how far prefill has progressed
        bool prefill_done;
        int generated_count;
        int max_new_tokens;
        std::vector<int> block_table;       // block indices from BlockPool
        SamplingParams sampling_params;
    };
    std::unordered_map<int64_t, ActiveRequest> active_;
    std::vector<CompletedRequest> completed_;

    // Allocate KV cache blocks for a new request.
    // First checks PrefixCache for reusable prefix blocks, then allocates
    // new blocks from BlockPool for remaining tokens.
    // Returns false if insufficient blocks (request stays in pending).
    bool allocate_blocks_for_request(ActiveRequest& req);

    // Free blocks when request completes. Inserts completed prefix into PrefixCache.
    void release_blocks(ActiveRequest& req);
};
```

Implementation:
- `enqueue()`: acquires `pending_mutex_`, pushes to `pending_` queue. Thread-safe.
- `form_batch()`:
  1. Moves requests from `pending_` (under lock) to local queue
  2. For each pending request: check prefix cache for reusable blocks (`prefix_cache_.lookup(tokens)`), set `prefill_start_pos` to matched tokens, `add_ref` on matched blocks
  3. Allocate new blocks from `block_pool_` for unmatched tokens. If insufficient blocks, skip this request (leave in pending).
  4. Populate `RequestContext::block_table` with prefix blocks + new blocks
  5. Budget prefill tokens (chunked) and decode tokens within `max_tokens_per_batch`
- `update_after_step()`: advances prefill positions, appends generated tokens, allocates additional blocks as sequences grow, checks EOS/stop/max_tokens
- `release_blocks()`: on completion, inserts the full token sequence into prefix cache, releases ref counts, frees any unshared blocks
- `get_completed()`: returns and clears completed queue

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd build && cmake .. && make -j && ctest --output-on-failure`
Expected: All scheduler tests pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -s -m "feat: implement continuous batching scheduler with chunked prefill"
```

---

## Task 12: LLaMA Model Module

**Files:**
- Modify: `models/llama/llama_model.h`, `models/llama/llama_model.cpp`
- Modify: `models/llama/llama_runtime.h`, `models/llama/llama_runtime.cpp`
- Create: `tests/test_llama_model.cpp`

- [ ] **Step 1: Write test for LLaMA model config and weight loading**

```cpp
TEST(LlamaModelTest, ConfigFromHuggingFace) {
    const char* model_path = std::getenv("TEST_MODEL_PATH");
    if (!model_path) GTEST_SKIP() << "TEST_MODEL_PATH not set";

    CublasProvider kernels;
    auto model = create_llama_model(&kernels);

    model->load_weights(model_path);
    auto cfg = model->config();

    EXPECT_GT(cfg.num_layers, 0);
    EXPECT_GT(cfg.hidden_size, 0);
    EXPECT_GT(cfg.vocab_size, 0);
}
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — `create_llama_model` not defined.

- [ ] **Step 3: Implement LLaMA model — weight loading and config**

`llama_model.h / llama_model.cpp`:

```cpp
// models/llama/llama_model.h
#pragma once
#include "models/model_interface.h"
#include "kernels/kernel_provider.h"
#include "core/safetensors.h"

class LlamaModel : public ModelModule {
public:
    explicit LlamaModel(KernelProvider* kernels);

    ModelConfig config() const override;
    void load_weights(const std::string& weight_path) override;
    KVCacheConfig kv_cache_config() const override;
    ForwardResult prefill(const std::vector<RequestContext>& requests,
                          cudaStream_t stream) override;
    ForwardResult decode(const std::vector<RequestContext>& requests,
                         cudaStream_t stream) override;
    int max_batch_size() const override;
    int max_tokens_per_batch() const override;

private:
    KernelProvider* kernels_;
    ModelConfig config_;

    // Weights on GPU (all stored as contiguous float16/bfloat16 tensors)
    void* embed_tokens_ = nullptr;       // [vocab_size, hidden_size]
    void* lm_head_ = nullptr;            // [vocab_size, hidden_size]
    void* final_norm_weight_ = nullptr;  // [hidden_size]

    struct LayerWeights {
        void* input_norm;     // [hidden_size]
        void* post_attn_norm; // [hidden_size]
        void* q_proj;         // [num_heads * head_dim, hidden_size]
        void* k_proj;         // [num_kv_heads * head_dim, hidden_size]
        void* v_proj;         // [num_kv_heads * head_dim, hidden_size]
        void* o_proj;         // [hidden_size, num_heads * head_dim]
        void* gate_proj;      // [intermediate_size, hidden_size]
        void* up_proj;        // [intermediate_size, hidden_size]
        void* down_proj;      // [hidden_size, intermediate_size]
    };
    std::vector<LayerWeights> layers_;

    // Workspace buffers (allocated once during load_weights)
    void* workspace_ = nullptr;   // scratch space for intermediates
    size_t workspace_size_ = 0;

    int head_dim() const { return config_.hidden_size / config_.num_attention_heads; }

    // Internal forward: processes tokens through the decoder stack
    // token_ids: flattened token IDs for all requests in the batch
    // positions: position index per token (for RoPE)
    // block_tables: KV cache block tables, one per request
    // seq_lens: current total sequence length per request
    // is_prefill: if true, writes K,V to cache; if false, reads existing K,V
    void forward_batch(
        const int32_t* token_ids, const int* positions,
        const std::vector<std::vector<int>>& block_tables,
        const int* seq_lens, int total_tokens, int batch_size,
        bool is_prefill, float* output_logits,
        cudaStream_t stream);
};

std::unique_ptr<ModelModule> create_llama_model(KernelProvider* kernels);
```

`load_weights()` implementation:
- Read HF `config.json` (nlohmann/json) to populate `ModelConfig` fields:
  `num_hidden_layers`, `hidden_size`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`, `vocab_size`, `max_position_embeddings`, `eos_token_id`
- Open all `model-*.safetensors` files via `SafetensorsFile`
- For each tensor, `cudaMalloc` + `cudaMemcpy` from mmap'd data to GPU
- HF weight name mapping:
  - `model.embed_tokens.weight` → `embed_tokens_`
  - `model.layers.{i}.input_layernorm.weight` → `layers_[i].input_norm`
  - `model.layers.{i}.self_attn.q_proj.weight` → `layers_[i].q_proj`
  - `model.layers.{i}.self_attn.k_proj.weight` → `layers_[i].k_proj`
  - `model.layers.{i}.self_attn.v_proj.weight` → `layers_[i].v_proj`
  - `model.layers.{i}.self_attn.o_proj.weight` → `layers_[i].o_proj`
  - `model.layers.{i}.mlp.gate_proj.weight` → `layers_[i].gate_proj`
  - `model.layers.{i}.mlp.up_proj.weight` → `layers_[i].up_proj`
  - `model.layers.{i}.mlp.down_proj.weight` → `layers_[i].down_proj`
  - `model.layers.{i}.post_attention_layernorm.weight` → `layers_[i].post_attn_norm`
  - `model.norm.weight` → `final_norm_weight_`
  - `lm_head.weight` → `lm_head_`
- Allocate workspace: `max_batch_size * max_seq_len * hidden_size * dtype_size * 4` (room for Q, K, V, intermediates)

- [ ] **Step 4: Run tests to verify config loading passes**

Run: `cd build && cmake .. && make -j && ctest --output-on-failure`
Expected: Config test passes with a real model directory.

- [ ] **Step 5: Implement forward pass**

The `forward_batch()` method processes all requests in one pass:

```
// Pseudocode for forward_batch:
// Input: token_ids[total_tokens], positions[total_tokens]

// 1. Embedding lookup (gather, not GEMM)
//    For each token t: hidden[t] = embed_tokens_[token_ids[t]]
//    This is a simple GPU gather kernel: one thread per (token, hidden_dim) element.

// 2. For each layer i = 0..num_layers-1:
//    a. RMSNorm: normed = rms_norm(hidden, layers_[i].input_norm)
//
//    b. Q/K/V projections (three GEMMs):
//       Q = normed @ q_proj.T    → [total_tokens, num_heads * head_dim]
//       K = normed @ k_proj.T    → [total_tokens, num_kv_heads * head_dim]
//       V = normed @ v_proj.T    → [total_tokens, num_kv_heads * head_dim]
//
//    c. RoPE: apply rotary embeddings to Q and K using positions[]
//       Uses theta=10000.0 (or from config rope_theta)
//
//    d. KV cache write (paged):
//       For each request r, for each token position p in this batch:
//         block_idx = block_tables[r][p / block_size]
//         offset = p % block_size
//         Write K[token] and V[token] into the block at the correct offset.
//       This is a scatter kernel: given block_table and positions, write K,V
//       into the correct locations in the global KV cache pool.
//
//    e. Attention:
//       Call kernels_->fused_attention() with Q, k_cache, v_cache,
//       block_tables, seq_lens. Output: attn_output[total_tokens, hidden_size]
//
//    f. O projection: output = attn_output @ o_proj.T
//    g. Residual: hidden = hidden + output
//
//    h. Post-attention norm: normed = rms_norm(hidden, layers_[i].post_attn_norm)
//
//    i. MLP (SwiGLU):
//       gate = normed @ gate_proj.T   → [total_tokens, intermediate_size]
//       up   = normed @ up_proj.T     → [total_tokens, intermediate_size]
//       fused = silu(gate) * up       (fused SwiGLU kernel)
//       down = fused @ down_proj.T    → [total_tokens, hidden_size]
//
//    j. Residual: hidden = hidden + down

// 3. Final RMSNorm: hidden = rms_norm(hidden, final_norm_weight_)

// 4. LM head: logits = hidden @ lm_head_.T  → [total_tokens, vocab_size]
//    For decode, total_tokens == batch_size (one token per request)
//    For prefill, only the last token per request's logits are needed
//    → extract logits for the last token of each request into output
```

**Key detail — embedding lookup**: Not a GEMM, but a gather. Add a small CUDA kernel `launch_embedding_lookup(token_ids, embed_table, output, num_tokens, hidden_size, stream)` in `kernels/src/`. One thread per element, reads `embed_table[token_id * hidden_size + dim]`.

**Key detail — KV cache write**: A scatter kernel writes K, V values into paged blocks. Add `launch_kv_cache_scatter(K, V, k_cache, v_cache, block_table, positions, ...)` in `kernels/src/`. For each token, computes the target block and offset from `block_table` and `position`, then writes K and V.

**Key detail — position tracking for chunked prefill**: The `RequestContext` carries `prefill_start_pos`. Positions for RoPE are computed as `prefill_start_pos + local_offset` for each token in the chunk. This ensures RoPE embeddings are correct across chunks.

**Key detail — decode vs prefill**: In decode, each request contributes exactly 1 token. The attention kernel reads the full KV cache (all previous positions) but Q has only 1 row. In prefill, Q has multiple rows (the chunk) and K, V are both written and read.

- [ ] **Step 6: Write forward pass shape test**

```cpp
TEST(LlamaModelTest, ForwardPassShape) {
    const char* model_path = std::getenv("TEST_MODEL_PATH");
    if (!model_path) GTEST_SKIP() << "TEST_MODEL_PATH not set";

    CublasProvider kernels;
    auto model = create_llama_model(&kernels);
    model->load_weights(model_path);
    auto cfg = model->config();

    // Set up a minimal request with block table
    RequestContext req;
    req.request_id = 1;
    req.token_ids = {1, 2, 3, 4, 5};
    req.seq_len = 5;
    req.prefill_start_pos = 0;
    req.prefill_chunk_len = 5;
    // Allocate blocks (need ceil(5/block_size) blocks)
    // For test, use a small BlockPool
    BlockPool pool(10, /* block_size_bytes computed from model config */);
    for (int i = 0; i < 1; i++) {  // 1 block for 5 tokens with block_size=16
        auto blk = pool.allocate();
        req.block_table.push_back(blk.value());
    }
    req.max_new_tokens = 10;

    auto result = model->prefill({req}, nullptr);
    // Should return logits for 1 request * vocab_size
    EXPECT_EQ(result.logits.size(), static_cast<size_t>(cfg.vocab_size));
    EXPECT_EQ(result.vocab_size, cfg.vocab_size);
}
```

- [ ] **Step 7: Run tests**

Run: `cd build && cmake .. && make -j && ctest --output-on-failure`
Expected: Shape test passes.

- [ ] **Step 8: Implement kv_cache_config and scheduling hints**

`llama_runtime.h / llama_runtime.cpp` (integrated into `LlamaModel`):
- `kv_cache_config()`:
  - `block_size = 16`
  - Per-block memory: `2 * num_layers * num_kv_heads * block_size * head_dim * dtype_size(dtype)`
  - `max_blocks` = `(available_gpu_memory * 0.9 - model_weight_memory) / per_block_memory`
  - `cache_dtype` = model dtype
- `max_batch_size()`: `256` for prototype (tunable)
- `max_tokens_per_batch()`: `2048` for prototype

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -s -m "feat: implement LLaMA model module with forward pass and weight loading"
```

---

## Task 13: HTTP Server (OpenAI-compatible)

**Files:**
- Create: `core/include/core/server.h`
- Modify: `core/src/server.cpp`
- Create: `tests/test_server.cpp`

- [ ] **Step 1: Write tests for server request parsing**

```cpp
TEST(ServerTest, ParseChatCompletionRequest) {
    std::string json_body = R"({
        "model": "llama-3-8b",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"}
        ],
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": true
    })";

    auto req = parse_chat_completion_request(json_body);
    EXPECT_EQ(req.model, "llama-3-8b");
    EXPECT_EQ(req.messages.size(), 2u);
    EXPECT_EQ(req.max_tokens, 100);
    EXPECT_NEAR(req.temperature, 0.7f, 1e-6);
    EXPECT_TRUE(req.stream);
}

TEST(ServerTest, FormatSSEChunk) {
    auto chunk = format_sse_chunk("llama-3-8b", "Hello", /*finish_reason=*/"");
    EXPECT_TRUE(chunk.find("data: ") == 0);
    EXPECT_TRUE(chunk.find("\"Hello\"") != std::string::npos);
}

TEST(ServerTest, FormatSSEDone) {
    auto done = format_sse_done("llama-3-8b", "stop");
    EXPECT_TRUE(done.find("\"finish_reason\":\"stop\"") != std::string::npos);
    EXPECT_TRUE(done.find("[DONE]") != std::string::npos);
}

TEST(ServerTest, ParseCompletionRequest) {
    std::string json_body = R"({
        "model": "llama-3-8b",
        "prompt": "The capital of France is",
        "max_tokens": 50,
        "temperature": 0.5
    })";

    auto req = parse_completion_request(json_body);
    EXPECT_EQ(req.model, "llama-3-8b");
    EXPECT_EQ(req.prompt, "The capital of France is");
    EXPECT_EQ(req.max_tokens, 50);
    EXPECT_NEAR(req.temperature, 0.5f, 1e-6);
}

TEST(ServerTest, FormatModelsResponse) {
    auto json = format_models_response("llama-3-8b");
    EXPECT_TRUE(json.find("\"id\":\"llama-3-8b\"") != std::string::npos);
    EXPECT_TRUE(json.find("\"object\":\"list\"") != std::string::npos);
}
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — functions not defined.

- [ ] **Step 3: Implement server**

```cpp
// core/include/core/server.h
#pragma once
#include <string>
#include <functional>
#include <vector>
#include "core/types.h"
#include "core/sampling.h"

struct ChatMessage {
    std::string role;
    std::string content;
};

struct ChatCompletionRequest {
    std::string model;
    std::vector<ChatMessage> messages;
    int max_tokens = 256;
    float temperature = 1.0f;
    float top_p = 1.0f;
    int top_k = 0;
    bool stream = false;
};

struct CompletionRequest {
    std::string model;
    std::string prompt;
    int max_tokens = 256;
    float temperature = 1.0f;
    float top_p = 1.0f;
    int top_k = 0;
    bool stream = false;
};

ChatCompletionRequest parse_chat_completion_request(const std::string& json_body);
CompletionRequest parse_completion_request(const std::string& json_body);
std::string format_sse_chunk(const std::string& model, const std::string& token,
                              const std::string& finish_reason);
std::string format_sse_done(const std::string& model, const std::string& finish_reason);
std::string format_models_response(const std::string& model_id);

// Callback: server calls this with prompt tokens + params, gets back tokens via stream_cb
using GenerateCallback = std::function<void(
    const std::vector<int32_t>& prompt_tokens,
    const SamplingParams& params, int max_tokens,
    std::function<void(const std::string& token, bool done)> stream_cb)>;

class Server {
public:
    Server(const std::string& host, int port, GenerateCallback callback);
    void start();   // blocking
    void stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
```

Implementation uses cpp-httplib:
- `POST /v1/chat/completions` — parses request, applies chat template, tokenizes, calls generate callback
- `POST /v1/completions` — parses request, tokenizes prompt directly, calls generate callback
- For streaming: uses `httplib::Response::set_chunked_content_provider` for SSE
- `GET /v1/models` — returns model info JSON via `format_models_response()`
- `GET /health` — returns 200 (simple health check)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd build && cmake .. && make -j && ctest --output-on-failure`
Expected: Parse/format tests pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -s -m "feat: implement OpenAI-compatible HTTP server with SSE streaming"
```

---

## Task 14: Main Application — Wire Everything Together

**Files:**
- Modify: `app/main.cpp`
- Create: `tests/test_integration.cpp`

**Threading model**: The HTTP server runs on its own threads (managed by cpp-httplib). The main inference loop runs on a dedicated background thread. The only shared state is the `Scheduler`, whose `enqueue()` method is thread-safe (protected by mutex). All other components (`model`, `sampler`, `BlockPool`, `PrefixCache`) are accessed only from the main loop thread.

- [ ] **Step 1: Implement main.cpp**

```cpp
// app/main.cpp
#include <atomic>
#include <iostream>
#include <thread>
#include <string>
#include "core/server.h"
#include "core/scheduler.h"
#include "core/tokenizer.h"
#include "core/sampling.h"
#include "core/memory.h"
#include "kernels/cublas_provider.h"
#include "models/model_interface.h"

extern std::unique_ptr<ModelModule> create_llama_model(KernelProvider* kernels);

int main(int argc, char** argv) {
    // Parse CLI args: --model-dir, --port, --host
    std::string model_dir, host = "0.0.0.0";
    int port = 8000;
    // ... parse args ...

    // 1. Create kernel provider
    CublasProvider kernels;

    // 2. Create and load model
    auto model = create_llama_model(&kernels);
    model->load_weights(model_dir);
    auto model_config = model->config();

    // 3. Create tokenizer
    Tokenizer tokenizer;
    tokenizer.load(model_dir + "/tokenizer.model");

    // 4. Create memory pool + prefix cache
    auto kv_cfg = model->kv_cache_config();
    size_t block_size_bytes = 2 * model_config.num_layers
        * (model_config.num_kv_heads)
        * kv_cfg.block_size
        * (model_config.hidden_size / model_config.num_attention_heads)
        * dtype_size(model_config.dtype);
    BlockPool block_pool(kv_cfg.max_blocks, block_size_bytes);
    PrefixCache prefix_cache(kv_cfg.block_size);

    // 5. Create scheduler (takes references to block_pool and prefix_cache)
    SchedulerConfig sched_cfg;
    sched_cfg.max_batch_size = model->max_batch_size();
    sched_cfg.max_tokens_per_batch = model->max_tokens_per_batch();
    sched_cfg.eos_token_id = model_config.eos_token_id;
    sched_cfg.stop_token_ids = model_config.stop_token_ids;
    sched_cfg.kv_block_size = kv_cfg.block_size;
    Scheduler scheduler(sched_cfg, block_pool, prefix_cache);

    // 6. Create sampler
    Sampler sampler;

    // 7. Request-to-token streaming map (for SSE callbacks)
    // Maps request_id → stream callback. Protected by mutex.
    std::mutex stream_map_mutex;
    std::unordered_map<int64_t, std::function<void(const std::string&, bool)>> stream_map;
    std::atomic<int64_t> next_request_id{1};

    // 8. Main inference loop (background thread)
    std::atomic<bool> running{true};
    cudaStream_t cuda_stream;
    cudaStreamCreate(&cuda_stream);

    std::thread inference_thread([&]() {
        while (running) {
            auto batch = scheduler.form_batch();
            bool has_work = !batch.prefill_requests.empty() || !batch.decode_requests.empty();
            if (!has_work) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                continue;
            }

            // Run prefill
            if (!batch.prefill_requests.empty()) {
                auto result = model->prefill(batch.prefill_requests, cuda_stream);
                // Sample tokens for requests that completed prefill (last chunk)
                std::vector<std::pair<int64_t, int32_t>> sampled;
                for (size_t i = 0; i < batch.prefill_requests.size(); i++) {
                    auto& req = batch.prefill_requests[i];
                    // Only sample if this is the last prefill chunk
                    if (req.prefill_start_pos + req.prefill_chunk_len >= req.seq_len) {
                        float* logits = result.logits.data() + i * result.vocab_size;
                        auto token = sampler.sample(logits, result.vocab_size, {});
                        sampled.push_back({req.request_id, token});

                        // Stream the token
                        std::lock_guard<std::mutex> lock(stream_map_mutex);
                        if (auto it = stream_map.find(req.request_id); it != stream_map.end()) {
                            it->second(tokenizer.decode({token}), false);
                        }
                    }
                }
                scheduler.update_after_step(batch, sampled);
            }

            // Run decode
            if (!batch.decode_requests.empty()) {
                auto result = model->decode(batch.decode_requests, cuda_stream);
                std::vector<std::pair<int64_t, int32_t>> sampled;
                for (size_t i = 0; i < batch.decode_requests.size(); i++) {
                    float* logits = result.logits.data() + i * result.vocab_size;
                    auto token = sampler.sample(logits, result.vocab_size, {});
                    sampled.push_back({batch.decode_requests[i].request_id, token});

                    std::lock_guard<std::mutex> lock(stream_map_mutex);
                    if (auto it = stream_map.find(batch.decode_requests[i].request_id);
                        it != stream_map.end()) {
                        it->second(tokenizer.decode({token}), false);
                    }
                }
                scheduler.update_after_step(batch, sampled);
            }

            // Notify completed requests
            for (auto& completed : scheduler.get_completed()) {
                std::lock_guard<std::mutex> lock(stream_map_mutex);
                if (auto it = stream_map.find(completed.request_id); it != stream_map.end()) {
                    it->second("", true);  // signal done
                    stream_map.erase(it);
                }
            }
        }
    });

    // 9. Start server — generate callback enqueues to scheduler
    Server server(host, port, [&](const std::vector<int32_t>& prompt_tokens,
                                   const SamplingParams& params, int max_tokens,
                                   auto stream_cb) {
        auto id = next_request_id.fetch_add(1);
        {
            std::lock_guard<std::mutex> lock(stream_map_mutex);
            stream_map[id] = stream_cb;
        }
        GenerationRequest gen_req;
        gen_req.request_id = id;
        gen_req.prompt_tokens = prompt_tokens;
        gen_req.max_new_tokens = max_tokens;
        gen_req.sampling_params = params;
        scheduler.enqueue(gen_req);
    });

    std::cout << "LLM-Singularity serving on " << host << ":" << port << std::endl;
    server.start();  // blocking

    running = false;
    inference_thread.join();
    cudaStreamDestroy(cuda_stream);
    return 0;
}
```

- [ ] **Step 2: Write automated integration test (in-process, no server)**

```cpp
// tests/test_integration.cpp
// Tests the full pipeline without the HTTP server layer
TEST(IntegrationTest, FullPipelineInProcess) {
    const char* model_path = std::getenv("TEST_MODEL_PATH");
    if (!model_path) GTEST_SKIP() << "TEST_MODEL_PATH not set";

    // Build the full stack
    CublasProvider kernels;
    auto model = create_llama_model(&kernels);
    model->load_weights(model_path);
    auto cfg = model->config();
    auto kv_cfg = model->kv_cache_config();

    size_t block_bytes = /* compute from config */;
    BlockPool pool(kv_cfg.max_blocks, block_bytes);
    PrefixCache prefix_cache(kv_cfg.block_size);

    SchedulerConfig sched_cfg;
    sched_cfg.max_batch_size = 4;
    sched_cfg.max_tokens_per_batch = 512;
    sched_cfg.eos_token_id = cfg.eos_token_id;
    sched_cfg.kv_block_size = kv_cfg.block_size;
    Scheduler scheduler(sched_cfg, pool, prefix_cache);
    Sampler sampler;

    // Enqueue a request
    GenerationRequest req;
    req.request_id = 1;
    req.prompt_tokens = {1, 2, 3};  // arbitrary tokens
    req.max_new_tokens = 5;
    scheduler.enqueue(req);

    // Run inference loop
    std::vector<int32_t> output;
    for (int step = 0; step < 20 && output.empty(); ++step) {
        auto batch = scheduler.form_batch();
        if (!batch.prefill_requests.empty()) {
            auto result = model->prefill(batch.prefill_requests, nullptr);
            std::vector<std::pair<int64_t, int32_t>> sampled;
            for (size_t i = 0; i < batch.prefill_requests.size(); i++) {
                float* logits = result.logits.data() + i * result.vocab_size;
                sampled.push_back({batch.prefill_requests[i].request_id,
                                   sampler.sample(logits, result.vocab_size, {})});
            }
            scheduler.update_after_step(batch, sampled);
        }
        if (!batch.decode_requests.empty()) {
            auto result = model->decode(batch.decode_requests, nullptr);
            std::vector<std::pair<int64_t, int32_t>> sampled;
            for (size_t i = 0; i < batch.decode_requests.size(); i++) {
                float* logits = result.logits.data() + i * result.vocab_size;
                sampled.push_back({batch.decode_requests[i].request_id,
                                   sampler.sample(logits, result.vocab_size, {})});
            }
            scheduler.update_after_step(batch, sampled);
        }
        for (auto& c : scheduler.get_completed()) {
            output = c.output_tokens;
        }
    }

    EXPECT_GT(output.size(), 0u);
    EXPECT_LE(output.size(), 5u);
}
```

- [ ] **Step 3: Verify it compiles**

Run: `cd build && cmake .. && make -j`
Expected: `llm-serve-llama` binary produced, test binary includes integration test.

- [ ] **Step 4: Smoke test with a real model (manual)**

Run:
```bash
./llm-serve-llama --model-dir /path/to/Llama-3.2-1B --port 8000
```
In another terminal:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama","messages":[{"role":"user","content":"Hello"}],"max_tokens":20}'
```
Expected: JSON response with generated text.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -s -m "feat: wire together main application with CLI, model loading, and serving"
```

---

## Task 15: End-to-End Integration Test

**Files:**
- Create: `tests/test_e2e.cpp`

- [ ] **Step 1: Write end-to-end test**

```cpp
// tests/test_e2e.cpp
// This test requires a model directory (set TEST_MODEL_PATH).
// It spins up the full stack in-process and verifies generation.

TEST(E2ETest, GenerateSingleRequest) {
    const char* model_path = std::getenv("TEST_MODEL_PATH");
    if (!model_path) GTEST_SKIP() << "TEST_MODEL_PATH not set";

    // Create full stack: kernels → model → tokenizer → scheduler → sampler
    CublasProvider kernels;
    auto model = create_llama_model(&kernels);
    model->load_weights(model_path);

    Tokenizer tok;
    tok.load(std::string(model_path) + "/tokenizer.model");

    Scheduler sched(/* config from model */);
    Sampler sampler;

    // Enqueue a request
    auto tokens = tok.encode("The capital of France is");
    GenerationRequest req;
    req.request_id = 1;
    req.prompt_tokens = tokens;
    req.max_new_tokens = 10;
    sched.enqueue(req);

    // Run generation loop
    std::vector<int32_t> output;
    for (int step = 0; step < 20 && output.size() < 10; ++step) {
        auto batch = sched.form_batch();
        if (!batch.prefill_requests.empty()) {
            auto result = model->prefill(batch.prefill_requests, nullptr);
            // sample and update...
        }
        if (!batch.decode_requests.empty()) {
            auto result = model->decode(batch.decode_requests, nullptr);
            // sample and update...
        }
        for (auto& c : sched.get_completed()) {
            output = c.output_tokens;
        }
    }

    auto text = tok.decode(output);
    // Should contain something meaningful
    EXPECT_GT(text.size(), 0u);
    std::cout << "Generated: " << text << std::endl;
}
```

- [ ] **Step 2: Run the test**

Run: `TEST_MODEL_PATH=/path/to/Llama-3.2-1B cd build && ctest --output-on-failure -R E2E`
Expected: Generates coherent text.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -s -m "test: add end-to-end integration test for full generation pipeline"
```

---

## Task 16: Push to GitHub

- [ ] **Step 1: Push all commits to origin/main**

```bash
cd /home/suyogg/dev/llm-singularity
git push -u origin main
```

- [ ] **Step 2: Verify repo on GitHub**

Run: `gh repo view suyoggupta/llm-singularity --web`
