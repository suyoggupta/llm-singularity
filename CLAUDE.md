# CLAUDE.md — llm-singularity

Standalone C++ LLM inference server. One binary per model (`llm-serve-llama`), OpenAI-compatible REST API, paged KV cache, continuous batching, prefix caching. Float32 prototype — fp16/TP are future work.

## Build

```bash
mkdir build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DMODEL=llama [-DBUILD_TESTS=ON]
ninja -j$(nproc)
```

**`MODEL`** selects which model library to compile and link (currently only `llama`).
**`BUILD_TESTS`** (default `ON`) compiles the `tests` executable and `debug_dump` tool.

Dependencies are auto-fetched via CMake FetchContent (cpp-httplib, nlohmann/json, sentencepiece, googletest). Requires CUDA 12.x and CMake 3.24+.

## Run

```bash
./app/llm-serve-llama --model-dir /path/to/model   # local safetensors dir
./app/llm-serve-llama --model-dir meta-llama/Llama-3.1-8B-Instruct  # HF auto-download
```

Use `-Instruct` variants (not base) — base models have no chat template and `apply_chat_template` will throw.

## Test

```bash
cd build && ctest --output-on-failure
# Model-dependent tests need a real checkpoint:
TEST_MODEL_PATH=/path/to/Llama-3.2-1B ctest --output-on-failure
```

Unit test source lives in `tests/test_*.cpp`. End-to-end: `tests/test_e2e.py`.

## Architecture

```
Request
  → HTTP Server (core/server.cpp, httplib)
  → Tokenizer (core/tokenizer.cpp, sentencepiece + Python tokenizer_worker.py)
  → Scheduler (core/scheduler.cpp, continuous batching + chunked prefill)
  → LlamaModel::prefill() / decode() (models/llama/llama_model.cpp)
      → KernelProvider (kernels/, cuBLAS GEMM + custom CUDA kernels)
      → Paged KV Cache (BlockPool, block table indirection)
  → Sampler (core/sampling.cpp, top-k/top-p/temperature)
  → SSE stream back to client
```

### Key Components

| Component | Files | Notes |
|-----------|-------|-------|
| HTTP server | `core/src/server.cpp` | pimpl pattern; OpenAI `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health` |
| Scheduler | `core/src/scheduler.cpp` | Continuous batching, chunked prefill, prefix cache lookup |
| Block pool | `core/src/memory.cpp` | GPU block allocator; `PrefixCache` = radix tree + LRU eviction |
| Tokenizer | `core/src/tokenizer.cpp` | sentencepiece C++ + Python subprocess for `apply_chat_template` |
| Sampler | `core/src/sampling.cpp` | temperature, top-k, top-p, repetition penalty |
| Safetensors | `core/src/safetensors.cpp` | mmap-backed HF weight reader |
| Kernel provider | `kernels/include/kernels/kernel_provider.h` | Abstract interface; swap cuBLAS for CUTLASS/Triton without touching model code |
| cuBLAS provider | `kernels/src/cublas_provider.cpp` | Concrete implementation |
| RoPE | `kernels/src/rope.cu` | Split-half convention `(i, i+head_dim/2)` — **not** interleaved `(2i, 2i+1)` |
| Paged attention | `kernels/src/attention.cu` | Block table indirection, GQA support |
| LLaMA model | `models/llama/llama_model.cpp` | Weights as flat GPU float* arrays; single `forward_impl()` for prefill + decode |
| App entrypoint | `app/main.cpp` | CLI parsing, HF download, wires all components together |

### Model Interface

Every model implements `ModelModule` (`models/include/models/model_interface.h`):

```cpp
class ModelModule {
    ModelConfig    config();
    KVCacheConfig  kv_cache_config();
    void           load_weights(const std::string& model_dir);
    ForwardResult  prefill(std::vector<RequestContext>, cudaStream_t);
    ForwardResult  decode(std::vector<RequestContext>, cudaStream_t);
};
```

Adding a new model = new directory under `models/`, implement `ModelModule`, register in `app/main.cpp`, add `MODEL=<name>` cmake option.

## Known Gotchas

### RoPE convention
LLaMA uses **split-half** pairs: `(i, i + head_dim/2)`. The interleaved convention `(2i, 2i+1)` is wrong for LLaMA. Position 0 masks the bug (`sin(0)=0, cos(0)=1`), so single-token tests pass even with wrong pairing — always verify with ≥2-token sequences against HF reference logits.

### Chat template on base models
`apply_chat_template` throws for base (non-instruct) models. The server wraps this in a try/catch and returns HTTP 500. Always use `-Instruct` checkpoints with `/v1/chat/completions`; for base models use `/v1/completions` with a raw `"prompt"` string.

### Streaming content provider threading
The httplib `set_chunked_content_provider` lambda runs on a different thread from where `res` was set up. Exceptions inside the lambda must be caught within the lambda — they do not propagate to the handler.

### Docker runtime image
`nvidia/cuda:12.6.3-runtime-ubuntu22.04` does **not** include cuBLAS. Must explicitly install `libcublas-12-6` in the runtime stage.

## Debugging

`tests/debug_dump.cpp` runs a forward pass and prints top-5 logits alongside HF reference values. Build it with `BUILD_TESTS=ON` and run:

```bash
./build/tests/debug_dump /path/to/model
```

Reference HF logits for Llama-3.1-8B-Instruct:
- 1-token `[128000]`: tok14924=11.74
- 2-token `[128000, 128006]`: tok128006=24.62
- 3-token `[128000, 128006, 9125]`: tok198=15.40

`tests/debug_*.py` — layer-by-layer scripts for comparing individual operations (embedding, GEMM, attention) against PyTorch.
