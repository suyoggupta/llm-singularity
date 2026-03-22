# LLM-Serve: Per-Model Bundled C++ LLM Serving System

**Date**: 2026-03-22
**Status**: Draft
**Author**: Suyog (design), Claude (spec generation)

## Motivation

TensorRT-LLM's current architecture separates a generic runtime (scheduler, KV cache manager, batch manager) from model definitions. The runtime works across all ~170 supported models but introduces complexity — multiple abstraction layers, Python orchestration over C++ components, and generic policies that can't exploit model-specific optimization opportunities.

LLM-Serve takes a different approach: for each model, generate a self-contained C++ bundle that fuses the model's forward logic with its runtime policy. Three forces make this viable now:

1. **Performance**: Model-specific runtime code unlocks optimizations a generic runtime cannot — specialized KV cache layouts, tighter scheduling heuristics, architecture-aware memory budgeting.
2. **Simplicity**: A per-model bundle is easier to reason about than a layered generic framework. Two files describe the entire model's behavior.
3. **AI-assisted development**: Claude can read a HuggingFace model config and generate the C++ model + runtime code, making the per-model approach economically viable despite apparent duplication.

## Architecture Overview

**Layered Library + Generated Model Module** (Approach B):

A small shared core library handles infrastructure concerns (HTTP serving, request queuing, tokenization, sampling, memory pooling). For each model, Claude generates a **model module** that implements the forward pass and runtime policy. The model module plugs into the shared core via a narrow interface.

```
┌─────────────────────────────────────────────┐
│              HTTP Server (REST)              │  core/server/
│         OpenAI-compatible endpoints          │
├─────────────────────────────────────────────┤
│              Scheduler                       │  core/scheduler/
│   continuous batching, chunked prefill,      │
│   prefix cache lookup, admission control     │
├──────────────────────┬──────────────────────┤
│   Sampling           │   Tokenizer          │  core/sampling/, core/tokenizer/
│   top-k/p, temp      │   SP/tiktoken        │
├──────────────────────┴──────────────────────┤
│              Memory Pool                     │  core/memory/
│   GPU allocator, prefix cache (radix tree)   │
├─────────────────────────────────────────────┤
│          Model Module Interface              │  models/model_interface.h
├─────────────────────────────────────────────┤
│  ┌─────────────────────────────────────┐    │
│  │  Generated Model Module (e.g. LLaMA)│    │  models/llama/
│  │  ├── model.cpp   (forward pass)     │    │
│  │  └── runtime.cpp (KV cache policy,  │    │
│  │       batching strategy, memory)    │    │
│  └─────────────────────────────────────┘    │
├─────────────────────────────────────────────┤
│           Kernel Interface                   │  kernels/interface/
├─────────────────────────────────────────────┤
│  CUTLASS/CuTe Kernels                       │  kernels/cutlass/
│  GEMM, FlashAttention, RMSNorm, SwiGLU, RoPE│
└─────────────────────────────────────────────┘
```

The final artifact is a **single binary per model** (e.g., `llm-serve-llama`).

## Repository Structure

```
llm-serve/
├── CMakeLists.txt                  # top-level CMake
├── core/                           # shared library layer
│   ├── server/                     # HTTP server (OpenAI-compatible REST)
│   ├── scheduler/                  # request queue, continuous batching
│   ├── tokenizer/                  # tokenizer interface (sentencepiece/tiktoken)
│   ├── sampling/                   # sampling strategies (top-k, top-p, temperature)
│   └── memory/                     # GPU memory pool, prefix cache (radix tree)
├── kernels/                        # kernel interface + implementations
│   ├── interface/                  # abstract kernel API headers
│   └── cutlass/                    # CUTLASS/CuTe implementations
├── models/                         # per-model generated modules
│   ├── model_interface.h           # the contract between core and model modules
│   └── llama/                      # Claude-generated LLaMA module
│       ├── model.h / model.cpp     # forward pass, layer definitions
│       └── runtime.h / runtime.cpp # KV cache policy, batching strategy
├── tools/                          # utilities
│   └── generate_model.py           # invokes Claude to generate a model module from HF config
└── third_party/                    # dependencies (httplib, nlohmann/json, sentencepiece, etc.)
```

## Model Module Interface

The critical boundary between shared core and generated model code.

```cpp
struct ModelConfig {
    int num_layers;
    int hidden_size;
    int num_attention_heads;
    int num_kv_heads;              // for GQA
    int intermediate_size;
    int vocab_size;
    int max_seq_len;
    int eos_token_id;              // end-of-sequence token
    std::vector<int> stop_token_ids; // additional stop tokens
    DataType dtype;
    std::unordered_map<std::string, std::string> extra;
};

struct KVCacheConfig {
    int block_size;                // tokens per block
    int max_blocks;                // total blocks in pool
    DataType cache_dtype;
};

// Per-request context passed to model forward methods.
// Core layer populates this; model module reads it.
struct RequestContext {
    int64_t request_id;
    std::vector<int32_t> token_ids;     // full token sequence (prefill) or single token (decode)
    int seq_len;                        // total sequence length so far
    int prefill_start_pos;              // where to start prefill (>0 if prefix cached)
    int prefill_chunk_len;              // tokens to process this iteration (chunked prefill)
    std::vector<int> block_table;       // KV cache block indices (managed by core/memory)
    int max_new_tokens;                 // per-request generation limit
    // Sampling params are not here — sampling is handled by core/sampling/
};

// Model forward pass returns logits only. Sampling is done by the core layer.
struct ForwardResult {
    // logits[i] is the logit vector for the i-th request in the batch
    // Shape: [batch_size, vocab_size]
    std::vector<float> logits;
};

// Factory function — each generated model module provides this.
// KernelProvider is injected here so the model can use it throughout its forward pass.
std::unique_ptr<ModelModule> create_model_module(KernelProvider* kernels);

class ModelModule {
public:
    virtual ~ModelModule() = default;

    // Initialization
    virtual ModelConfig config() const = 0;
    virtual void load_weights(const std::string& weight_path) = 0;

    // KV Cache Policy (model-specific layout decisions)
    virtual KVCacheConfig kv_cache_config() const = 0;

    // Forward Pass — returns logits, NOT sampled tokens
    virtual ForwardResult prefill(const std::vector<RequestContext>& requests,
                                  cudaStream_t stream) = 0;
    virtual ForwardResult decode(const std::vector<RequestContext>& requests,
                                 cudaStream_t stream) = 0;

    // Scheduling Hints
    virtual int max_batch_size() const = 0;
    virtual int max_tokens_per_batch() const = 0;
    virtual bool supports_in_flight_batching() const { return true; }
};
```

### Ownership boundaries

- **Core/memory** owns the KV cache block pool: allocates blocks, manages the radix tree for prefix caching, handles eviction (LRU), and tracks memory budget for admission control. Block indices are passed to the model via `RequestContext::block_table`.
- **Model module** owns the KV data layout *within* blocks (e.g., `[num_kv_heads, block_size, head_dim]`) and reads/writes KV data at the addresses corresponding to the block indices. The model does not allocate or free blocks.
- **Core/sampling** owns token sampling. The model returns raw logits via `ForwardResult`; the core applies temperature, top-k/p, repetition penalty, and produces the sampled token. This keeps sampling logic reusable and out of generated code.
- **Core/scheduler** owns request lifecycle: tracks EOS/stop conditions using `ModelConfig::eos_token_id` and `stop_token_ids`, enforces per-request `max_new_tokens`, and removes completed requests from the active batch.
- Admission control: `min(available_memory_blocks, model.max_batch_size())`. The `max_tokens_per_batch()` hint budgets the total token count per iteration for chunked prefill + decode.

## Kernel Interface

```cpp
class KernelProvider {
public:
    virtual void gemm(GemmDescriptor desc, void* A, void* B, void* C,
                      cudaStream_t stream) = 0;
    virtual void fused_attention(AttentionDescriptor desc, ...) = 0;
    virtual void rms_norm(void* input, void* weight, void* output,
                          int hidden_size, cudaStream_t stream) = 0;
    virtual void silu_mul(void* input, void* gate, void* output,
                          int size, cudaStream_t stream) = 0;
};
```

The model module receives a `KernelProvider*` at construction and uses it for all compute. This is the plug point for swapping backends later (Triton, cuBLAS, MLIR codegen).

### Initial Kernel Implementations (Prototype)

For the prototype, prefer existing high-quality libraries behind the `KernelProvider` interface to minimize risk. Custom CUTLASS/CuTe kernels can replace these later.

- **GEMM**: cuBLAS for prototype (well-optimized, zero authoring effort). CUTLASS 3.x as a future drop-in replacement.
- **Fused Attention**: FlashAttention-2/3 library with paged KV cache support. This is the hardest kernel to write from scratch — use existing implementations.
- **RMSNorm**: Small hand-written CUDA kernel (straightforward fused reduce + normalize + scale).
- **SwiGLU**: Small hand-written CUDA kernel (fused SiLU(x) * y).
- **RoPE**: Hand-written CUDA kernel or fused into attention.

## Shared Core Components

### Server (`core/server/`)
- Lightweight HTTP server (cpp-httplib, header-only)
- OpenAI-compatible endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/models`
- SSE streaming for token-by-token responses
- Parses requests into `GenerationRequest` structs, hands to scheduler

### Scheduler (`core/scheduler/`)
- Request queue: pending → active → completed
- **Continuous batching**: new requests join active batches between decode steps
- **Chunked prefill**: long prompts split across iterations; scheduler budgets tokens per step using model's `max_tokens_per_batch()`
- First-come-first-serve priority (extensible later)
- Admission control based on memory availability

### Tokenizer (`core/tokenizer/`)
- Common interface wrapping sentencepiece or tiktoken
- Loads from HF model directory
- Encode/decode, special tokens handling
- Chat templates: hardcoded per-model templates for the prototype (e.g., LLaMA chat format). A Jinja2 engine in C++ is out of scope — if needed, a Python preprocessing script can apply HF chat templates before feeding to the server

### Sampling (`core/sampling/`)
- Takes logits from `BatchResult`, applies temperature, top-k, top-p, repetition penalty
- Returns sampled token IDs
- Separated from model module for reuse; models can override if needed

### Memory (`core/memory/`)
- GPU memory pool: pre-allocates, hands out blocks
- **Prefix cache (radix tree)**: maps token sequences to KV cache blocks for reuse
- Reference counting on shared blocks — LRU eviction when no longer referenced
- Usage tracking for scheduler admission control

## Runtime Features

### Chunked Prefill
Long prompts are split into chunks processed across multiple iterations. This allows decode requests to share the batch, preventing head-of-line blocking.

- Scheduler tracks partially-prefilled requests with current position offset
- Each iteration budgets a token count across all prefill chunks + decode tokens
- `RequestContext` carries chunk offset and length; model appends KV cache incrementally

### Prefix Caching (KV Cache Reuse)
When multiple requests share a common prefix (e.g., same system prompt), cached KV blocks are reused instead of recomputed.

- Radix tree in `core/memory/` indexes token sequences → KV cache blocks
- On new request: scheduler checks for prefix match, sets `prefill_start_pos` accordingly
- Reference counting on shared blocks; unreferenced blocks become LRU eviction candidates
- Model module handles "start prefill from position N" — prefix tokens already in cache

## Claude Model Generation Workflow

```bash
python tools/generate_model.py --hf-model meta-llama/Llama-3-8B
```

1. Script downloads/reads HF `config.json` and model class code
2. Invokes Claude with the config + `KernelProvider` API + `ModelModule` interface
3. Claude generates `models/llama/model.{h,cpp}` and `models/llama/runtime.{h,cpp}`

### Weight Loading

Models load weights from HuggingFace safetensors format. A lightweight C++ safetensors reader parses the binary format (header is JSON, tensors are memory-mapped). This avoids a separate conversion step — weights are loaded directly from the HF download directory.
4. Build: `cmake .. -DMODEL=llama && make -j`
5. Run: `./llm-serve-llama --model-dir /path/to/Llama-3-8B --port 8000`

## End-to-End Request Flow

```
1. Client POST /v1/chat/completions {messages, stream: true}
2. Server parses → GenerationRequest (tokens, sampling params)
3. Scheduler enqueues
   ├── Prefix cache lookup → reuse cached KV blocks if match found
   └── Request enters pending queue with prefill_start_pos
4. Main loop iteration:
   ├── Scheduler forms batch:
   │   ├── New request: prefill chunk (e.g., tokens 200-400, budgeted)
   │   └── Active requests: decode (1 token each)
   ├── model.prefill(chunked_requests) → append KV, return ForwardResult (logits)
   ├── model.decode(decode_requests) → ForwardResult (logits)
   ├── sampling.sample(logits) → sampled token IDs
   ├── scheduler checks EOS / stop tokens / max_new_tokens → mark completed requests
   └── server.stream_token() via SSE
5. Repeat until stop token or max_tokens
6. Server sends final SSE chunk with usage stats
```

## Build & Run

```bash
mkdir build && cd build
cmake .. -DMODEL=llama
make -j
./llm-serve-llama --model-dir /path/to/Llama-3-8B --port 8000
```

Single binary, single process, single GPU.

## Prototype Scope

| In scope | Out of scope (future) |
|----------|----------------------|
| LLaMA-style dense decoder-only model | MoE models |
| Single GPU | Tensor/pipeline parallelism |
| FP16/BF16 inference | Quantization (INT8/INT4) |
| Continuous batching | Speculative decoding |
| Chunked prefill | Disaggregated serving |
| Prefix caching (KV reuse) | LoRA / adapter support |
| OpenAI-compatible REST + SSE streaming | gRPC |
| CUTLASS/CuTe kernels | Alternative kernel backends |

## Deferred for Prototype

The following are explicitly out of scope but acknowledged as needed for production:
- **Request cancellation**: client disconnect on SSE does not cancel in-flight work
- **Error recovery**: OOM during prefill is fatal (no graceful degradation)
- **Health check endpoints**: no `/health` or `/ready`
- **Graceful shutdown**: SIGINT kills the process (no drain)
- **Validation of generated code**: future work — reference output comparison, unit tests per generated module

## Dependencies

- CUDA Toolkit (12.x)
- cuBLAS (ships with CUDA)
- FlashAttention-2/3 (for fused attention kernel)
- CUTLASS 3.x (for future custom kernels; header-only)
- cpp-httplib (header-only HTTP server)
- nlohmann/json (header-only JSON)
- sentencepiece or tiktoken (tokenizer)
- CMake 3.24+
