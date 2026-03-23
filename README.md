# LLM-Singularity

A standalone C++ LLM serving system where each model is a self-contained bundle — model forward pass and runtime policy fused together — plugged into a shared core library.

## Why

Traditional LLM serving frameworks use a generic runtime that works across all models. LLM-Singularity takes a different approach: for each model, generate specialized C++ code that bundles the forward pass with its runtime policy (KV cache layout, batching strategy, memory budgeting). An AI assistant (Claude) reads a HuggingFace model config and generates the model module.

This gives you:
- **Performance** — model-specific optimizations that a generic runtime can't do
- **Simplicity** — two files describe the entire model's behavior
- **AI-assisted development** — Claude generates model code, making per-model specialization economically viable

## Features

- OpenAI-compatible REST API (`/v1/chat/completions`, `/v1/completions`)
- SSE streaming for token-by-token responses
- Continuous batching with in-flight request joining
- Chunked prefill (long prompts don't block decode)
- Prefix caching / KV cache reuse (radix tree with LRU eviction)
- Paged KV cache with block-table indirection
- Auto-download models from HuggingFace
- Single binary per model

## Quickstart

### Using Docker (recommended)

```bash
# Build the image
docker build -t llm-singularity .

# Run with a HuggingFace model (auto-downloads)
docker run --gpus all -p 8000:8000 \
  llm-singularity --model-dir meta-llama/Llama-3.2-1B

# For gated models, pass your HF token
docker run --gpus all -p 8000:8000 \
  -e HF_TOKEN=hf_xxx \
  llm-singularity --model-dir meta-llama/Llama-3.2-1B --hf-token $HF_TOKEN

# With a local model directory
docker run --gpus all -p 8000:8000 \
  -v /path/to/model:/model \
  llm-singularity --model-dir /model
```

### Building from source

**Prerequisites:**
- CUDA Toolkit 12.x
- CMake 3.24+
- C++17 compiler (GCC 9+ or Clang 10+)
- Python 3.8+ with `huggingface_hub` (for model download only)

```bash
# Clone
git clone https://github.com/suyoggupta/llm-singularity.git
cd llm-singularity

# Build
mkdir build && cd build
cmake .. -DMODEL=llama
make -j$(nproc)

# Run (auto-downloads from HuggingFace)
./llm-serve-llama --model-dir meta-llama/Llama-3.2-1B --port 8000

# Or with a local model directory
./llm-serve-llama --model-dir /path/to/Llama-3.2-1B --port 8000
```

### Send a request

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama",
    "messages": [{"role": "user", "content": "What is the meaning of life?"}],
    "max_tokens": 100,
    "stream": true
  }'
```

## CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--model-dir` | Local path or HuggingFace model ID | (required) |
| `--hf-token` | HuggingFace auth token for gated models | |
| `--host` | Listen address | `0.0.0.0` |
| `--port` | Listen port | `8000` |

## Architecture

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
│   top-k/p, temp      │   sentencepiece      │
├──────────────────────┴──────────────────────┤
│              Memory Pool                     │  core/memory/
│   GPU block allocator, prefix cache (radix)  │
├─────────────────────────────────────────────┤
│          Model Module Interface              │  models/model_interface.h
├─────────────────────────────────────────────┤
│  ┌─────────────────────────────────────┐    │
│  │  Generated Model (e.g. LLaMA)      │    │  models/llama/
│  │  ├── llama_model.cpp  (forward)    │    │
│  │  └── (runtime policy built-in)     │    │
│  └─────────────────────────────────────┘    │
├─────────────────────────────────────────────┤
│           Kernel Provider                    │  kernels/
│  cuBLAS GEMM, paged attention, RMSNorm,     │
│  SwiGLU, RoPE, embedding, KV cache scatter  │
└─────────────────────────────────────────────┘
```

**Shared core** handles infrastructure (HTTP, scheduling, tokenization, sampling, memory).
**Model modules** implement the forward pass and KV cache policy.
**Kernel provider** abstracts compute — cuBLAS today, CUTLASS/Triton/MLIR tomorrow.

## Project Structure

```
llm-singularity/
├── app/main.cpp                    # CLI entry point, wires everything together
├── core/                           # Shared library
│   ├── include/core/
│   │   ├── memory.h                # BlockPool + PrefixCache
│   │   ├── safetensors.h           # HF safetensors reader (mmap)
│   │   ├── sampling.h              # top-k, top-p, temperature, rep penalty
│   │   ├── scheduler.h             # continuous batching scheduler
│   │   ├── server.h                # OpenAI-compatible HTTP server
│   │   ├── tokenizer.h             # sentencepiece wrapper
│   │   └── types.h                 # DataType, shared types
│   └── src/                        # Implementations
├── kernels/                        # CUDA kernels + kernel provider
│   ├── include/kernels/
│   │   ├── kernel_provider.h       # Abstract kernel API
│   │   ├── cublas_provider.h       # cuBLAS + CUDA kernel provider
│   │   ├── attention.h             # Paged attention
│   │   ├── rmsnorm.h, rope.h, ... # Kernel launch headers
│   └── src/                        # .cu implementations
├── models/
│   ├── include/models/
│   │   └── model_interface.h       # ModelModule ABC
│   └── llama/                      # LLaMA model module
├── tests/                          # Google Test suite
├── tools/
│   └── download_model.py           # HuggingFace model downloader
└── third_party/                    # CMake FetchContent deps
```

## Running Tests

```bash
cd build
ctest --output-on-failure

# With a real model (enables model-dependent tests)
TEST_MODEL_PATH=/path/to/Llama-3.2-1B ctest --output-on-failure
```

## Current Limitations (Prototype)

- Single GPU only (no tensor/pipeline parallelism)
- Float32 compute only (no FP16/INT8 quantization)
- LLaMA architecture only (more models can be generated)
- No request cancellation on client disconnect
- No graceful shutdown (SIGINT kills immediately)
- Sampling params from request not fully wired through

## Roadmap

- [ ] Tensor parallelism (multi-GPU)
- [ ] FP16/BF16 inference
- [ ] CUTLASS/CuTe custom kernels (replace cuBLAS)
- [ ] MoE model support (Mixtral, Qwen3.5)
- [ ] `tools/generate_model.py` — Claude-powered model code generation from HF configs
- [ ] Speculative decoding
- [ ] INT8/INT4 quantization

## License

Apache 2.0
