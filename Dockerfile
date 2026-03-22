# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0

# Multi-stage build: compile in a full CUDA devel image, run in a slim runtime image.

# ============================================================================
# Stage 1: Build
# ============================================================================
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    git \
    g++ \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

RUN mkdir build && cd build \
    && cmake .. \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DMODEL=llama \
        -DBUILD_TESTS=OFF \
    && ninja -j$(nproc)

# ============================================================================
# Stage 2: Runtime
# ============================================================================
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    && pip3 install --no-cache-dir huggingface_hub transformers \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the built binary
COPY --from=builder /src/build/app/llm-serve-llama /app/llm-serve-llama

# Copy tools
COPY tools/ /app/tools/

# Cache dir for downloaded models (mount a volume here for persistence)
ENV HF_HOME=/app/.cache
VOLUME /app/.cache

EXPOSE 8000

ENTRYPOINT ["/app/llm-serve-llama"]
CMD ["--help"]
