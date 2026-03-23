// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION &
// AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0

#pragma once
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "core/sampling.h"
#include "core/types.h"

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

// Parsing
ChatCompletionRequest parse_chat_completion_request(const std::string& json_body);
CompletionRequest parse_completion_request(const std::string& json_body);

// Formatting
std::string format_sse_chunk(const std::string& model, const std::string& token,
                              const std::string& finish_reason);
std::string format_sse_done(const std::string& model, const std::string& finish_reason);
std::string format_models_response(const std::string& model_id);

// Callback type for generation: accepts raw prompt text; main.cpp wires in
// tokenization internally.
using GenerateCallback = std::function<void(
    const std::string& prompt_text, const SamplingParams& params, int max_tokens,
    bool stream, std::function<void(const std::string& token, bool done)> stream_cb)>;

// Callback type for applying a chat template: accepts role/content pairs,
// returns the formatted prompt string ready for tokenization.
using ChatTemplateCallback = std::function<std::string(
    const std::vector<std::pair<std::string, std::string>>& messages)>;

class Server {
public:
    Server(const std::string& host, int port, const std::string& model_name,
           GenerateCallback callback,
           ChatTemplateCallback chat_template_fn = nullptr);
    ~Server();
    void start();  // blocking
    void stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
