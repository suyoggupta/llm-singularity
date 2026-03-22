// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION &
// AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "core/server.h"

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

TEST(ServerTest, FormatSSEChunk) {
    auto chunk = format_sse_chunk("llama-3-8b", "Hello", "");
    EXPECT_TRUE(chunk.find("data: ") == 0);
    EXPECT_TRUE(chunk.find("\"Hello\"") != std::string::npos);
}

TEST(ServerTest, FormatSSEDone) {
    auto done = format_sse_done("llama-3-8b", "stop");
    EXPECT_TRUE(done.find("\"finish_reason\":\"stop\"") != std::string::npos);
    EXPECT_TRUE(done.find("[DONE]") != std::string::npos);
}

TEST(ServerTest, FormatModelsResponse) {
    auto json = format_models_response("llama-3-8b");
    EXPECT_TRUE(json.find("\"id\":\"llama-3-8b\"") != std::string::npos);
    EXPECT_TRUE(json.find("\"object\":\"list\"") != std::string::npos);
}
