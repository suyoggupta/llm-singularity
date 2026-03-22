// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION &
// AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0

#include "core/server.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <sstream>
#include <stdexcept>

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string make_completion_id() {
    // Simple deterministic id based on wall-clock microseconds
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::steady_clock::now().time_since_epoch())
                  .count();
    std::ostringstream oss;
    oss << "chatcmpl-" << us;
    return oss.str();
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

ChatCompletionRequest parse_chat_completion_request(const std::string& json_body) {
    auto j = json::parse(json_body);
    ChatCompletionRequest req;

    req.model = j.value("model", "");

    if (j.contains("messages") && j["messages"].is_array()) {
        for (const auto& m : j["messages"]) {
            ChatMessage msg;
            msg.role = m.value("role", "");
            msg.content = m.value("content", "");
            req.messages.push_back(std::move(msg));
        }
    }

    req.max_tokens = j.value("max_tokens", 256);
    req.temperature = j.value("temperature", 1.0f);
    req.top_p = j.value("top_p", 1.0f);
    req.top_k = j.value("top_k", 0);
    req.stream = j.value("stream", false);

    return req;
}

CompletionRequest parse_completion_request(const std::string& json_body) {
    auto j = json::parse(json_body);
    CompletionRequest req;

    req.model = j.value("model", "");
    req.prompt = j.value("prompt", "");
    req.max_tokens = j.value("max_tokens", 256);
    req.temperature = j.value("temperature", 1.0f);
    req.top_p = j.value("top_p", 1.0f);
    req.top_k = j.value("top_k", 0);
    req.stream = j.value("stream", false);

    return req;
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

std::string format_sse_chunk(const std::string& model, const std::string& token,
                              const std::string& finish_reason) {
    json delta;
    if (!token.empty()) {
        delta["content"] = token;
    }

    json choice;
    choice["index"] = 0;
    choice["delta"] = delta;
    if (finish_reason.empty()) {
        choice["finish_reason"] = nullptr;
    } else {
        choice["finish_reason"] = finish_reason;
    }

    json obj;
    obj["id"] = make_completion_id();
    obj["object"] = "chat.completion.chunk";
    obj["model"] = model;
    obj["choices"] = json::array({choice});

    return "data: " + obj.dump() + "\n\n";
}

std::string format_sse_done(const std::string& model, const std::string& finish_reason) {
    json delta = json::object();

    json choice;
    choice["index"] = 0;
    choice["delta"] = delta;
    choice["finish_reason"] = finish_reason;

    json obj;
    obj["id"] = make_completion_id();
    obj["object"] = "chat.completion.chunk";
    obj["model"] = model;
    obj["choices"] = json::array({choice});

    return "data: " + obj.dump() + "\n\ndata: [DONE]\n\n";
}

std::string format_models_response(const std::string& model_id) {
    json model_obj;
    model_obj["id"] = model_id;
    model_obj["object"] = "model";
    model_obj["created"] = 0;
    model_obj["owned_by"] = "user";

    json resp;
    resp["object"] = "list";
    resp["data"] = json::array({model_obj});

    return resp.dump();
}

// ---------------------------------------------------------------------------
// Server (pimpl)
// ---------------------------------------------------------------------------

struct Server::Impl {
    httplib::Server svr;
    std::string host;
    int port;
    std::string model_name;
    GenerateCallback callback;

    Impl(const std::string& h, int p, const std::string& m, GenerateCallback cb)
        : host(h), port(p), model_name(m), callback(std::move(cb)) {
        setup_routes();
    }

    void setup_routes() {
        // Health check
        svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
            res.set_content(R"({"status":"ok"})", "application/json");
        });

        // List models
        svr.Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
            res.set_content(format_models_response(model_name), "application/json");
        });

        // Chat completions
        svr.Post("/v1/chat/completions",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_chat_completions(req, res);
                 });

        // Completions
        svr.Post("/v1/completions",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_completions(req, res);
                 });
    }

    void handle_chat_completions(const httplib::Request& req, httplib::Response& res) {
        ChatCompletionRequest chat_req;
        try {
            chat_req = parse_chat_completion_request(req.body);
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(std::string(R"({"error":"bad request: ")") + e.what() + "\"}",
                            "application/json");
            return;
        }

        // Build a simple prompt from messages (tokenization done in callback)
        std::string prompt_text;
        for (const auto& msg : chat_req.messages) {
            prompt_text += msg.role + ": " + msg.content + "\n";
        }

        SamplingParams sp;
        sp.temperature = chat_req.temperature;
        sp.top_p = chat_req.top_p;
        sp.top_k = chat_req.top_k;

        if (chat_req.stream) {
            res.set_chunked_content_provider(
                "text/event-stream",
                [this, prompt_text, sp, max_tokens = chat_req.max_tokens,
                 mdl = chat_req.model](size_t /*offset*/, httplib::DataSink& sink) -> bool {
                    bool ok = true;
                    callback(prompt_text, sp, max_tokens, true,
                             [&](const std::string& token, bool done) {
                                 if (!ok) return;
                                 std::string chunk;
                                 if (done) {
                                     chunk = format_sse_done(mdl, "stop");
                                 } else {
                                     chunk = format_sse_chunk(mdl, token, "");
                                 }
                                 if (!sink.write(chunk.c_str(), chunk.size())) {
                                     ok = false;
                                 }
                             });
                    return false;  // signal end of chunked body
                });
        } else {
            std::string generated;
            callback(prompt_text, sp, chat_req.max_tokens, false,
                     [&](const std::string& token, bool done) {
                         if (!done) generated += token;
                     });

            json choice;
            choice["index"] = 0;
            choice["message"] = {{"role", "assistant"}, {"content", generated}};
            choice["finish_reason"] = "stop";

            json resp;
            resp["id"] = make_completion_id();
            resp["object"] = "chat.completion";
            resp["model"] = chat_req.model;
            resp["choices"] = json::array({choice});

            res.set_content(resp.dump(), "application/json");
        }
    }

    void handle_completions(const httplib::Request& req, httplib::Response& res) {
        CompletionRequest comp_req;
        try {
            comp_req = parse_completion_request(req.body);
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(std::string(R"({"error":"bad request: ")") + e.what() + "\"}",
                            "application/json");
            return;
        }

        SamplingParams sp;
        sp.temperature = comp_req.temperature;
        sp.top_p = comp_req.top_p;
        sp.top_k = comp_req.top_k;

        if (comp_req.stream) {
            res.set_chunked_content_provider(
                "text/event-stream",
                [this, prompt = comp_req.prompt, sp, max_tokens = comp_req.max_tokens,
                 mdl = comp_req.model](size_t /*offset*/, httplib::DataSink& sink) -> bool {
                    bool ok = true;
                    callback(prompt, sp, max_tokens, true,
                             [&](const std::string& token, bool done) {
                                 if (!ok) return;
                                 std::string chunk;
                                 if (done) {
                                     chunk = format_sse_done(mdl, "stop");
                                 } else {
                                     chunk = format_sse_chunk(mdl, token, "");
                                 }
                                 if (!sink.write(chunk.c_str(), chunk.size())) {
                                     ok = false;
                                 }
                             });
                    return false;
                });
        } else {
            std::string generated;
            callback(comp_req.prompt, sp, comp_req.max_tokens, false,
                     [&](const std::string& token, bool done) {
                         if (!done) generated += token;
                     });

            json choice;
            choice["index"] = 0;
            choice["text"] = generated;
            choice["finish_reason"] = "stop";

            json resp;
            resp["id"] = make_completion_id();
            resp["object"] = "text_completion";
            resp["model"] = comp_req.model;
            resp["choices"] = json::array({choice});

            res.set_content(resp.dump(), "application/json");
        }
    }
};

Server::Server(const std::string& host, int port, const std::string& model_name,
               GenerateCallback callback)
    : impl_(std::make_unique<Impl>(host, port, model_name, std::move(callback))) {}

Server::~Server() = default;

void Server::start() { impl_->svr.listen(impl_->host, impl_->port); }

void Server::stop() { impl_->svr.stop(); }
