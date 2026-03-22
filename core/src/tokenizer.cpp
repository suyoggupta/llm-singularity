// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION &
// AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0

#include "core/tokenizer.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>
#include <sys/wait.h>

#include <nlohmann/json.hpp>

// ---------------------------------------------------------------------------
// Tokenizer implementation using a Python subprocess (tokenizer_worker.py).
// Handles both sentencepiece and tiktoken-based HF tokenizers.
// The Python process stays alive for the server's lifetime.
// Communication: JSON lines over stdin/stdout pipes.
// ---------------------------------------------------------------------------

struct Tokenizer::Impl {
    FILE* to_py = nullptr;   // write to Python's stdin
    FILE* from_py = nullptr; // read from Python's stdout
    int pid = -1;

    int32_t eos_id = -1;
    int32_t bos_id = -1;
    int vocab_sz = 0;

    ~Impl() {
        if (to_py) fclose(to_py);
        if (from_py) fclose(from_py);
    }

    void send(const nlohmann::json& j) {
        std::string line = j.dump() + "\n";
        if (fwrite(line.data(), 1, line.size(), to_py) != line.size()) {
            throw std::runtime_error("Failed to write to tokenizer subprocess");
        }
        fflush(to_py);
    }

    nlohmann::json recv() {
        char buf[65536];
        if (!fgets(buf, sizeof(buf), from_py)) {
            throw std::runtime_error(
                "Tokenizer subprocess closed unexpectedly");
        }
        return nlohmann::json::parse(buf);
    }
};

static std::string find_tokenizer_worker() {
    // Try cwd first
    std::string script = "tools/tokenizer_worker.py";
    if (std::filesystem::exists(script)) return script;

    // Try relative to executable
    if (std::filesystem::exists("/proc/self/exe")) {
        auto exe_path = std::filesystem::read_symlink("/proc/self/exe");
        auto repo_root =
            exe_path.parent_path().parent_path().parent_path();
        script = (repo_root / "tools" / "tokenizer_worker.py").string();
        if (std::filesystem::exists(script)) return script;
    }

    throw std::runtime_error(
        "Cannot find tools/tokenizer_worker.py. "
        "Run from the repo root or install tokenizer_worker.py alongside "
        "the binary.");
}

void Tokenizer::load(const std::string& model_path) {
    impl_ = std::make_shared<Impl>();

    std::string script = find_tokenizer_worker();
    std::string cmd =
        "python3 " + script + " " + model_path;

    // Open bidirectional pipe to the Python subprocess.
    // We need both read and write, so use pipe() + fork/exec.
    int pipe_to_child[2];   // parent writes, child reads
    int pipe_from_child[2]; // child writes, parent reads
    if (pipe(pipe_to_child) != 0 || pipe(pipe_from_child) != 0) {
        throw std::runtime_error("Failed to create pipes for tokenizer");
    }

    pid_t pid = fork();
    if (pid < 0) {
        throw std::runtime_error("fork() failed for tokenizer subprocess");
    }

    if (pid == 0) {
        // Child process
        close(pipe_to_child[1]);   // close write end
        close(pipe_from_child[0]); // close read end
        dup2(pipe_to_child[0], STDIN_FILENO);
        dup2(pipe_from_child[1], STDOUT_FILENO);
        close(pipe_to_child[0]);
        close(pipe_from_child[1]);

        execlp("python3", "python3", script.c_str(), model_path.c_str(),
               nullptr);
        // If exec fails
        _exit(1);
    }

    // Parent process
    close(pipe_to_child[0]);   // close read end
    close(pipe_from_child[1]); // close write end

    impl_->to_py = fdopen(pipe_to_child[1], "w");
    impl_->from_py = fdopen(pipe_from_child[0], "r");
    impl_->pid = pid;

    if (!impl_->to_py || !impl_->from_py) {
        throw std::runtime_error(
            "Failed to open file descriptors for tokenizer subprocess");
    }

    // Wait for "ready" signal
    auto ready = impl_->recv();
    if (ready.contains("error")) {
        throw std::runtime_error(
            "Tokenizer subprocess error: " +
            ready["error"].get<std::string>());
    }

    // Query tokenizer info
    impl_->send({{"cmd", "info"}});
    auto info = impl_->recv();
    if (info.contains("error")) {
        throw std::runtime_error(
            "Tokenizer info error: " + info["error"].get<std::string>());
    }
    impl_->eos_id = info["eos_token_id"].get<int32_t>();
    impl_->bos_id = info["bos_token_id"].get<int32_t>();
    impl_->vocab_sz = info["vocab_size"].get<int>();
}

std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
    if (!impl_) {
        throw std::runtime_error("Tokenizer not loaded");
    }
    impl_->send({{"cmd", "encode"}, {"text", text}});
    auto resp = impl_->recv();
    if (resp.contains("error")) {
        throw std::runtime_error(
            "Tokenizer encode error: " + resp["error"].get<std::string>());
    }
    return resp["ids"].get<std::vector<int32_t>>();
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    if (!impl_) {
        throw std::runtime_error("Tokenizer not loaded");
    }
    impl_->send({{"cmd", "decode"}, {"ids", ids}});
    auto resp = impl_->recv();
    if (resp.contains("error")) {
        throw std::runtime_error(
            "Tokenizer decode error: " + resp["error"].get<std::string>());
    }
    return resp["text"].get<std::string>();
}

int32_t Tokenizer::eos_token_id() const {
    if (!impl_) {
        throw std::runtime_error("Tokenizer not loaded");
    }
    return impl_->eos_id;
}

int32_t Tokenizer::bos_token_id() const {
    if (!impl_) {
        throw std::runtime_error("Tokenizer not loaded");
    }
    return impl_->bos_id;
}

int Tokenizer::vocab_size() const {
    if (!impl_) {
        throw std::runtime_error("Tokenizer not loaded");
    }
    return impl_->vocab_sz;
}

std::string Tokenizer::apply_chat_template(
    const std::vector<std::pair<std::string, std::string>>& messages) const {
    if (!impl_) {
        throw std::runtime_error("Tokenizer not loaded");
    }
    // Convert to the format the Python worker expects
    nlohmann::json msg_array = nlohmann::json::array();
    for (const auto& [role, content] : messages) {
        msg_array.push_back({{"role", role}, {"content", content}});
    }
    impl_->send({{"cmd", "apply_chat_template"}, {"messages", msg_array}});
    auto resp = impl_->recv();
    if (resp.contains("error")) {
        throw std::runtime_error(
            "Chat template error: " + resp["error"].get<std::string>());
    }
    return resp["text"].get<std::string>();
}
