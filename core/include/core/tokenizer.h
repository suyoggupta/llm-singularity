#pragma once
#include <memory>
#include <string>
#include <utility>
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
    struct Impl;
    std::shared_ptr<Impl> impl_;
};
