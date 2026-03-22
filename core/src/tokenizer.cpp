#include "core/tokenizer.h"

#include <stdexcept>
#include <string>
#include <vector>

#include <sentencepiece_processor.h>

struct Tokenizer::Impl {
    sentencepiece::SentencePieceProcessor processor;
};

void Tokenizer::load(const std::string& model_path) {
    impl_ = std::make_shared<Impl>();
    const auto status = impl_->processor.Load(model_path);
    if (!status.ok()) {
        throw std::runtime_error("Failed to load tokenizer model from '" + model_path +
                                 "': " + status.ToString());
    }
}

std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
    if (!impl_) {
        throw std::runtime_error("Tokenizer not loaded — call load() first");
    }
    std::vector<int32_t> ids;
    const auto status = impl_->processor.Encode(text, &ids);
    if (!status.ok()) {
        throw std::runtime_error("Tokenizer encode failed: " + status.ToString());
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    if (!impl_) {
        throw std::runtime_error("Tokenizer not loaded — call load() first");
    }
    std::string text;
    const auto status = impl_->processor.Decode(ids, &text);
    if (!status.ok()) {
        throw std::runtime_error("Tokenizer decode failed: " + status.ToString());
    }
    return text;
}

int32_t Tokenizer::eos_token_id() const {
    if (!impl_) {
        throw std::runtime_error("Tokenizer not loaded — call load() first");
    }
    return static_cast<int32_t>(impl_->processor.eos_id());
}

int32_t Tokenizer::bos_token_id() const {
    if (!impl_) {
        throw std::runtime_error("Tokenizer not loaded — call load() first");
    }
    return static_cast<int32_t>(impl_->processor.bos_id());
}

int Tokenizer::vocab_size() const {
    if (!impl_) {
        throw std::runtime_error("Tokenizer not loaded — call load() first");
    }
    return impl_->processor.GetPieceSize();
}

std::string Tokenizer::apply_chat_template(
    const std::vector<std::pair<std::string, std::string>>& messages) const {
    std::string result;
    for (const auto& [role, content] : messages) {
        result += "<|start_header_id|>" + role + "<|end_header_id|>\n\n" + content +
                  "<|eot_id|>\n";
    }
    // Signal the model to begin its reply
    result += "<|start_header_id|>assistant<|end_header_id|>\n\n";
    return result;
}
