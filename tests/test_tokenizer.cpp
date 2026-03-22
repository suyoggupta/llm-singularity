#include <gtest/gtest.h>
#include "core/tokenizer.h"

TEST(TokenizerTest, LoadAndEncodeDecode) {
    const char* model_path = std::getenv("TEST_TOKENIZER_PATH");
    if (!model_path) GTEST_SKIP() << "TEST_TOKENIZER_PATH not set";

    Tokenizer tok;
    tok.load(model_path);

    auto ids = tok.encode("Hello, world!");
    EXPECT_GT(ids.size(), 0u);

    auto text = tok.decode(ids);
    EXPECT_EQ(text, "Hello, world!");
}

TEST(TokenizerTest, SpecialTokens) {
    const char* model_path = std::getenv("TEST_TOKENIZER_PATH");
    if (!model_path) GTEST_SKIP() << "TEST_TOKENIZER_PATH not set";

    Tokenizer tok;
    tok.load(model_path);

    EXPECT_GE(tok.eos_token_id(), 0);
    EXPECT_GE(tok.bos_token_id(), 0);
}
