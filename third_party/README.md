# Third-Party Dependencies

All third-party dependencies are fetched at configure time via CMake `FetchContent`.

| Library | Version | Purpose |
|---------|---------|---------|
| [cpp-httplib](https://github.com/yhirose/cpp-httplib) | v0.18.3 | Header-only HTTP/HTTPS server and client |
| [nlohmann/json](https://github.com/nlohmann/json) | v3.11.3 | Header-only JSON parsing and serialization |
| [Google Test](https://github.com/google/googletest) | v1.15.2 | C++ unit testing framework |
| [sentencepiece](https://github.com/google/sentencepiece) | v0.2.0 | Tokenizer (BPE/Unigram LM) |

No source code is vendored here. The `build/_deps/` directory (gitignored) holds the fetched sources.
