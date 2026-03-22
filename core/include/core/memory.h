#pragma once
#include <cstddef>
#include <optional>
#include <vector>
#include <stack>
#include <cuda_runtime.h>

class BlockPool {
public:
    BlockPool(int num_blocks, size_t block_size_bytes);
    ~BlockPool();

    std::optional<int> allocate();          // returns block index
    void free(int block_idx);
    int num_free_blocks() const;
    int num_total_blocks() const;
    size_t block_size_bytes() const;
    void* block_ptr(int block_idx) const;   // GPU pointer for block

private:
    int num_blocks_;
    size_t block_size_bytes_;
    void* gpu_buffer_ = nullptr;            // single GPU allocation
    std::stack<int> free_blocks_;
};
