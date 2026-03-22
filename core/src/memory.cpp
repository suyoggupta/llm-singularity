#include "core/memory.h"
#include <stdexcept>

BlockPool::BlockPool(int num_blocks, size_t block_size_bytes)
    : num_blocks_(num_blocks), block_size_bytes_(block_size_bytes) {
    if (num_blocks > 0 && block_size_bytes > 0) {
        cudaMalloc(&gpu_buffer_, static_cast<size_t>(num_blocks) * block_size_bytes);
    }
    for (int i = num_blocks - 1; i >= 0; --i) {
        free_blocks_.push(i);
    }
}

BlockPool::~BlockPool() {
    if (gpu_buffer_) {
        cudaFree(gpu_buffer_);
    }
}

std::optional<int> BlockPool::allocate() {
    if (free_blocks_.empty()) return std::nullopt;
    int idx = free_blocks_.top();
    free_blocks_.pop();
    return idx;
}

void BlockPool::free(int block_idx) {
    free_blocks_.push(block_idx);
}

int BlockPool::num_free_blocks() const { return static_cast<int>(free_blocks_.size()); }
int BlockPool::num_total_blocks() const { return num_blocks_; }
size_t BlockPool::block_size_bytes() const { return block_size_bytes_; }

void* BlockPool::block_ptr(int block_idx) const {
    return static_cast<char*>(gpu_buffer_) + static_cast<size_t>(block_idx) * block_size_bytes_;
}
