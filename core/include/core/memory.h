#pragma once
#include <cstddef>
#include <list>
#include <map>
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

// ---------------------------------------------------------------------------
// PrefixCache — token-sequence → block-indices cache with LRU eviction
// ---------------------------------------------------------------------------

struct PrefixCacheResult {
    int matched_tokens;         // number of tokens matched (length of matched key)
    std::vector<int> blocks;    // block indices for the matched prefix
};

class PrefixCache {
public:
    explicit PrefixCache(int block_size = 16);

    // Store a mapping from token sequence to block indices (ref_count starts at 1).
    void insert(const std::vector<int>& tokens, const std::vector<int>& block_indices);

    // Return the longest stored prefix of `tokens`. matched_tokens==0 if none found.
    PrefixCacheResult lookup(const std::vector<int>& tokens) const;

    // Increment ref count for the entry keyed by `tokens`. Removes from LRU list.
    void add_ref(const std::vector<int>& tokens);

    // Decrement ref count. When it reaches 0 the entry is added to the LRU tail.
    void release(const std::vector<int>& tokens);

    // Return current ref count for entry, or 0 if not found.
    int ref_count(const std::vector<int>& tokens) const;

    // Remove and return the least-recently-used entry with ref_count == 0.
    std::optional<PrefixCacheResult> evict_lru();

private:
    struct Entry {
        std::vector<int> blocks;
        int ref_count = 1;
        // Iterator into lru_list_ (valid only when ref_count == 0).
        std::list<std::vector<int>>::iterator lru_it;
        bool in_lru = false;
    };

    int block_size_;
    std::map<std::vector<int>, Entry> entries_;
    // LRU list: front = least recently used (next to evict), back = most recent.
    std::list<std::vector<int>> lru_list_;
};
