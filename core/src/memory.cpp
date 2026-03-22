#include "core/memory.h"
#include <algorithm>
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

// ---------------------------------------------------------------------------
// PrefixCache
// ---------------------------------------------------------------------------

PrefixCache::PrefixCache(int block_size) : block_size_(block_size) {}

void PrefixCache::insert(const std::vector<int>& tokens, const std::vector<int>& block_indices) {
    auto it = entries_.find(tokens);
    if (it != entries_.end()) {
        // Entry already exists — just update blocks and leave ref_count/LRU alone.
        it->second.blocks = block_indices;
        return;
    }
    Entry entry;
    entry.blocks = block_indices;
    // Start with ref_count=1 but immediately add to LRU, representing a "cached
    // but not actively held" entry.  Callers that need to pin the entry call
    // add_ref() which removes it from the LRU list.
    entry.ref_count = 1;
    entry.in_lru = false;
    entries_[tokens] = std::move(entry);
    // Add to the LRU list so it is evictable by default.
    lru_list_.push_back(tokens);
    entries_[tokens].lru_it = std::prev(lru_list_.end());
    entries_[tokens].in_lru = true;
}

PrefixCacheResult PrefixCache::lookup(const std::vector<int>& tokens) const {
    // Scan all entries and find the one whose key is the longest prefix of `tokens`.
    const std::vector<int>* best_key = nullptr;
    for (const auto& kv : entries_) {
        const auto& key = kv.first;
        if (key.size() > tokens.size()) continue;
        if (!std::equal(key.begin(), key.end(), tokens.begin())) continue;
        if (best_key == nullptr || key.size() > best_key->size()) {
            best_key = &kv.first;
        }
    }
    if (best_key == nullptr) {
        return {0, {}};
    }
    const Entry& entry = entries_.at(*best_key);
    return {static_cast<int>(best_key->size()), entry.blocks};
}

void PrefixCache::add_ref(const std::vector<int>& tokens) {
    auto it = entries_.find(tokens);
    if (it == entries_.end()) return;
    Entry& entry = it->second;
    // Remove from LRU list if present (referenced entries are not evictable).
    if (entry.in_lru) {
        lru_list_.erase(entry.lru_it);
        entry.in_lru = false;
    }
    entry.ref_count++;
}

void PrefixCache::release(const std::vector<int>& tokens) {
    auto it = entries_.find(tokens);
    if (it == entries_.end()) return;
    Entry& entry = it->second;
    if (entry.ref_count <= 0) return;
    entry.ref_count--;
    if (entry.ref_count == 0 && !entry.in_lru) {
        // Add to back of LRU list (most recently released = least urgently evicted).
        lru_list_.push_back(it->first);
        entry.lru_it = std::prev(lru_list_.end());
        entry.in_lru = true;
    }
}

int PrefixCache::ref_count(const std::vector<int>& tokens) const {
    auto it = entries_.find(tokens);
    if (it == entries_.end()) return 0;
    return it->second.ref_count;
}

std::optional<PrefixCacheResult> PrefixCache::evict_lru() {
    if (lru_list_.empty()) return std::nullopt;
    // Front of list = least recently used.
    std::vector<int> key = lru_list_.front();
    lru_list_.pop_front();
    auto it = entries_.find(key);
    if (it == entries_.end()) return std::nullopt;
    PrefixCacheResult result{static_cast<int>(key.size()), it->second.blocks};
    entries_.erase(it);
    return result;
}
