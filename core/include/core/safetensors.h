#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include "core/types.h"

struct TensorInfo {
    std::string name;
    DataType dtype;
    std::vector<int64_t> shape;
    size_t offset;      // byte offset into data section
    size_t byte_size;

    int64_t num_elements() const;
};

class SafetensorsFile {
public:
    SafetensorsFile() = default;
    ~SafetensorsFile();

    // Non-copyable (owns mmap)
    SafetensorsFile(const SafetensorsFile&) = delete;
    SafetensorsFile& operator=(const SafetensorsFile&) = delete;

    void open(const std::string& path);
    void close();
    std::vector<std::string> tensor_names() const;
    TensorInfo tensor_info(const std::string& name) const;
    const void* tensor_data(const std::string& name) const;  // memory-mapped pointer

private:
    int fd_ = -1;
    void* mmap_ptr_ = nullptr;
    size_t file_size_ = 0;
    size_t data_offset_ = 0;
    std::unordered_map<std::string, TensorInfo> tensors_;
};
