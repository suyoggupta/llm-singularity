#include "core/safetensors.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <numeric>
#include <stdexcept>

#include <nlohmann/json.hpp>

// ---------------------------------------------------------------------------
// TensorInfo
// ---------------------------------------------------------------------------

int64_t TensorInfo::num_elements() const {
    if (shape.empty()) return 0;
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>{});
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static DataType parse_dtype(const std::string& s) {
    if (s == "F32")  return DataType::kFloat32;
    if (s == "F16")  return DataType::kFloat16;
    if (s == "BF16") return DataType::kBFloat16;
    throw std::runtime_error("safetensors: unsupported dtype '" + s + "'");
}

// ---------------------------------------------------------------------------
// SafetensorsFile
// ---------------------------------------------------------------------------

SafetensorsFile::~SafetensorsFile() {
    close();
}

void SafetensorsFile::open(const std::string& path) {
    close();  // close any previously opened file

    // 1. Open the file
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        throw std::runtime_error("safetensors: cannot open '" + path + "'");
    }

    // 2. Get file size
    struct stat st{};
    if (::fstat(fd_, &st) < 0) {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("safetensors: fstat failed for '" + path + "'");
    }
    file_size_ = static_cast<size_t>(st.st_size);

    if (file_size_ < 8) {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("safetensors: file too small '" + path + "'");
    }

    // 3. mmap the entire file
    mmap_ptr_ = ::mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mmap_ptr_ == MAP_FAILED) {
        mmap_ptr_ = nullptr;
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("safetensors: mmap failed for '" + path + "'");
    }

    // 4. Read header size (first 8 bytes, little-endian uint64)
    uint64_t header_size = 0;
    std::memcpy(&header_size, mmap_ptr_, 8);

    if (8 + header_size > file_size_) {
        ::munmap(mmap_ptr_, file_size_);
        mmap_ptr_ = nullptr;
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("safetensors: header_size exceeds file size in '" + path + "'");
    }

    // 5. Parse JSON header
    const char* json_start = static_cast<const char*>(mmap_ptr_) + 8;
    nlohmann::json j = nlohmann::json::parse(json_start, json_start + header_size);

    // 6. data section begins right after the header
    data_offset_ = 8 + static_cast<size_t>(header_size);

    // 7. Parse each tensor entry
    for (auto it = j.begin(); it != j.end(); ++it) {
        // The safetensors spec allows a "__metadata__" key — skip it
        if (it.key() == "__metadata__") continue;

        const auto& val = it.value();

        TensorInfo info;
        info.name = it.key();
        info.dtype = parse_dtype(val.at("dtype").get<std::string>());
        info.shape = val.at("shape").get<std::vector<int64_t>>();

        auto offsets = val.at("data_offsets").get<std::vector<size_t>>();
        if (offsets.size() != 2) {
            throw std::runtime_error(
                "safetensors: data_offsets must have 2 elements for tensor '" + info.name + "'");
        }
        info.offset    = offsets[0];
        info.byte_size = offsets[1] - offsets[0];

        tensors_[info.name] = std::move(info);
    }
}

void SafetensorsFile::close() {
    if (mmap_ptr_ != nullptr) {
        ::munmap(mmap_ptr_, file_size_);
        mmap_ptr_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    file_size_  = 0;
    data_offset_ = 0;
    tensors_.clear();
}

std::vector<std::string> SafetensorsFile::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for (const auto& kv : tensors_) {
        names.push_back(kv.first);
    }
    return names;
}

TensorInfo SafetensorsFile::tensor_info(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("safetensors: tensor '" + name + "' not found");
    }
    return it->second;
}

const void* SafetensorsFile::tensor_data(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("safetensors: tensor '" + name + "' not found");
    }
    const char* base = static_cast<const char*>(mmap_ptr_);
    return base + data_offset_ + it->second.offset;
}
