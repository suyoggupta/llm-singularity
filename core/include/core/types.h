#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum class DataType {
    kFloat16,
    kBFloat16,
    kFloat32,
};

inline size_t dtype_size(DataType dt) {
    switch (dt) {
        case DataType::kFloat16:  return 2;
        case DataType::kBFloat16: return 2;
        case DataType::kFloat32:  return 4;
    }
    return 0;
}

inline std::string dtype_name(DataType dt) {
    switch (dt) {
        case DataType::kFloat16:  return "float16";
        case DataType::kBFloat16: return "bfloat16";
        case DataType::kFloat32:  return "float32";
    }
    return "unknown";
}
