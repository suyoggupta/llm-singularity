// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include "kernels/rmsnorm.h"
#include "kernels/activations.h"
#include "kernels/rope.h"
#include "kernels/cublas_provider.h"

// ---------------------------------------------------------------------------
// Helper: CUDA memory RAII wrapper
// ---------------------------------------------------------------------------
struct CudaBuffer {
    void* ptr = nullptr;
    CudaBuffer(size_t bytes) { cudaMalloc(&ptr, bytes); }
    ~CudaBuffer() { if (ptr) cudaFree(ptr); }
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
};

// ---------------------------------------------------------------------------
// RMSNorm Tests
// ---------------------------------------------------------------------------
TEST(KernelTest, RMSNorm) {
    const int rows = 2, cols = 4;
    std::vector<float> h_input = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> h_weight = {1, 1, 1, 1};
    std::vector<float> h_output(rows * cols);

    CudaBuffer d_input(rows * cols * sizeof(float));
    CudaBuffer d_weight(cols * sizeof(float));
    CudaBuffer d_output(rows * cols * sizeof(float));

    cudaMemcpy(d_input.ptr, h_input.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight.ptr, h_weight.data(), cols * sizeof(float), cudaMemcpyHostToDevice);

    launch_rms_norm(static_cast<float*>(d_input.ptr),
                    static_cast<float*>(d_weight.ptr),
                    static_cast<float*>(d_output.ptr),
                    rows, cols, 1e-6f, nullptr);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output.data(), d_output.ptr, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Row 0: rms = sqrt((1+4+9+16)/4) = sqrt(7.5)
    float rms0 = std::sqrt((1.0f + 4.0f + 9.0f + 16.0f) / 4.0f);
    EXPECT_NEAR(h_output[0], 1.0f / rms0, 1e-4f);
    EXPECT_NEAR(h_output[1], 2.0f / rms0, 1e-4f);
    EXPECT_NEAR(h_output[2], 3.0f / rms0, 1e-4f);
    EXPECT_NEAR(h_output[3], 4.0f / rms0, 1e-4f);

    // Row 1: rms = sqrt((25+36+49+64)/4) = sqrt(43.5)
    float rms1 = std::sqrt((25.0f + 36.0f + 49.0f + 64.0f) / 4.0f);
    EXPECT_NEAR(h_output[4], 5.0f / rms1, 1e-4f);
    EXPECT_NEAR(h_output[5], 6.0f / rms1, 1e-4f);
    EXPECT_NEAR(h_output[6], 7.0f / rms1, 1e-4f);
    EXPECT_NEAR(h_output[7], 8.0f / rms1, 1e-4f);
}

TEST(KernelTest, RMSNormWithWeight) {
    const int rows = 1, cols = 4;
    std::vector<float> h_input = {2, 4, 6, 8};
    std::vector<float> h_weight = {0.5f, 1.0f, 1.5f, 2.0f};
    std::vector<float> h_output(cols);

    CudaBuffer d_input(cols * sizeof(float));
    CudaBuffer d_weight(cols * sizeof(float));
    CudaBuffer d_output(cols * sizeof(float));

    cudaMemcpy(d_input.ptr, h_input.data(), cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight.ptr, h_weight.data(), cols * sizeof(float), cudaMemcpyHostToDevice);

    launch_rms_norm(static_cast<float*>(d_input.ptr),
                    static_cast<float*>(d_weight.ptr),
                    static_cast<float*>(d_output.ptr),
                    rows, cols, 1e-6f, nullptr);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output.data(), d_output.ptr, cols * sizeof(float), cudaMemcpyDeviceToHost);

    float rms = std::sqrt((4.0f + 16.0f + 36.0f + 64.0f) / 4.0f);
    EXPECT_NEAR(h_output[0], 2.0f / rms * 0.5f, 1e-4f);
    EXPECT_NEAR(h_output[1], 4.0f / rms * 1.0f, 1e-4f);
    EXPECT_NEAR(h_output[2], 6.0f / rms * 1.5f, 1e-4f);
    EXPECT_NEAR(h_output[3], 8.0f / rms * 2.0f, 1e-4f);
}

// ---------------------------------------------------------------------------
// SwiGLU (SiLU * Mul) Tests
// ---------------------------------------------------------------------------
TEST(KernelTest, SiluMul) {
    const int size = 4;
    std::vector<float> h_gate = {0.0f, 1.0f, -1.0f, 2.0f};
    std::vector<float> h_up = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> h_output(size);

    CudaBuffer d_gate(size * sizeof(float));
    CudaBuffer d_up(size * sizeof(float));
    CudaBuffer d_output(size * sizeof(float));

    cudaMemcpy(d_gate.ptr, h_gate.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_up.ptr, h_up.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    launch_silu_mul(static_cast<float*>(d_gate.ptr),
                    static_cast<float*>(d_up.ptr),
                    static_cast<float*>(d_output.ptr),
                    size, nullptr);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output.data(), d_output.ptr, size * sizeof(float), cudaMemcpyDeviceToHost);

    // silu(x) = x / (1 + exp(-x))
    auto silu = [](float x) { return x / (1.0f + std::exp(-x)); };

    EXPECT_NEAR(h_output[0], silu(0.0f) * 1.0f, 1e-5f);   // silu(0) = 0
    EXPECT_NEAR(h_output[1], silu(1.0f) * 1.0f, 1e-5f);   // silu(1) ~ 0.7311
    EXPECT_NEAR(h_output[2], silu(-1.0f) * 1.0f, 1e-5f);  // silu(-1) ~ -0.2689
    EXPECT_NEAR(h_output[3], silu(2.0f) * 1.0f, 1e-5f);   // silu(2) ~ 1.7616
}

TEST(KernelTest, SiluMulWithScaling) {
    const int size = 3;
    std::vector<float> h_gate = {1.0f, 2.0f, 3.0f};
    std::vector<float> h_up = {2.0f, 0.5f, -1.0f};
    std::vector<float> h_output(size);

    CudaBuffer d_gate(size * sizeof(float));
    CudaBuffer d_up(size * sizeof(float));
    CudaBuffer d_output(size * sizeof(float));

    cudaMemcpy(d_gate.ptr, h_gate.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_up.ptr, h_up.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    launch_silu_mul(static_cast<float*>(d_gate.ptr),
                    static_cast<float*>(d_up.ptr),
                    static_cast<float*>(d_output.ptr),
                    size, nullptr);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output.data(), d_output.ptr, size * sizeof(float), cudaMemcpyDeviceToHost);

    auto silu = [](float x) { return x / (1.0f + std::exp(-x)); };

    EXPECT_NEAR(h_output[0], silu(1.0f) * 2.0f, 1e-5f);
    EXPECT_NEAR(h_output[1], silu(2.0f) * 0.5f, 1e-5f);
    EXPECT_NEAR(h_output[2], silu(3.0f) * -1.0f, 1e-5f);
}

// ---------------------------------------------------------------------------
// RoPE Tests
// ---------------------------------------------------------------------------
TEST(KernelTest, RoPEBasic) {
    // Single token, single head, head_dim=4 (2 dimension pairs)
    const int total_tokens = 1;
    const int num_heads = 1;
    const int num_kv_heads = 1;
    const int head_dim = 4;
    const float theta = 10000.0f;

    // Q = [1, 0, 1, 0], K = [0, 1, 0, 1], position = 1
    std::vector<float> h_q = {1.0f, 0.0f, 1.0f, 0.0f};
    std::vector<float> h_k = {0.0f, 1.0f, 0.0f, 1.0f};
    std::vector<int> h_pos = {1};

    CudaBuffer d_q(num_heads * head_dim * sizeof(float));
    CudaBuffer d_k(num_kv_heads * head_dim * sizeof(float));
    CudaBuffer d_pos(sizeof(int));

    cudaMemcpy(d_q.ptr, h_q.data(), h_q.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k.ptr, h_k.data(), h_k.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos.ptr, h_pos.data(), h_pos.size() * sizeof(int), cudaMemcpyHostToDevice);

    launch_rope(static_cast<float*>(d_q.ptr),
                static_cast<float*>(d_k.ptr),
                total_tokens, num_heads, num_kv_heads, head_dim,
                static_cast<int*>(d_pos.ptr), theta, nullptr);
    cudaDeviceSynchronize();

    std::vector<float> h_q_out(head_dim), h_k_out(head_dim);
    cudaMemcpy(h_q_out.data(), d_q.ptr, head_dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_k_out.data(), d_k.ptr, head_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Dim pair 0: freq = 1/theta^(0/4) = 1.0, angle = 1*1 = 1.0
    float cos0 = std::cos(1.0f);
    float sin0 = std::sin(1.0f);
    // Q pair 0: (1*cos - 0*sin, 1*sin + 0*cos) = (cos, sin)
    EXPECT_NEAR(h_q_out[0], cos0, 1e-5f);
    EXPECT_NEAR(h_q_out[1], sin0, 1e-5f);

    // Dim pair 1: freq = 1/theta^(2/4) = 1/100, angle = 1*0.01 = 0.01
    float freq1 = 1.0f / std::pow(theta, 2.0f / head_dim);
    float angle1 = 1.0f * freq1;
    float cos1 = std::cos(angle1);
    float sin1 = std::sin(angle1);
    // Q pair 1: (1*cos - 0*sin, 1*sin + 0*cos)
    EXPECT_NEAR(h_q_out[2], cos1, 1e-5f);
    EXPECT_NEAR(h_q_out[3], sin1, 1e-5f);

    // K pair 0: (0*cos - 1*sin, 0*sin + 1*cos) = (-sin, cos)
    EXPECT_NEAR(h_k_out[0], -sin0, 1e-5f);
    EXPECT_NEAR(h_k_out[1], cos0, 1e-5f);

    // K pair 1: (0*cos - 1*sin, 0*sin + 1*cos)
    EXPECT_NEAR(h_k_out[2], -sin1, 1e-5f);
    EXPECT_NEAR(h_k_out[3], cos1, 1e-5f);
}

TEST(KernelTest, RoPEPositionZero) {
    // Position 0 should leave values unchanged (angle=0 => cos=1, sin=0)
    const int head_dim = 4;
    std::vector<float> h_q = {3.0f, 7.0f, 1.5f, 2.5f};
    std::vector<float> h_k = {4.0f, 5.0f, 6.0f, 8.0f};
    std::vector<int> h_pos = {0};

    CudaBuffer d_q(head_dim * sizeof(float));
    CudaBuffer d_k(head_dim * sizeof(float));
    CudaBuffer d_pos(sizeof(int));

    cudaMemcpy(d_q.ptr, h_q.data(), head_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k.ptr, h_k.data(), head_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos.ptr, h_pos.data(), sizeof(int), cudaMemcpyHostToDevice);

    launch_rope(static_cast<float*>(d_q.ptr),
                static_cast<float*>(d_k.ptr),
                1, 1, 1, head_dim,
                static_cast<int*>(d_pos.ptr), 10000.0f, nullptr);
    cudaDeviceSynchronize();

    std::vector<float> h_q_out(head_dim), h_k_out(head_dim);
    cudaMemcpy(h_q_out.data(), d_q.ptr, head_dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_k_out.data(), d_k.ptr, head_dim * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < head_dim; ++i) {
        EXPECT_NEAR(h_q_out[i], h_q[i], 1e-5f);
        EXPECT_NEAR(h_k_out[i], h_k[i], 1e-5f);
    }
}

// ---------------------------------------------------------------------------
// CublasProvider GEMM Test
// ---------------------------------------------------------------------------
TEST(KernelTest, CublasGemm) {
    // 2x3 * 3x4 = 2x4 matmul in float32
    const int M = 2, N = 4, K = 3;
    // A is row-major [2,3], B is row-major [3,4]
    std::vector<float> h_A = {1, 2, 3, 4, 5, 6};
    std::vector<float> h_B = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<float> h_C(M * N, 0.0f);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    CublasProvider provider;
    GemmDescriptor desc;
    desc.M = M;
    desc.N = N;
    desc.K = K;
    desc.input_dtype = DataType::kFloat32;
    desc.output_dtype = DataType::kFloat32;

    provider.gemm(desc, d_A, d_B, d_C, nullptr);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // C = A * B (row-major)
    // C[0][0] = 1*1 + 2*5 + 3*9 = 38
    // C[0][1] = 1*2 + 2*6 + 3*10 = 44
    // C[0][2] = 1*3 + 2*7 + 3*11 = 50
    // C[0][3] = 1*4 + 2*8 + 3*12 = 56
    // C[1][0] = 4*1 + 5*5 + 6*9 = 83
    EXPECT_NEAR(h_C[0], 38.0f, 1e-3);
    EXPECT_NEAR(h_C[1], 44.0f, 1e-3);
    EXPECT_NEAR(h_C[2], 50.0f, 1e-3);
    EXPECT_NEAR(h_C[3], 56.0f, 1e-3);
    EXPECT_NEAR(h_C[4], 83.0f, 1e-3);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}
