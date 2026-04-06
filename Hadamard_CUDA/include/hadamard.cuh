#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace hadamard {

struct cli_options {
    int iterations = 100;
    int warmup_iterations = 10;
    int measurement_trials = 5;
    int head_dim = 0;
    int total_rows = 0;
    bool verbose = false;
    std::string csv_path;
    std::string dtype_filter = "all";
    std::string benchmark_mode = "all";
};

inline void cuda_check(cudaError_t status, const char* expr, const char* file, int line) {
    if (status == cudaSuccess) {
        return;
    }

    std::ostringstream oss;
    oss << "CUDA error: " << cudaGetErrorString(status) << " for " << expr << " at " << file << ":" << line;
    throw std::runtime_error(oss.str());
}

#define CUDA_CHECK(expr) ::hadamard::cuda_check((expr), #expr, __FILE__, __LINE__)

constexpr int TILE_DIM = 16;

inline bool is_power_of_two(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

inline int checked_total_rows(int batch_size, int seq_len, int num_heads) {
    const long long total_rows = static_cast<long long>(batch_size) * seq_len * num_heads;
    if (total_rows <= 0 || total_rows > static_cast<long long>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("Invalid tensor shape for Hadamard transform");
    }
    return static_cast<int>(total_rows);
}

template <typename T>
__host__ __device__ inline float scalar_to_float(T value) {
    return static_cast<float>(value);
}

template <>
__host__ __device__ inline float scalar_to_float<half>(half value) {
    return __half2float(value);
}

template <>
__host__ __device__ inline float scalar_to_float<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__host__ __device__ inline T float_to_scalar(float value) {
    return static_cast<T>(value);
}

template <>
__host__ __device__ inline half float_to_scalar<half>(float value) {
    return __float2half_rn(value);
}

template <>
__host__ __device__ inline __nv_bfloat16 float_to_scalar<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

template <typename T>
constexpr const char* dtype_name() {
    return "unknown";
}

template <>
constexpr const char* dtype_name<half>() {
    return "FP16";
}

template <>
constexpr const char* dtype_name<__nv_bfloat16>() {
    return "BF16";
}

__device__ inline int butterfly_index(int pair_index, int len) {
    const int group = pair_index / len;
    const int offset = pair_index - group * len;
    return group * (len << 1) + offset;
}

inline void hadamard_transform_cpu(float* data, int n) {
    if (!is_power_of_two(n)) {
        throw std::invalid_argument("Hadamard transform expects n to be a power of two");
    }

    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += (len << 1)) {
            for (int j = 0; j < len; ++j) {
                const float u = data[i + j];
                const float v = data[i + j + len];
                data[i + j] = u + v;
                data[i + j + len] = u - v;
            }
        }
    }

    const float inv_sqrt_n = 1.0f / std::sqrt(static_cast<float>(n));
    for (int i = 0; i < n; ++i) {
        data[i] *= inv_sqrt_n;
    }
}

inline void hadamard_transform_cpu_batch(float* data, int total_rows, int head_dim) {
    if (!is_power_of_two(head_dim)) {
        throw std::invalid_argument("Head dimension must be a power of two");
    }

    for (int row = 0; row < total_rows; ++row) {
        hadamard_transform_cpu(data + static_cast<size_t>(row) * head_dim, head_dim);
    }
}

cli_options parse_cli_options(int argc, char** argv);
void print_usage(const char* exe_name);

template <typename T>
void hadamard_transform_gpu(
    T* d_data,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    cudaStream_t stream = 0);

template <typename T>
void hadamard_transform_gpu_wmma(
    T* d_data,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    cudaStream_t stream = 0);

template <typename T>
void hadamard_transform_gpu_best(
    T* d_data,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    cudaStream_t stream = 0);

}  // namespace hadamard
