#pragma once

#include "hadamard.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace hadamard {

constexpr float FP8_E4M3_MAX = 240.0f;

__host__ __device__ inline int clamp_int(int value, int lo, int hi) {
    return value < lo ? lo : (value > hi ? hi : value);
}

__host__ __device__ inline int8_t quantize_int4_scalar(float value, float scale) {
    if (scale <= 0.0f) {
        return 0;
    }

    int quantized = static_cast<int>(::roundf(value / scale));
    quantized = clamp_int(quantized, -8, 7);
    return static_cast<int8_t>(quantized);
}

__host__ __device__ inline uint8_t pack_int4_values(int8_t low, int8_t high) {
    return static_cast<uint8_t>((static_cast<uint8_t>(low) & 0x0F) |
                                ((static_cast<uint8_t>(high) & 0x0F) << 4));
}

__host__ __device__ inline int8_t unpack_int4_low(uint8_t packed) {
    const uint8_t nibble = packed & 0x0F;
    return static_cast<int8_t>(nibble >= 8 ? static_cast<int>(nibble) - 16 : nibble);
}

__host__ __device__ inline int8_t unpack_int4_high(uint8_t packed) {
    const uint8_t nibble = (packed >> 4) & 0x0F;
    return static_cast<int8_t>(nibble >= 8 ? static_cast<int>(nibble) - 16 : nibble);
}

__host__ __device__ inline uint8_t float_to_fp8_e4m3(float value) {
    if (value == 0.0f) {
        return 0;
    }

    const uint8_t sign = value < 0.0f ? 0x80 : 0x00;
    const float magnitude = fabsf(value);

    int exponent = 0;
    const float normalized = frexpf(magnitude, &exponent);
    exponent -= 1;

    int encoded_exponent = exponent + 7;
    if (encoded_exponent <= 0) {
        const int mantissa = clamp_int(static_cast<int>(::roundf(magnitude * 512.0f)), 0, 7);
        return static_cast<uint8_t>(sign | mantissa);
    }

    if (encoded_exponent >= 0x0F) {
        return static_cast<uint8_t>(sign | (0x0E << 3) | 0x07);
    }

    float mantissa_float = (normalized * 2.0f - 1.0f) * 8.0f;
    int mantissa = static_cast<int>(::roundf(mantissa_float));
    if (mantissa == 8) {
        mantissa = 0;
        ++encoded_exponent;
        if (encoded_exponent >= 0x0F) {
            return static_cast<uint8_t>(sign | (0x0E << 3) | 0x07);
        }
    }

    return static_cast<uint8_t>(sign | (encoded_exponent << 3) | (mantissa & 0x07));
}

inline void quantize_int4_cpu(const float* input, uint8_t* output, float* scales, int total_rows, int head_dim) {
    const int packed_cols = (head_dim + 1) / 2;
    for (int row = 0; row < total_rows; ++row) {
        const float* row_ptr = input + static_cast<size_t>(row) * head_dim;
        uint8_t* out_ptr = output + static_cast<size_t>(row) * packed_cols;

        float absmax = 0.0f;
        for (int col = 0; col < head_dim; ++col) {
            absmax = std::max(absmax, std::fabs(row_ptr[col]));
        }

        const float scale = absmax > 0.0f ? absmax / 7.0f : 1.0f;
        scales[row] = scale;

        for (int col = 0; col < head_dim; col += 2) {
            const int8_t low = quantize_int4_scalar(row_ptr[col], scale);
            const int8_t high = quantize_int4_scalar(row_ptr[col + 1], scale);
            out_ptr[col / 2] = pack_int4_values(low, high);
        }
    }
}

inline void quantize_fp8_cpu(const float* input, uint8_t* output, float* scales, int total_rows, int head_dim) {
    for (int row = 0; row < total_rows; ++row) {
        const float* row_ptr = input + static_cast<size_t>(row) * head_dim;
        uint8_t* out_ptr = output + static_cast<size_t>(row) * head_dim;

        float absmax = 0.0f;
        for (int col = 0; col < head_dim; ++col) {
            absmax = std::max(absmax, std::fabs(row_ptr[col]));
        }

        const float scale = absmax > 0.0f ? absmax / FP8_E4M3_MAX : 1.0f;
        scales[row] = scale;

        for (int col = 0; col < head_dim; ++col) {
            out_ptr[col] = float_to_fp8_e4m3(row_ptr[col] / scale);
        }
    }
}

template <typename T>
void quantize_int4_gpu(
    const T* d_input,
    uint8_t* d_output,
    float* d_scales,
    int total_rows,
    int head_dim,
    cudaStream_t stream = 0);

template <typename T>
void hadamard_quantize_int4_gpu(
    const T* d_input,
    uint8_t* d_output,
    float* d_scales,
    int total_rows,
    int head_dim,
    cudaStream_t stream = 0);

template <typename T>
void quantize_fp8_gpu(
    const T* d_input,
    uint8_t* d_output,
    float* d_scales,
    int total_rows,
    int head_dim,
    cudaStream_t stream = 0);

template <typename T>
void hadamard_quantize_fp8_gpu(
    const T* d_input,
    uint8_t* d_output,
    float* d_scales,
    int total_rows,
    int head_dim,
    cudaStream_t stream = 0);

}  // namespace hadamard
