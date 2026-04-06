#include "hadamard.cuh"
#include "quantize.cuh"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace {

template <typename T>
std::vector<T> to_device_vector(const std::vector<float>& input) {
    std::vector<T> converted(input.size());
    for (size_t index = 0; index < input.size(); ++index) {
        converted[index] = hadamard::float_to_scalar<T>(input[index]);
    }
    return converted;
}

template <typename T>
bool run_int4_case(int total_rows, int head_dim) {
    const size_t element_count = static_cast<size_t>(total_rows) * head_dim;
    const size_t packed_count = static_cast<size_t>(total_rows) * (head_dim / 2);

    std::mt19937 rng(static_cast<uint32_t>(head_dim * 4099 + total_rows));
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<float> input(element_count);
    for (float& value : input) {
        value = dist(rng);
    }

    std::vector<T> converted = to_device_vector<T>(input);

    T* d_separate = nullptr;
    T* d_fused = nullptr;
    uint8_t* d_q_separate = nullptr;
    uint8_t* d_q_fused = nullptr;
    float* d_scales_separate = nullptr;
    float* d_scales_fused = nullptr;

    CUDA_CHECK(cudaMalloc(&d_separate, element_count * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_fused, element_count * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_q_separate, packed_count * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_q_fused, packed_count * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_scales_separate, total_rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scales_fused, total_rows * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_separate, converted.data(), element_count * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fused, converted.data(), element_count * sizeof(T), cudaMemcpyHostToDevice));

    hadamard::hadamard_transform_gpu(d_separate, total_rows, 1, 1, head_dim);
    hadamard::quantize_int4_gpu(d_separate, d_q_separate, d_scales_separate, total_rows, head_dim);
    hadamard::hadamard_quantize_int4_gpu(d_fused, d_q_fused, d_scales_fused, total_rows, head_dim);

    std::vector<uint8_t> q_separate(packed_count);
    std::vector<uint8_t> q_fused(packed_count);
    std::vector<float> scales_separate(total_rows);
    std::vector<float> scales_fused(total_rows);
    CUDA_CHECK(cudaMemcpy(q_separate.data(), d_q_separate, packed_count, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(q_fused.data(), d_q_fused, packed_count, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(scales_separate.data(), d_scales_separate, total_rows * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(scales_fused.data(), d_scales_fused, total_rows * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_separate));
    CUDA_CHECK(cudaFree(d_fused));
    CUDA_CHECK(cudaFree(d_q_separate));
    CUDA_CHECK(cudaFree(d_q_fused));
    CUDA_CHECK(cudaFree(d_scales_separate));
    CUDA_CHECK(cudaFree(d_scales_fused));

    const bool bit_exact = q_separate == q_fused;
    float max_scale_err = 0.0f;
    for (int row = 0; row < total_rows; ++row) {
        max_scale_err = std::max(max_scale_err, std::fabs(scales_separate[row] - scales_fused[row]));
    }

    const bool pass = bit_exact && max_scale_err < 1.0e-6f;
    std::cout << '[' << (pass ? "PASS" : "FAIL") << "] INT4 fused dtype=" << hadamard::dtype_name<T>()
              << " head_dim=" << head_dim
              << " total_rows=" << total_rows
              << " max_scale_err=" << std::scientific << std::setprecision(3) << max_scale_err
              << '\n';
    return pass;
}

template <typename T>
bool run_fp8_case(int total_rows, int head_dim) {
    const size_t element_count = static_cast<size_t>(total_rows) * head_dim;

    std::mt19937 rng(static_cast<uint32_t>(head_dim * 8191 + total_rows));
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<float> input(element_count);
    for (float& value : input) {
        value = dist(rng);
    }

    std::vector<T> converted = to_device_vector<T>(input);

    T* d_separate = nullptr;
    T* d_fused = nullptr;
    uint8_t* d_q_separate = nullptr;
    uint8_t* d_q_fused = nullptr;
    float* d_scales_separate = nullptr;
    float* d_scales_fused = nullptr;

    CUDA_CHECK(cudaMalloc(&d_separate, element_count * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_fused, element_count * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_q_separate, element_count * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_q_fused, element_count * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_scales_separate, total_rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scales_fused, total_rows * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_separate, converted.data(), element_count * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fused, converted.data(), element_count * sizeof(T), cudaMemcpyHostToDevice));

    hadamard::hadamard_transform_gpu(d_separate, total_rows, 1, 1, head_dim);
    hadamard::quantize_fp8_gpu(d_separate, d_q_separate, d_scales_separate, total_rows, head_dim);
    hadamard::hadamard_quantize_fp8_gpu(d_fused, d_q_fused, d_scales_fused, total_rows, head_dim);

    std::vector<uint8_t> q_separate(element_count);
    std::vector<uint8_t> q_fused(element_count);
    std::vector<float> scales_separate(total_rows);
    std::vector<float> scales_fused(total_rows);
    CUDA_CHECK(cudaMemcpy(q_separate.data(), d_q_separate, element_count, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(q_fused.data(), d_q_fused, element_count, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(scales_separate.data(), d_scales_separate, total_rows * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(scales_fused.data(), d_scales_fused, total_rows * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_separate));
    CUDA_CHECK(cudaFree(d_fused));
    CUDA_CHECK(cudaFree(d_q_separate));
    CUDA_CHECK(cudaFree(d_q_fused));
    CUDA_CHECK(cudaFree(d_scales_separate));
    CUDA_CHECK(cudaFree(d_scales_fused));

    const bool bit_exact = q_separate == q_fused;
    float max_scale_err = 0.0f;
    for (int row = 0; row < total_rows; ++row) {
        max_scale_err = std::max(max_scale_err, std::fabs(scales_separate[row] - scales_fused[row]));
    }

    const bool pass = bit_exact && max_scale_err < 1.0e-6f;
    std::cout << '[' << (pass ? "PASS" : "FAIL") << "] FP8 fused dtype=" << hadamard::dtype_name<T>()
              << " head_dim=" << head_dim
              << " total_rows=" << total_rows
              << " max_scale_err=" << std::scientific << std::setprecision(3) << max_scale_err
              << '\n';
    return pass;
}

}  // namespace

int run_fused_tests(const hadamard::cli_options& options) {
    const std::vector<int> head_dims = options.head_dim != 0 ? std::vector<int>{options.head_dim}
                                                             : std::vector<int>{64, 128, 256};
    const std::vector<int> total_rows_list = options.total_rows != 0 ? std::vector<int>{options.total_rows}
                                                                     : std::vector<int>{1024, 4096};

    bool all_pass = true;
    for (const int head_dim : head_dims) {
        for (const int total_rows : total_rows_list) {
            all_pass &= run_int4_case<half>(total_rows, head_dim);
            all_pass &= run_int4_case<__nv_bfloat16>(total_rows, head_dim);
            all_pass &= run_fp8_case<half>(total_rows, head_dim);
            all_pass &= run_fp8_case<__nv_bfloat16>(total_rows, head_dim);
        }
    }

    return all_pass ? 0 : 1;
}
