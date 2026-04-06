#include "hadamard.cuh"

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
bool run_case(int batch_size, int seq_len, int num_heads, int head_dim, bool use_wmma, float tolerance) {
    const int total_rows = batch_size * seq_len * num_heads;
    const size_t element_count = static_cast<size_t>(total_rows) * head_dim;

    std::mt19937 rng(static_cast<uint32_t>(head_dim * 1000003 + total_rows));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> reference_input(element_count);
    for (float& value : reference_input) {
        value = dist(rng);
    }

    std::vector<float> cpu_output = reference_input;
    hadamard::hadamard_transform_cpu_batch(cpu_output.data(), total_rows, head_dim);

    std::vector<T> host_input = to_device_vector<T>(reference_input);
    std::vector<T> host_output(element_count);

    T* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, element_count * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_data, host_input.data(), element_count * sizeof(T), cudaMemcpyHostToDevice));

    if (use_wmma) {
        hadamard::hadamard_transform_gpu_wmma(d_data, batch_size, seq_len, num_heads, head_dim);
    } else {
        hadamard::hadamard_transform_gpu(d_data, batch_size, seq_len, num_heads, head_dim);
    }

    CUDA_CHECK(cudaMemcpy(host_output.data(), d_data, element_count * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_data));

    float max_err = 0.0f;
    for (size_t index = 0; index < element_count; ++index) {
        const float gpu_value = hadamard::scalar_to_float(host_output[index]);
        max_err = std::max(max_err, std::fabs(cpu_output[index] - gpu_value));
    }

    const bool pass = max_err < tolerance;
    std::cout << '[' << (pass ? "PASS" : "FAIL") << "] "
              << hadamard::dtype_name<T>()
              << (use_wmma ? "-WMMA" : "")
              << " head_dim=" << head_dim
              << " batch=" << batch_size
              << " seq=" << seq_len
              << " heads=" << num_heads
              << " max_err=" << std::scientific << std::setprecision(3) << max_err
              << '\n';
    return pass;
}

}  // namespace

int run_correctness_tests(const hadamard::cli_options& options) {
    const std::vector<int> head_dims = options.head_dim != 0 ? std::vector<int>{options.head_dim}
                                                             : std::vector<int>{64, 128, 256};
    const std::vector<int> batch_sizes{1, 4};
    const std::vector<int> seq_lens{128, 512};
    const std::vector<int> num_heads_list{8, 32};

    bool all_pass = true;
    for (const int head_dim : head_dims) {
        for (const int batch_size : batch_sizes) {
            for (const int seq_len : seq_lens) {
                for (const int num_heads : num_heads_list) {
                    all_pass &= run_case<half>(batch_size, seq_len, num_heads, head_dim, false, 1.0e-2f);
                    all_pass &= run_case<__nv_bfloat16>(batch_size, seq_len, num_heads, head_dim, false, 5.0e-2f);
                    all_pass &= run_case<half>(batch_size, seq_len, num_heads, head_dim, true, 1.0e-2f);
                    all_pass &= run_case<__nv_bfloat16>(batch_size, seq_len, num_heads, head_dim, true, 5.0e-2f);
                }
            }
        }
    }

    return all_pass ? 0 : 1;
}
