#include "hadamard.cuh"
#include "quantize.cuh"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>

namespace {

template <typename T>
std::vector<T> make_input(int total_rows, int head_dim) {
    const size_t element_count = static_cast<size_t>(total_rows) * head_dim;
    std::mt19937 rng(static_cast<uint32_t>(head_dim * 65537 + total_rows));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<T> values(element_count);
    for (size_t index = 0; index < element_count; ++index) {
        values[index] = hadamard::float_to_scalar<T>(dist(rng));
    }
    return values;
}

double measure_ms(const std::function<void()>& launch, int warmup_iterations, int iterations) {
    for (int iter = 0; iter < warmup_iterations; ++iter) {
        launch();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < iterations; ++iter) {
        launch();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return static_cast<double>(elapsed_ms) / iterations;
}

double median_ms(std::vector<double> samples) {
    if (samples.empty()) {
        return 0.0;
    }

    std::sort(samples.begin(), samples.end());
    const size_t middle = samples.size() / 2;
    if ((samples.size() & 1U) != 0U) {
        return samples[middle];
    }
    return 0.5 * (samples[middle - 1] + samples[middle]);
}

double measure_ms_trials(const std::function<void()>& reset,
                         const std::function<void()>& launch,
                         int warmup_iterations,
                         int iterations,
                         int trials) {
    std::vector<double> samples;
    samples.reserve(trials);

    for (int trial = 0; trial < trials; ++trial) {
        reset();
        samples.push_back(measure_ms(launch, warmup_iterations, iterations));
    }

    return median_ms(std::move(samples));
}

std::vector<double> measure_interleaved_ms(const std::vector<std::function<void()>>& launches,
                                           int warmup_iterations,
                                           int iterations) {
    std::vector<double> elapsed_totals(launches.size(), 0.0);

    for (int iter = 0; iter < warmup_iterations; ++iter) {
        for (size_t slot = 0; slot < launches.size(); ++slot) {
            const size_t index = (static_cast<size_t>(iter) + slot) % launches.size();
            launches[index]();
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int iter = 0; iter < iterations; ++iter) {
        for (size_t slot = 0; slot < launches.size(); ++slot) {
            const size_t index = (static_cast<size_t>(iter) + slot) % launches.size();
            CUDA_CHECK(cudaEventRecord(start));
            launches[index]();
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float elapsed_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
            elapsed_totals[index] += elapsed_ms;
        }
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    for (double& total_ms : elapsed_totals) {
        total_ms /= iterations;
    }
    return elapsed_totals;
}

std::vector<double> measure_interleaved_ms_trials(const std::function<void()>& reset,
                                                  const std::vector<std::function<void()>>& launches,
                                                  int warmup_iterations,
                                                  int iterations,
                                                  int trials) {
    std::vector<std::vector<double>> per_kernel_samples(launches.size());

    for (int trial = 0; trial < trials; ++trial) {
        reset();
        const std::vector<double> trial_times = measure_interleaved_ms(launches, warmup_iterations, iterations);
        for (size_t index = 0; index < launches.size(); ++index) {
            per_kernel_samples[index].push_back(trial_times[index]);
        }
    }

    std::vector<double> medians(launches.size(), 0.0);
    for (size_t index = 0; index < launches.size(); ++index) {
        medians[index] = median_ms(std::move(per_kernel_samples[index]));
    }
    return medians;
}

double throughput_gbps(size_t bytes_moved, double milliseconds) {
    const double seconds = milliseconds / 1000.0;
    return seconds > 0.0 ? static_cast<double>(bytes_moved) / seconds / 1.0e9 : 0.0;
}

template <typename T>
void benchmark_dtype(const hadamard::cli_options& options, std::ofstream& csv) {
    if ((options.dtype_filter == "fp16" && !std::is_same_v<T, half>) ||
        (options.dtype_filter == "bf16" && !std::is_same_v<T, __nv_bfloat16>)) {
        return;
    }

    const std::vector<int> head_dims = options.head_dim != 0 ? std::vector<int>{options.head_dim}
                                                             : std::vector<int>{64, 128, 256};
    const std::vector<int> total_rows_list = options.total_rows != 0 ? std::vector<int>{options.total_rows}
                                                                     : std::vector<int>{4096, 16384, 65536};

    for (const int head_dim : head_dims) {
        for (const int total_rows : total_rows_list) {
            const size_t element_count = static_cast<size_t>(total_rows) * head_dim;
            const size_t packed_int4_count = static_cast<size_t>(total_rows) * (head_dim / 2);
            std::vector<T> host_input = make_input<T>(total_rows, head_dim);

            T* d_input = nullptr;
            T* d_input_source = nullptr;
            T* d_input_v1 = nullptr;
            T* d_input_v2 = nullptr;
            T* d_input_best = nullptr;
            uint8_t* d_q_int4 = nullptr;
            uint8_t* d_q_fp8 = nullptr;
            float* d_scales = nullptr;

            CUDA_CHECK(cudaMalloc(&d_input, element_count * sizeof(T)));
            CUDA_CHECK(cudaMalloc(&d_input_source, element_count * sizeof(T)));
            CUDA_CHECK(cudaMalloc(&d_input_v1, element_count * sizeof(T)));
            CUDA_CHECK(cudaMalloc(&d_input_v2, element_count * sizeof(T)));
            CUDA_CHECK(cudaMalloc(&d_input_best, element_count * sizeof(T)));
            CUDA_CHECK(cudaMalloc(&d_q_int4, packed_int4_count * sizeof(uint8_t)));
            CUDA_CHECK(cudaMalloc(&d_q_fp8, element_count * sizeof(uint8_t)));
            CUDA_CHECK(cudaMalloc(&d_scales, total_rows * sizeof(float)));

            CUDA_CHECK(cudaMemcpy(d_input, host_input.data(), element_count * sizeof(T), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_input_source, host_input.data(), element_count * sizeof(T), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_input_v1, host_input.data(), element_count * sizeof(T), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_input_v2, host_input.data(), element_count * sizeof(T), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_input_best, host_input.data(), element_count * sizeof(T), cudaMemcpyHostToDevice));

            const auto log_row = [&](const std::string& kernel_name, double time_ms, size_t bytes_moved) {
                const double gbps = throughput_gbps(bytes_moved, time_ms);
                csv << kernel_name << ','
                    << (std::is_same_v<T, half> ? "fp16" : "bf16") << ','
                    << head_dim << ','
                    << total_rows << ','
                    << std::fixed << std::setprecision(6) << time_ms << ','
                    << std::setprecision(3) << gbps << '\n';

                std::cout << kernel_name << ','
                          << (std::is_same_v<T, half> ? "fp16" : "bf16") << ','
                          << head_dim << ','
                          << total_rows << ','
                          << std::fixed << std::setprecision(6) << time_ms << ','
                          << std::setprecision(3) << gbps << '\n';
            };

            if (options.benchmark_mode == "all" || options.benchmark_mode == "hadamard") {
                const auto reset_hadamard_inputs = [&]() {
                    CUDA_CHECK(cudaMemcpy(d_input_v1, d_input_source, element_count * sizeof(T), cudaMemcpyDeviceToDevice));
                    CUDA_CHECK(cudaMemcpy(d_input_v2, d_input_source, element_count * sizeof(T), cudaMemcpyDeviceToDevice));
                    CUDA_CHECK(cudaMemcpy(d_input_best, d_input_source, element_count * sizeof(T), cudaMemcpyDeviceToDevice));
                };
                const std::vector<std::function<void()>> hadamard_launches{
                    [&]() { hadamard::hadamard_transform_gpu(d_input_v1, total_rows, 1, 1, head_dim); },
                    [&]() { hadamard::hadamard_transform_gpu_wmma(d_input_v2, total_rows, 1, 1, head_dim); },
                    [&]() { hadamard::hadamard_transform_gpu_best(d_input_best, total_rows, 1, 1, head_dim); }};

                const std::vector<double> hadamard_times =
                    measure_interleaved_ms_trials(reset_hadamard_inputs,
                                                 hadamard_launches,
                                                 options.warmup_iterations,
                                                 options.iterations,
                                                 options.measurement_trials);
                log_row("v1_butterfly", hadamard_times[0], element_count * sizeof(T) * 2);
                log_row("v2_wmma", hadamard_times[1], element_count * sizeof(T) * 2);
                log_row("best_auto", hadamard_times[2], element_count * sizeof(T) * 2);
            }

            if (options.benchmark_mode == "all" || options.benchmark_mode == "quant") {
                const auto reset_main_input = [&]() {
                    CUDA_CHECK(cudaMemcpy(d_input, d_input_source, element_count * sizeof(T), cudaMemcpyDeviceToDevice));
                };
                const double step_int4_ms = measure_ms_trials(
                    reset_main_input,
                    [&]() {
                        hadamard::hadamard_transform_gpu(d_input, total_rows, 1, 1, head_dim);
                        hadamard::quantize_int4_gpu(d_input, d_q_int4, d_scales, total_rows, head_dim);
                    },
                    options.warmup_iterations,
                    options.iterations,
                    options.measurement_trials);
                log_row("step_had_int4", step_int4_ms,
                        element_count * sizeof(T) * 3 + packed_int4_count * sizeof(uint8_t) + total_rows * sizeof(float));

                const double fused_int4_ms = measure_ms_trials(
                    reset_main_input,
                    [&]() { hadamard::hadamard_quantize_int4_gpu(d_input, d_q_int4, d_scales, total_rows, head_dim); },
                    options.warmup_iterations,
                    options.iterations,
                    options.measurement_trials);
                log_row("fused_had_int4", fused_int4_ms,
                        element_count * sizeof(T) + packed_int4_count * sizeof(uint8_t) + total_rows * sizeof(float));

                const double step_fp8_ms = measure_ms_trials(
                    reset_main_input,
                    [&]() {
                        hadamard::hadamard_transform_gpu(d_input, total_rows, 1, 1, head_dim);
                        hadamard::quantize_fp8_gpu(d_input, d_q_fp8, d_scales, total_rows, head_dim);
                    },
                    options.warmup_iterations,
                    options.iterations,
                    options.measurement_trials);
                log_row("step_had_fp8", step_fp8_ms,
                        element_count * sizeof(T) * 3 + element_count * sizeof(uint8_t) + total_rows * sizeof(float));

                const double fused_fp8_ms = measure_ms_trials(
                    reset_main_input,
                    [&]() { hadamard::hadamard_quantize_fp8_gpu(d_input, d_q_fp8, d_scales, total_rows, head_dim); },
                    options.warmup_iterations,
                    options.iterations,
                    options.measurement_trials);
                log_row("fused_had_fp8", fused_fp8_ms,
                        element_count * sizeof(T) + element_count * sizeof(uint8_t) + total_rows * sizeof(float));
            }

            CUDA_CHECK(cudaFree(d_input));
            CUDA_CHECK(cudaFree(d_input_source));
            CUDA_CHECK(cudaFree(d_input_v1));
            CUDA_CHECK(cudaFree(d_input_v2));
            CUDA_CHECK(cudaFree(d_input_best));
            CUDA_CHECK(cudaFree(d_q_int4));
            CUDA_CHECK(cudaFree(d_q_fp8));
            CUDA_CHECK(cudaFree(d_scales));
        }
    }
}

}  // namespace

int run_benchmark(const hadamard::cli_options& options) {
    std::filesystem::create_directories(std::filesystem::path(options.csv_path).parent_path());
    std::ofstream csv(options.csv_path, std::ios::trunc);
    if (!csv) {
        throw std::runtime_error("Failed to open benchmark CSV output: " + options.csv_path);
    }

    csv << "kernel,dtype,head_dim,total_rows,time_ms,throughput_GB_s\n";
    std::cout << "kernel,dtype,head_dim,total_rows,time_ms,throughput_GB_s\n";

    benchmark_dtype<half>(options, csv);
    benchmark_dtype<__nv_bfloat16>(options, csv);

    return 0;
}
