#include "hadamard.cuh"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace hadamard {

void print_usage(const char* exe_name) {
    std::cout << "Usage: " << exe_name << " [--iterations N] [--warmup N] [--trials N] [--head-dim N] "
              << "[--total-rows N] [--dtype all|fp16|bf16] [--mode all|hadamard|quant] "
              << "[--csv PATH] [--verbose]\n";
}

cli_options parse_cli_options(int argc, char** argv) {
    cli_options options;
    options.csv_path = std::string(HADAMARD_PROJECT_ROOT) + "/bench/results.csv";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string("Missing value for ") + flag);
            }
            return argv[++i];
        };

        if (arg == "--iterations") {
            options.iterations = std::atoi(require_value("--iterations"));
        } else if (arg == "--warmup") {
            options.warmup_iterations = std::atoi(require_value("--warmup"));
        } else if (arg == "--trials") {
            options.measurement_trials = std::atoi(require_value("--trials"));
        } else if (arg == "--head-dim") {
            options.head_dim = std::atoi(require_value("--head-dim"));
        } else if (arg == "--total-rows") {
            options.total_rows = std::atoi(require_value("--total-rows"));
        } else if (arg == "--dtype") {
            options.dtype_filter = require_value("--dtype");
        } else if (arg == "--mode") {
            options.benchmark_mode = require_value("--mode");
        } else if (arg == "--csv") {
            options.csv_path = require_value("--csv");
        } else if (arg == "--verbose") {
            options.verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("Unknown option: " + arg);
        }
    }

    if (options.dtype_filter != "all" && options.dtype_filter != "fp16" && options.dtype_filter != "bf16") {
        throw std::invalid_argument("dtype must be one of: all, fp16, bf16");
    }

    if (options.benchmark_mode != "all" && options.benchmark_mode != "hadamard" &&
        options.benchmark_mode != "quant") {
        throw std::invalid_argument("mode must be one of: all, hadamard, quant");
    }

    if (options.measurement_trials <= 0) {
        throw std::invalid_argument("trials must be > 0");
    }

    return options;
}

}  // namespace hadamard

#if defined(HADAMARD_BUILD_TEST)
int run_correctness_tests(const hadamard::cli_options& options);
int run_fused_tests(const hadamard::cli_options& options);

int main(int argc, char** argv) {
    try {
        const hadamard::cli_options options = hadamard::parse_cli_options(argc, argv);
        const int rc_correctness = run_correctness_tests(options);
        const int rc_fused = run_fused_tests(options);
        return (rc_correctness == 0 && rc_fused == 0) ? 0 : 1;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
#elif defined(HADAMARD_BUILD_BENCH)
int run_benchmark(const hadamard::cli_options& options);

int main(int argc, char** argv) {
    try {
        return run_benchmark(hadamard::parse_cli_options(argc, argv));
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
#else
#error "Either HADAMARD_BUILD_TEST or HADAMARD_BUILD_BENCH must be defined."
#endif
