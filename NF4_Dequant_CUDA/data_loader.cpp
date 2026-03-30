#include "nf4_dequant.h"

#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>

#include <cuda_runtime.h>

namespace {

constexpr size_t kHeaderBytes = sizeof(int64_t) + sizeof(int64_t) + sizeof(int32_t);
constexpr size_t kCode2Entries = 256;

bool safe_mul_size_t(size_t a, size_t b, size_t* out) {
    if (out == nullptr) {
        return false;
    }
    if (a == 0 || b == 0) {
        *out = 0;
        return true;
    }
    if (a > (std::numeric_limits<size_t>::max() / b)) {
        return false;
    }
    *out = a * b;
    return true;
}

bool read_exact(FILE* fp, void* dst, size_t bytes) {
    if (bytes == 0) {
        return true;
    }
    return std::fread(dst, 1, bytes, fp) == bytes;
}

bool write_exact(FILE* fp, const void* src, size_t bytes) {
    if (bytes == 0) {
        return true;
    }
    return std::fwrite(src, 1, bytes, fp) == bytes;
}

template <typename T>
T* alloc_count(size_t count) {
    if (count == 0) {
        return nullptr;
    }
    if (count > (std::numeric_limits<size_t>::max() / sizeof(T))) {
        return nullptr;
    }
    return static_cast<T*>(std::malloc(count * sizeof(T)));
}

template <typename T>
T* alloc_host_count(size_t count, bool* pinned, bool allow_pinned) {
    if (pinned == nullptr) {
        return nullptr;
    }
    (void)allow_pinned;
    *pinned = false;
    if (count == 0) {
        return nullptr;
    }
    if (count > (std::numeric_limits<size_t>::max() / sizeof(T))) {
        return nullptr;
    }
    return alloc_count<T>(count);
}

bool try_register_host_buffer(void* ptr, size_t bytes, bool allow_pinned, bool* pinned, const char* label) {
    if (pinned == nullptr) {
        return false;
    }
    *pinned = false;
    if (!allow_pinned || ptr == nullptr || bytes == 0) {
        return true;
    }

    const cudaError_t err = cudaHostRegister(ptr, bytes, cudaHostRegisterDefault);
    if (err == cudaSuccess) {
        *pinned = true;
        return true;
    }

    std::fprintf(
        stderr,
        "Warning: cudaHostRegister failed for %s (%s). Continuing with pageable host memory.\n",
        label != nullptr ? label : "buffer",
        cudaGetErrorString(err));
    return true;
}

template <typename T>
void free_host_count(T* ptr, bool pinned) {
    if (ptr == nullptr) {
        return;
    }
    if (pinned) {
        cudaHostUnregister(ptr);
    }
    std::free(ptr);
}

}  // namespace

bool load_nf4_file(const char* bin_path, NF4QuantState* state, bool use_pinned_host_input) {
    if (bin_path == nullptr || state == nullptr) {
        std::fprintf(stderr, "[load_nf4_file] invalid input pointer.\n");
        return false;
    }

    free_nf4_state(state);
    state->blocks_per_group = 256;

    std::error_code ec;
    const uintmax_t file_size_u = std::filesystem::file_size(bin_path, ec);
    if (ec) {
        std::fprintf(stderr, "[load_nf4_file] cannot get file size: %s\n", ec.message().c_str());
        return false;
    }
    if (file_size_u < kHeaderBytes || file_size_u > std::numeric_limits<size_t>::max()) {
        std::fprintf(stderr, "[load_nf4_file] invalid file size: %llu\n", static_cast<unsigned long long>(file_size_u));
        return false;
    }
    const size_t file_size = static_cast<size_t>(file_size_u);

    FILE* fp = std::fopen(bin_path, "rb");
    if (fp == nullptr) {
        std::fprintf(stderr, "[load_nf4_file] fopen failed: %s\n", std::strerror(errno));
        return false;
    }

    int64_t rows = 0;
    int64_t cols = 0;
    int32_t blocksize = 0;
    bool ok = false;
    size_t groups_from_file = 0;

    do {
        if (!read_exact(fp, &rows, sizeof(rows)) ||
            !read_exact(fp, &cols, sizeof(cols)) ||
            !read_exact(fp, &blocksize, sizeof(blocksize))) {
            std::fprintf(stderr, "[load_nf4_file] failed to read header.\n");
            break;
        }

        if (rows <= 0 || cols <= 0 || blocksize <= 0) {
            std::fprintf(stderr, "[load_nf4_file] invalid header values. rows=%lld cols=%lld blocksize=%d\n",
                static_cast<long long>(rows), static_cast<long long>(cols), static_cast<int>(blocksize));
            break;
        }

        size_t num_elements = 0;
        if (!safe_mul_size_t(static_cast<size_t>(rows), static_cast<size_t>(cols), &num_elements)) {
            std::fprintf(stderr, "[load_nf4_file] rows*cols overflow.\n");
            break;
        }

        const size_t num_packed = (num_elements + 1) / 2;
        const size_t num_blocks = (num_elements + static_cast<size_t>(blocksize) - 1) / static_cast<size_t>(blocksize);

        const size_t code2_bytes = kCode2Entries * sizeof(__half);
        const size_t fixed_tail = code2_bytes + sizeof(float);
        const size_t payload_bytes = file_size - kHeaderBytes;
        const size_t fixed_prefix = num_packed + num_blocks;

        if (payload_bytes < fixed_prefix + fixed_tail) {
            std::fprintf(stderr, "[load_nf4_file] file too small for inferred layout.\n");
            break;
        }

        const size_t absmax2_bytes = payload_bytes - fixed_prefix - fixed_tail;
        if ((absmax2_bytes % sizeof(__half)) != 0) {
            std::fprintf(stderr, "[load_nf4_file] absmax2 payload misaligned: %zu bytes.\n", absmax2_bytes);
            break;
        }
        groups_from_file = absmax2_bytes / sizeof(__half);

        state->num_rows = rows;
        state->num_cols = cols;
        state->blocksize = blocksize;
        state->num_elements = num_elements;
        state->num_packed_bytes = num_packed;
        state->num_blocks = num_blocks;
        state->num_groups = groups_from_file;
        if (state->num_groups == 0) {
            state->num_groups = 1;
        }

        state->h_packed_weights = alloc_host_count<uint8_t>(
            state->num_packed_bytes, &state->h_packed_weights_pinned, use_pinned_host_input);
        state->h_absmax_q = alloc_host_count<uint8_t>(
            state->num_blocks, &state->h_absmax_q_pinned, use_pinned_host_input);
        state->h_absmax2 = alloc_host_count<__half>(
            state->num_groups, &state->h_absmax2_pinned, use_pinned_host_input);
        state->h_code2 = alloc_host_count<__half>(
            kCode2Entries, &state->h_code2_pinned, use_pinned_host_input);

        if (state->h_packed_weights == nullptr ||
            state->h_absmax_q == nullptr ||
            state->h_absmax2 == nullptr ||
            state->h_code2 == nullptr) {
            std::fprintf(stderr, "[load_nf4_file] host allocation failed.\n");
            break;
        }

        if (!read_exact(fp, state->h_packed_weights, state->num_packed_bytes)) {
            std::fprintf(stderr, "[load_nf4_file] failed to read packed_weights.\n");
            break;
        }
        if (!read_exact(fp, state->h_absmax_q, state->num_blocks)) {
            std::fprintf(stderr, "[load_nf4_file] failed to read absmax_q.\n");
            break;
        }

        if (groups_from_file > 0) {
            if (!read_exact(fp, state->h_absmax2, groups_from_file * sizeof(__half))) {
                std::fprintf(stderr, "[load_nf4_file] failed to read absmax2.\n");
                break;
            }
        } else {
            state->h_absmax2[0] = __float2half(1.0f);
        }

        if (!read_exact(fp, state->h_code2, code2_bytes)) {
            std::fprintf(stderr, "[load_nf4_file] failed to read code2.\n");
            break;
        }

        if (!read_exact(fp, &state->h_offset, sizeof(float))) {
            std::fprintf(stderr, "[load_nf4_file] failed to read offset.\n");
            break;
        }

        if (!try_register_host_buffer(
                state->h_packed_weights,
                state->num_packed_bytes,
                use_pinned_host_input,
                &state->h_packed_weights_pinned,
                "packed_weights") ||
            !try_register_host_buffer(
                state->h_absmax_q,
                state->num_blocks * sizeof(uint8_t),
                use_pinned_host_input,
                &state->h_absmax_q_pinned,
                "absmax_q") ||
            !try_register_host_buffer(
                state->h_absmax2,
                state->num_groups * sizeof(__half),
                use_pinned_host_input,
                &state->h_absmax2_pinned,
                "absmax2") ||
            !try_register_host_buffer(
                state->h_code2,
                kCode2Entries * sizeof(__half),
                use_pinned_host_input,
                &state->h_code2_pinned,
                "code2")) {
            break;
        }

        ok = true;
    } while (false);

    std::fclose(fp);

    if (!ok) {
        free_nf4_state(state);
    }
    return ok;
}

void free_nf4_state(NF4QuantState* state) {
    if (state == nullptr) {
        return;
    }

    free_host_count(state->h_packed_weights, state->h_packed_weights_pinned);
    free_host_count(state->h_absmax_q, state->h_absmax_q_pinned);
    free_host_count(state->h_absmax2, state->h_absmax2_pinned);
    free_host_count(state->h_code2, state->h_code2_pinned);

    state->h_packed_weights = nullptr;
    state->h_absmax_q = nullptr;
    state->h_absmax2 = nullptr;
    state->h_code2 = nullptr;
    state->h_packed_weights_pinned = false;
    state->h_absmax_q_pinned = false;
    state->h_absmax2_pinned = false;
    state->h_code2_pinned = false;

    state->num_rows = 0;
    state->num_cols = 0;
    state->blocksize = 0;
    state->blocks_per_group = 256;
    state->h_offset = 0.0f;
    state->num_elements = 0;
    state->num_packed_bytes = 0;
    state->num_blocks = 0;
    state->num_groups = 0;
}

void save_dequant(const void* data, int64_t rows, int64_t cols, const char* out_path, bool is_bf16) {
    (void)is_bf16;
    if (data == nullptr || out_path == nullptr || rows <= 0 || cols <= 0) {
        std::fprintf(stderr, "[save_dequant] invalid arguments.\n");
        return;
    }

    size_t num_elements = 0;
    if (!safe_mul_size_t(static_cast<size_t>(rows), static_cast<size_t>(cols), &num_elements)) {
        std::fprintf(stderr, "[save_dequant] rows*cols overflow.\n");
        return;
    }

    size_t bytes = 0;
    if (!safe_mul_size_t(num_elements, sizeof(uint16_t), &bytes)) {
        std::fprintf(stderr, "[save_dequant] byte size overflow.\n");
        return;
    }

    FILE* fp = std::fopen(out_path, "wb");
    if (fp == nullptr) {
        std::fprintf(stderr, "[save_dequant] fopen failed: %s\n", std::strerror(errno));
        return;
    }

    const bool ok = write_exact(fp, data, bytes);
    std::fclose(fp);

    if (!ok) {
        std::fprintf(stderr, "[save_dequant] failed to write output.\n");
    }
}
