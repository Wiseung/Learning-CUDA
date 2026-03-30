#ifndef NF4_DEQUANT_H_
#define NF4_DEQUANT_H_

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

enum class NF4KernelVariant : int {
    kAuto = 0,
    kGeneric = 1,
    kSpecialized = 2,
};

struct NF4QuantState {
    int64_t num_rows = 0;
    int64_t num_cols = 0;
    int32_t blocksize = 0;
    int32_t blocks_per_group = 256;

    uint8_t* h_packed_weights = nullptr;
    uint8_t* h_absmax_q = nullptr;
    __half* h_absmax2 = nullptr;
    __half* h_code2 = nullptr;
    bool h_packed_weights_pinned = false;
    bool h_absmax_q_pinned = false;
    bool h_absmax2_pinned = false;
    bool h_code2_pinned = false;
    float h_offset = 0.0f;

    size_t num_elements = 0;
    size_t num_packed_bytes = 0;
    size_t num_blocks = 0;
    size_t num_groups = 0;
};

bool load_nf4_file(const char* bin_path, NF4QuantState* state, bool use_pinned_host_input = true);
void free_nf4_state(NF4QuantState* state);
void save_dequant(const void* data, int64_t rows, int64_t cols, const char* out_path, bool is_bf16);
void cpu_dequant_nf4(const NF4QuantState& state, void* output, bool use_bf16);
bool cuda_dequant_nf4(
    const NF4QuantState& state,
    void* output,
    bool use_bf16,
    float* kernel_time_ms,
    int block_dim = 256,
    bool copy_output_to_host = true,
    bool reuse_device_buffers = true,
    bool use_cuda_graph = false,
    NF4KernelVariant kernel_variant = NF4KernelVariant::kAuto,
    float* end_to_end_ms = nullptr,
    float* graph_build_time_ms = nullptr);
void cuda_release_nf4_device_cache();

#endif  // NF4_DEQUANT_H_
