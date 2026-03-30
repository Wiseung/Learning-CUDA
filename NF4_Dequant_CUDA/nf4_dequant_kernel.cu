#include "nf4_dequant.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <cuda_runtime.h>

namespace {

constexpr int kNF4TableSize = 16;
constexpr int kCode2Entries = 256;

// QLoRA Appendix E NF4 codebook.
constexpr float kNF4TableHost[kNF4TableSize] = {
    -1.0f,
    -0.6961928009986877f,
    -0.5250730514526367f,
    -0.39491748809814453f,
    -0.28444138169288635f,
    -0.18477343022823334f,
    -0.09105003625154495f,
    0.0f,
    0.07958029955625534f,
    0.16093020141124725f,
    0.24611230194568634f,
    0.33791524171829224f,
    0.44070982933044434f,
    0.5626170039176941f,
    0.7229568362236023f,
    1.0f,
};

}  // namespace

__device__ const __half nf4_table[kNF4TableSize] = {
    __half_raw{0xBC00},
    __half_raw{0xB992},
    __half_raw{0xB833},
    __half_raw{0xB652},
    __half_raw{0xB48D},
    __half_raw{0xB1EA},
    __half_raw{0xADD4},
    __half_raw{0x0000},
    __half_raw{0x2D18},
    __half_raw{0x3126},
    __half_raw{0x33E0},
    __half_raw{0x3568},
    __half_raw{0x370D},
    __half_raw{0x3880},
    __half_raw{0x39C9},
    __half_raw{0x3C00},
};

template <int kBlocksPerGroup = 0>
__device__ __forceinline__ __half load_scale_from_double_quant(
    const uint8_t* __restrict__ absmax_q,
    const __half* __restrict__ absmax2,
    const __half* __restrict__ code2,
    const __half offset_h,
    const int32_t safe_blocks_per_group,
    const size_t block_id) {
    size_t group_id = 0;
    if constexpr (kBlocksPerGroup == 256) {
        group_id = block_id >> 8;
    } else if constexpr (kBlocksPerGroup > 0) {
        group_id = block_id / static_cast<size_t>(kBlocksPerGroup);
    } else {
        group_id = block_id / static_cast<size_t>(safe_blocks_per_group);
    }
    const uint8_t qidx = __ldg(absmax_q + block_id);
    const __half code = __ldg(code2 + qidx);
    const __half abs2 = __ldg(absmax2 + group_id);
    const __half mul = __hmul(code, abs2);
    return __hadd(mul, offset_h);
}

__device__ __forceinline__ uint16_t half_to_bits(const __half x) {
    union HalfBits {
        __half h;
        uint16_t bits;
    };
    HalfBits raw{};
    raw.h = x;
    return raw.bits;
}

__device__ __forceinline__ __half half_from_bits(const uint16_t bits) {
    union HalfBits {
        __half h;
        uint16_t bits;
    };
    HalfBits raw{};
    raw.bits = bits;
    return raw.h;
}

template <int kBlocksPerGroup = 0>
__device__ __forceinline__ __half load_scale_warp_broadcast(
    const uint8_t* __restrict__ absmax_q,
    const __half* __restrict__ absmax2,
    const __half* __restrict__ code2,
    const __half offset_h,
    const int32_t safe_blocks_per_group,
    const size_t block_id,
    const unsigned int active_mask,
    const int lane) {
    const int leader_lane = __ffs(static_cast<int>(active_mask)) - 1;
    uint32_t scale_bits = 0U;
    if (lane == leader_lane) {
        const __half scale = load_scale_from_double_quant<kBlocksPerGroup>(
            absmax_q, absmax2, code2, offset_h, safe_blocks_per_group, block_id);
        scale_bits = static_cast<uint32_t>(half_to_bits(scale));
    }
    scale_bits = __shfl_sync(active_mask, scale_bits, leader_lane);
    return half_from_bits(static_cast<uint16_t>(scale_bits));
}

__device__ __forceinline__ uint32_t pack_half2_to_u32(const __half v0, const __half v1) {
    const __half2 h2 = __halves2half2(v0, v1);
    const __half2_raw raw = static_cast<__half2_raw>(h2);
    return (static_cast<uint32_t>(raw.y) << 16) | static_cast<uint32_t>(raw.x);
}

__device__ __forceinline__ __nv_bfloat16 half_to_bf16(const __half x) {
    return __float2bfloat16(__half2float(x));
}

__device__ __forceinline__ uint32_t pack_bf16x2_to_u32(const __half v0, const __half v1) {
    const __nv_bfloat16 b0 = half_to_bf16(v0);
    const __nv_bfloat16 b1 = half_to_bf16(v1);
    const __nv_bfloat162 p = __halves2bfloat162(b0, b1);
    const __nv_bfloat162_raw raw = static_cast<__nv_bfloat162_raw>(p);
    return (static_cast<uint32_t>(raw.y) << 16) | static_cast<uint32_t>(raw.x);
}

template <bool kUseBf16>
__device__ __forceinline__ void store_scalar_output(void* output, const size_t idx, const __half value) {
    if constexpr (kUseBf16) {
        reinterpret_cast<__nv_bfloat16*>(output)[idx] = half_to_bf16(value);
    } else {
        reinterpret_cast<__half*>(output)[idx] = value;
    }
}

template <bool kUseBf16>
__device__ __forceinline__ void store_packed_output(
    void* output,
    const size_t idx,
    const __half val0,
    const __half val1) {
    uint32_t packed_out = 0U;
    if constexpr (kUseBf16) {
        packed_out = pack_bf16x2_to_u32(val0, val1);
    } else {
        packed_out = pack_half2_to_u32(val0, val1);
    }

    uint32_t* out_u32 = reinterpret_cast<uint32_t*>(
        reinterpret_cast<char*>(output) + idx * sizeof(uint16_t));
    *out_u32 = packed_out;
}

template <int kBlockSize>
__device__ __forceinline__ size_t fixed_block_id_from_idx(const size_t idx) {
    static_assert(kBlockSize == 64 || kBlockSize == 128, "fixed_block_id_from_idx expects blocksize 64 or 128");
    if constexpr (kBlockSize == 64) {
        return idx >> 6;
    } else {
        return idx >> 7;
    }
}

template <int kBlockSize, bool kUseBf16, int kBlocksPerGroup = 0>
__global__ void nf4_dequant_kernel_fixed_blocksize(
    const uint8_t* __restrict__ packed_weights,
    const uint8_t* __restrict__ absmax_q,
    const __half* __restrict__ absmax2,
    const __half* __restrict__ code2,
    float offset,
    void* __restrict__ output,
    int64_t num_elements,
    int32_t blocks_per_group) {
    const size_t n = static_cast<size_t>(num_elements);
    const size_t idx = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 2ULL;
    if (idx >= n) {
        return;
    }

    const int lane = static_cast<int>(threadIdx.x & 31);
    const unsigned int active_mask = __activemask();
    const int32_t safe_bpg =
        kBlocksPerGroup > 0 ? kBlocksPerGroup : (blocks_per_group > 0 ? blocks_per_group : 1);
    const __half offset_h = __float2half_rn(offset);
    const size_t warp_base_idx = idx - static_cast<size_t>(lane) * 2ULL;
    const size_t block_id = fixed_block_id_from_idx<kBlockSize>(warp_base_idx);
    const __half scale = load_scale_warp_broadcast<kBlocksPerGroup>(
        absmax_q, absmax2, code2, offset_h, safe_bpg, block_id, active_mask, lane);

    const uint8_t byte = __ldg(packed_weights + (idx >> 1));
    const uint8_t idx0 = static_cast<uint8_t>(byte >> 4);
    const uint8_t idx1 = static_cast<uint8_t>(byte & 0x0F);
    const __half val0 = __hmul(nf4_table[idx0], scale);

    if (idx + 1ULL >= n) {
        store_scalar_output<kUseBf16>(output, idx, val0);
        return;
    }

    const __half val1 = __hmul(nf4_table[idx1], scale);
    store_packed_output<kUseBf16>(output, idx, val0, val1);
}

template <bool kUseBf16>
__global__ void nf4_dequant_kernel_impl(
    const uint8_t* __restrict__ packed_weights,
    const uint8_t* __restrict__ absmax_q,
    const __half* __restrict__ absmax2,
    const __half* __restrict__ code2,
    float offset,
    void* __restrict__ output,
    int64_t num_elements,
    int32_t blocksize,
    int32_t blocks_per_group) {
    const size_t n = static_cast<size_t>(num_elements);
    const size_t idx = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 2ULL;
    if (idx >= n) {
        return;
    }
    const size_t blocksize_sz = static_cast<size_t>(blocksize > 0 ? blocksize : 1);
    const bool blocksize_even = (blocksize_sz & 1ULL) == 0ULL;
    const int32_t safe_bpg = blocks_per_group > 0 ? blocks_per_group : 1;
    const __half offset_h = __float2half_rn(offset);
    const int lane = static_cast<int>(threadIdx.x & 31);
    const unsigned int active_mask = __activemask();
    const bool use_warp_scale_broadcast = (blocksize == 64 || blocksize == 128);
    const size_t warp_base_idx = idx - static_cast<size_t>(lane) * 2ULL;

    const uint8_t byte = __ldg(packed_weights + (idx >> 1));
    const uint8_t idx0 = static_cast<uint8_t>(byte >> 4);    // high nibble first
    const uint8_t idx1 = static_cast<uint8_t>(byte & 0x0F);  // low nibble second

    size_t block_id0 = 0;
    __half scale0;
    if (use_warp_scale_broadcast) {
        const size_t warp_block_id = warp_base_idx / blocksize_sz;
        scale0 = load_scale_warp_broadcast<>(
            absmax_q, absmax2, code2, offset_h, safe_bpg, warp_block_id, active_mask, lane);
    } else {
        block_id0 = idx / blocksize_sz;
        scale0 = load_scale_from_double_quant<>(
            absmax_q, absmax2, code2, offset_h, safe_bpg, block_id0);
    }
    const __half val0 = __hmul(nf4_table[idx0], scale0);

    if (idx + 1ULL >= n) {
        store_scalar_output<kUseBf16>(output, idx, val0);
        return;
    }

    __half scale1 = scale0;
    if (!use_warp_scale_broadcast && !blocksize_even) {
        const size_t block_id1 = (idx + 1ULL) / blocksize_sz;
        if (block_id1 != block_id0) {
            scale1 = load_scale_from_double_quant<>(
                absmax_q, absmax2, code2, offset_h, safe_bpg, block_id1);
        }
    }
    const __half val1 = __hmul(nf4_table[idx1], scale1);
    store_packed_output<kUseBf16>(output, idx, val0, val1);
}

void cpu_dequant_nf4(const NF4QuantState& state, void* output, bool use_bf16) {
    if (output == nullptr) {
        std::fprintf(stderr, "[cpu_dequant_nf4] output pointer is null.\n");
        return;
    }
    if (state.h_packed_weights == nullptr || state.h_absmax_q == nullptr ||
        state.h_absmax2 == nullptr || state.h_code2 == nullptr) {
        std::fprintf(stderr, "[cpu_dequant_nf4] quantized buffers are not initialized.\n");
        return;
    }
    if (state.blocksize <= 0 || state.num_elements == 0 || state.num_blocks == 0) {
        std::fprintf(stderr, "[cpu_dequant_nf4] invalid tensor/block metadata.\n");
        return;
    }

    const size_t num_elements = state.num_elements;
    const size_t num_groups = std::max<size_t>(state.num_groups, 1);
    const size_t blocks_per_group = static_cast<size_t>(
        state.blocks_per_group > 0 ? state.blocks_per_group : 256);
    const size_t blocksize = static_cast<size_t>(state.blocksize);

    auto* out_fp16 = static_cast<__half*>(output);
    auto* out_bf16 = static_cast<__nv_bfloat16*>(output);

    for (size_t i = 0; i < num_elements; ++i) {
        const size_t block_id = i / blocksize;
        size_t group_id = block_id / blocks_per_group;
        if (group_id >= num_groups) {
            group_id = num_groups - 1;
        }

        const uint8_t code_idx = state.h_absmax_q[block_id];
        const float code2 = __half2float(state.h_code2[code_idx]);
        const float absmax2 = __half2float(state.h_absmax2[group_id]);
        const float scale = code2 * absmax2 + state.h_offset;

        const uint8_t packed = state.h_packed_weights[i >> 1];
        const uint8_t nibble = ((i & 1U) == 0U) ? static_cast<uint8_t>(packed >> 4)
                                                : static_cast<uint8_t>(packed & 0x0FU);
        const float value = kNF4TableHost[nibble] * scale;

        if (use_bf16) {
            out_bf16[i] = __float2bfloat16(value);
        } else {
            out_fp16[i] = __float2half(value);
        }
    }
}

namespace {

bool check_cuda(cudaError_t err, const char* expr) {
    if (err == cudaSuccess) {
        return true;
    }
    std::fprintf(stderr, "[CUDA] %s failed: %s\n", expr, cudaGetErrorString(err));
    return false;
}

template <typename T>
bool ensure_capacity(T** ptr, size_t* capacity_bytes, size_t required_bytes, const char* malloc_expr) {
    if (ptr == nullptr || capacity_bytes == nullptr) {
        return false;
    }
    if (*capacity_bytes >= required_bytes) {
        return true;
    }
    if (*ptr != nullptr) {
        if (!check_cuda(cudaFree(*ptr), "cudaFree(resize)")) {
            return false;
        }
        *ptr = nullptr;
        *capacity_bytes = 0;
    }
    if (required_bytes == 0) {
        return true;
    }
    if (!check_cuda(cudaMalloc(reinterpret_cast<void**>(ptr), required_bytes), malloc_expr)) {
        return false;
    }
    *capacity_bytes = required_bytes;
    return true;
}

struct DeviceCache {
    uint8_t* d_packed_weights = nullptr;
    uint8_t* d_absmax_q = nullptr;
    __half* d_absmax2 = nullptr;
    __half* d_code2 = nullptr;
    void* d_output = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraphExec_t graph_exec = nullptr;

    size_t packed_capacity = 0;
    size_t absmax_q_capacity = 0;
    size_t absmax2_capacity = 0;
    size_t code2_capacity = 0;
    size_t output_capacity = 0;

    bool inputs_uploaded = false;
    uintptr_t state_signature = 0;
    uintptr_t graph_signature = 0;
    size_t num_blocks = 0;
    size_t num_groups = 0;
    bool graph_valid = false;
};

DeviceCache g_device_cache;

void invalidate_cached_graph(DeviceCache* cache) {
    if (cache == nullptr) {
        return;
    }
    if (cache->graph_exec != nullptr) {
        cudaGraphExecDestroy(cache->graph_exec);
        cache->graph_exec = nullptr;
    }
    cache->graph_signature = 0;
    cache->graph_valid = false;
}

void release_device_cache(DeviceCache* cache) {
    if (cache == nullptr) {
        return;
    }
    invalidate_cached_graph(cache);
    if (cache->d_packed_weights != nullptr) {
        cudaFree(cache->d_packed_weights);
    }
    if (cache->d_absmax_q != nullptr) {
        cudaFree(cache->d_absmax_q);
    }
    if (cache->d_absmax2 != nullptr) {
        cudaFree(cache->d_absmax2);
    }
    if (cache->d_code2 != nullptr) {
        cudaFree(cache->d_code2);
    }
    if (cache->d_output != nullptr) {
        cudaFree(cache->d_output);
    }
    if (cache->stream != nullptr) {
        cudaStreamDestroy(cache->stream);
    }
    *cache = DeviceCache{};
}

bool ensure_cache_capacity(
    DeviceCache* cache,
    size_t packed_bytes,
    size_t absmax_q_bytes,
    size_t absmax2_bytes,
    size_t code2_bytes,
    size_t output_bytes,
    bool use_async_stream) {
    if (cache == nullptr) {
        return false;
    }
    if (use_async_stream && cache->stream == nullptr) {
        if (!check_cuda(cudaStreamCreateWithFlags(&cache->stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags")) {
            return false;
        }
    }
    const size_t prev_packed = cache->packed_capacity;
    const size_t prev_absmax_q = cache->absmax_q_capacity;
    const size_t prev_absmax2 = cache->absmax2_capacity;
    const size_t prev_code2 = cache->code2_capacity;
    const size_t prev_output = cache->output_capacity;

    if (!ensure_capacity(&cache->d_packed_weights, &cache->packed_capacity, packed_bytes, "cudaMalloc(d_packed_weights)") ||
        !ensure_capacity(&cache->d_absmax_q, &cache->absmax_q_capacity, absmax_q_bytes, "cudaMalloc(d_absmax_q)") ||
        !ensure_capacity(&cache->d_absmax2, &cache->absmax2_capacity, absmax2_bytes, "cudaMalloc(d_absmax2)") ||
        !ensure_capacity(&cache->d_code2, &cache->code2_capacity, code2_bytes, "cudaMalloc(d_code2)") ||
        !ensure_capacity(reinterpret_cast<uint8_t**>(&cache->d_output), &cache->output_capacity, output_bytes, "cudaMalloc(d_output)")) {
        return false;
    }

    if (cache->packed_capacity != prev_packed || cache->absmax_q_capacity != prev_absmax_q ||
        cache->absmax2_capacity != prev_absmax2 || cache->code2_capacity != prev_code2 ||
        cache->output_capacity != prev_output) {
        cache->inputs_uploaded = false;
        invalidate_cached_graph(cache);
    }
    return true;
}

inline uintptr_t mix_u64(uintptr_t seed, uintptr_t value) {
    seed ^= value + static_cast<uintptr_t>(0x9e3779b97f4a7c15ULL) + (seed << 6U) + (seed >> 2U);
    return seed;
}

uintptr_t compute_state_signature(
    const NF4QuantState& state,
    size_t packed_bytes,
    size_t absmax_q_bytes,
    size_t absmax2_bytes,
    size_t code2_bytes) {
    uintptr_t sig = reinterpret_cast<uintptr_t>(state.h_packed_weights);
    sig = mix_u64(sig, reinterpret_cast<uintptr_t>(state.h_absmax_q));
    sig = mix_u64(sig, reinterpret_cast<uintptr_t>(state.h_absmax2));
    sig = mix_u64(sig, reinterpret_cast<uintptr_t>(state.h_code2));
    sig = mix_u64(sig, static_cast<uintptr_t>(state.num_elements));
    sig = mix_u64(sig, static_cast<uintptr_t>(state.num_blocks));
    sig = mix_u64(sig, static_cast<uintptr_t>(state.num_groups));
    sig = mix_u64(sig, static_cast<uintptr_t>(state.blocksize));
    sig = mix_u64(sig, static_cast<uintptr_t>(state.blocks_per_group));
    sig = mix_u64(sig, static_cast<uintptr_t>(packed_bytes));
    sig = mix_u64(sig, static_cast<uintptr_t>(absmax_q_bytes));
    sig = mix_u64(sig, static_cast<uintptr_t>(absmax2_bytes));
    sig = mix_u64(sig, static_cast<uintptr_t>(code2_bytes));
    return sig;
}

uintptr_t compute_graph_signature(
    const NF4QuantState& state,
    const int block_dim,
    const bool use_bf16,
    const NF4KernelVariant kernel_variant) {
    union FloatBits {
        float f;
        uint32_t u;
    };
    FloatBits offset_bits{};
    offset_bits.f = state.h_offset;

    uintptr_t sig = static_cast<uintptr_t>(state.num_elements);
    sig = mix_u64(sig, static_cast<uintptr_t>(state.num_blocks));
    sig = mix_u64(sig, static_cast<uintptr_t>(state.num_groups));
    sig = mix_u64(sig, static_cast<uintptr_t>(state.blocksize));
    sig = mix_u64(sig, static_cast<uintptr_t>(state.blocks_per_group));
    sig = mix_u64(sig, static_cast<uintptr_t>(block_dim));
    sig = mix_u64(sig, static_cast<uintptr_t>(use_bf16 ? 1 : 0));
    sig = mix_u64(sig, static_cast<uintptr_t>(static_cast<int>(kernel_variant)));
    sig = mix_u64(sig, static_cast<uintptr_t>(offset_bits.u));
    return sig;
}

void launch_nf4_kernel(
    const NF4QuantState& state,
    DeviceCache* cache,
    const bool use_bf16,
    const int threads,
    const size_t blocks,
    const cudaStream_t stream,
    const NF4KernelVariant kernel_variant) {
    const bool specialized_supported = state.blocksize == 64 || state.blocksize == 128;
    const bool use_specialized =
        (kernel_variant == NF4KernelVariant::kSpecialized && specialized_supported) ||
        (kernel_variant == NF4KernelVariant::kAuto && specialized_supported);
    const bool use_bpg256_fastpath = use_specialized && state.blocks_per_group == 256;

    if (use_bf16) {
        if (use_bpg256_fastpath && state.blocksize == 64) {
            nf4_dequant_kernel_fixed_blocksize<64, true, 256><<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
                cache->d_packed_weights,
                cache->d_absmax_q,
                cache->d_absmax2,
                cache->d_code2,
                state.h_offset,
                cache->d_output,
                static_cast<int64_t>(state.num_elements),
                state.blocks_per_group);
        } else if (use_bpg256_fastpath && state.blocksize == 128) {
            nf4_dequant_kernel_fixed_blocksize<128, true, 256><<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
                cache->d_packed_weights,
                cache->d_absmax_q,
                cache->d_absmax2,
                cache->d_code2,
                state.h_offset,
                cache->d_output,
                static_cast<int64_t>(state.num_elements),
                state.blocks_per_group);
        } else if (use_specialized && state.blocksize == 64) {
            nf4_dequant_kernel_fixed_blocksize<64, true><<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
                cache->d_packed_weights,
                cache->d_absmax_q,
                cache->d_absmax2,
                cache->d_code2,
                state.h_offset,
                cache->d_output,
                static_cast<int64_t>(state.num_elements),
                state.blocks_per_group);
        } else if (use_specialized && state.blocksize == 128) {
            nf4_dequant_kernel_fixed_blocksize<128, true><<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
                cache->d_packed_weights,
                cache->d_absmax_q,
                cache->d_absmax2,
                cache->d_code2,
                state.h_offset,
                cache->d_output,
                static_cast<int64_t>(state.num_elements),
                state.blocks_per_group);
        } else {
            nf4_dequant_kernel_impl<true><<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
                cache->d_packed_weights,
                cache->d_absmax_q,
                cache->d_absmax2,
                cache->d_code2,
                state.h_offset,
                cache->d_output,
                static_cast<int64_t>(state.num_elements),
                state.blocksize,
                state.blocks_per_group);
        }
    } else {
        if (use_bpg256_fastpath && state.blocksize == 64) {
            nf4_dequant_kernel_fixed_blocksize<64, false, 256><<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
                cache->d_packed_weights,
                cache->d_absmax_q,
                cache->d_absmax2,
                cache->d_code2,
                state.h_offset,
                cache->d_output,
                static_cast<int64_t>(state.num_elements),
                state.blocks_per_group);
        } else if (use_bpg256_fastpath && state.blocksize == 128) {
            nf4_dequant_kernel_fixed_blocksize<128, false, 256><<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
                cache->d_packed_weights,
                cache->d_absmax_q,
                cache->d_absmax2,
                cache->d_code2,
                state.h_offset,
                cache->d_output,
                static_cast<int64_t>(state.num_elements),
                state.blocks_per_group);
        } else if (use_specialized && state.blocksize == 64) {
            nf4_dequant_kernel_fixed_blocksize<64, false><<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
                cache->d_packed_weights,
                cache->d_absmax_q,
                cache->d_absmax2,
                cache->d_code2,
                state.h_offset,
                cache->d_output,
                static_cast<int64_t>(state.num_elements),
                state.blocks_per_group);
        } else if (use_specialized && state.blocksize == 128) {
            nf4_dequant_kernel_fixed_blocksize<128, false><<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
                cache->d_packed_weights,
                cache->d_absmax_q,
                cache->d_absmax2,
                cache->d_code2,
                state.h_offset,
                cache->d_output,
                static_cast<int64_t>(state.num_elements),
                state.blocks_per_group);
        } else {
            nf4_dequant_kernel_impl<false><<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
                cache->d_packed_weights,
                cache->d_absmax_q,
                cache->d_absmax2,
                cache->d_code2,
                state.h_offset,
                cache->d_output,
                static_cast<int64_t>(state.num_elements),
                state.blocksize,
                state.blocks_per_group);
        }
    }
}

bool ensure_kernel_graph(
    const NF4QuantState& state,
    DeviceCache* cache,
    const bool use_bf16,
    const int block_dim,
    const size_t blocks,
    const NF4KernelVariant kernel_variant,
    float* graph_build_time_ms) {
    if (cache == nullptr || cache->stream == nullptr) {
        return false;
    }
    if (graph_build_time_ms != nullptr) {
        *graph_build_time_ms = 0.0f;
    }
    const uintptr_t signature = compute_graph_signature(state, block_dim, use_bf16, kernel_variant);
    if (cache->graph_valid && cache->graph_signature == signature && cache->graph_exec != nullptr) {
        return true;
    }

    const auto build_start = std::chrono::steady_clock::now();
    invalidate_cached_graph(cache);

    cudaGraph_t graph = nullptr;
    cudaError_t err = cudaStreamBeginCapture(cache->stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "Warning: cudaStreamBeginCapture failed, CUDA Graph disabled: %s\n", cudaGetErrorString(err));
        if (graph_build_time_ms != nullptr) {
            const auto build_end = std::chrono::steady_clock::now();
            *graph_build_time_ms = std::chrono::duration<float, std::milli>(build_end - build_start).count();
        }
        return false;
    }

    launch_nf4_kernel(state, cache, use_bf16, block_dim, blocks, cache->stream, kernel_variant);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "Warning: CUDA Graph kernel capture launch failed: %s\n", cudaGetErrorString(err));
        cudaGraph_t abort_graph = nullptr;
        cudaStreamEndCapture(cache->stream, &abort_graph);
        if (abort_graph != nullptr) {
            cudaGraphDestroy(abort_graph);
        }
        if (graph_build_time_ms != nullptr) {
            const auto build_end = std::chrono::steady_clock::now();
            *graph_build_time_ms = std::chrono::duration<float, std::milli>(build_end - build_start).count();
        }
        return false;
    }

    err = cudaStreamEndCapture(cache->stream, &graph);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "Warning: cudaStreamEndCapture failed, CUDA Graph disabled: %s\n", cudaGetErrorString(err));
        if (graph != nullptr) {
            cudaGraphDestroy(graph);
        }
        if (graph_build_time_ms != nullptr) {
            const auto build_end = std::chrono::steady_clock::now();
            *graph_build_time_ms = std::chrono::duration<float, std::milli>(build_end - build_start).count();
        }
        return false;
    }

    err = cudaGraphInstantiate(&cache->graph_exec, graph, nullptr, nullptr, 0);
    cudaGraphDestroy(graph);
    if (err != cudaSuccess) {
        cache->graph_exec = nullptr;
        std::fprintf(stderr, "Warning: cudaGraphInstantiate failed, CUDA Graph disabled: %s\n", cudaGetErrorString(err));
        if (graph_build_time_ms != nullptr) {
            const auto build_end = std::chrono::steady_clock::now();
            *graph_build_time_ms = std::chrono::duration<float, std::milli>(build_end - build_start).count();
        }
        return false;
    }

    cache->graph_signature = signature;
    cache->graph_valid = true;
    if (graph_build_time_ms != nullptr) {
        const auto build_end = std::chrono::steady_clock::now();
        *graph_build_time_ms = std::chrono::duration<float, std::milli>(build_end - build_start).count();
    }
    return true;
}

bool upload_inputs_if_needed(
    const NF4QuantState& state,
    DeviceCache* cache,
    size_t packed_bytes,
    size_t absmax_q_bytes,
    size_t absmax2_bytes,
    size_t code2_bytes,
    bool force_upload,
    bool use_async_stream) {
    if (cache == nullptr) {
        return false;
    }
    const uintptr_t signature =
        compute_state_signature(state, packed_bytes, absmax_q_bytes, absmax2_bytes, code2_bytes);
    const bool metadata_changed =
        cache->state_signature != signature || cache->num_blocks != state.num_blocks ||
        cache->num_groups != state.num_groups;
    const bool need_upload = force_upload || !cache->inputs_uploaded || metadata_changed;
    if (!need_upload) {
        return true;
    }

    const cudaStream_t stream = use_async_stream ? cache->stream : static_cast<cudaStream_t>(0);
    if (use_async_stream) {
        if (!check_cuda(
                cudaMemcpyAsync(
                    cache->d_packed_weights,
                    state.h_packed_weights,
                    packed_bytes,
                    cudaMemcpyHostToDevice,
                    stream),
                "cudaMemcpyAsync(packed_weights)") ||
            !check_cuda(
                cudaMemcpyAsync(
                    cache->d_absmax_q, state.h_absmax_q, absmax_q_bytes, cudaMemcpyHostToDevice, stream),
                "cudaMemcpyAsync(absmax_q)") ||
            !check_cuda(
                cudaMemcpyAsync(
                    cache->d_absmax2, state.h_absmax2, absmax2_bytes, cudaMemcpyHostToDevice, stream),
                "cudaMemcpyAsync(absmax2)") ||
            !check_cuda(
                cudaMemcpyAsync(cache->d_code2, state.h_code2, code2_bytes, cudaMemcpyHostToDevice, stream),
                "cudaMemcpyAsync(code2)")) {
            cache->inputs_uploaded = false;
            return false;
        }
    } else if (!check_cuda(
                   cudaMemcpy(cache->d_packed_weights, state.h_packed_weights, packed_bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy(packed_weights)") ||
               !check_cuda(
                   cudaMemcpy(cache->d_absmax_q, state.h_absmax_q, absmax_q_bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy(absmax_q)") ||
               !check_cuda(
                   cudaMemcpy(cache->d_absmax2, state.h_absmax2, absmax2_bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy(absmax2)") ||
               !check_cuda(
                   cudaMemcpy(cache->d_code2, state.h_code2, code2_bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy(code2)")) {
        cache->inputs_uploaded = false;
        return false;
    }

    cache->inputs_uploaded = true;
    cache->state_signature = signature;
    cache->num_blocks = state.num_blocks;
    cache->num_groups = state.num_groups;
    return true;
}

bool run_kernel_once(
    const NF4QuantState& state,
    DeviceCache* cache,
    void* output,
    bool use_bf16,
    float* kernel_time_ms,
    int block_dim,
    bool copy_output_to_host,
    bool use_async_stream,
    bool use_cuda_graph,
    NF4KernelVariant kernel_variant,
    float* graph_build_time_ms) {
    if (cache == nullptr) {
        return false;
    }
    if (graph_build_time_ms != nullptr) {
        *graph_build_time_ms = 0.0f;
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    auto destroy_events = [&]() {
        if (start != nullptr) {
            cudaEventDestroy(start);
        }
        if (stop != nullptr) {
            cudaEventDestroy(stop);
        }
    };

    if (!check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)") ||
        !check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)")) {
        destroy_events();
        return false;
    }

    const int threads = (block_dim > 0 && block_dim <= 1024) ? block_dim : 256;
    const size_t elems_per_block = static_cast<size_t>(threads) * 2ULL;
    const size_t blocks = (state.num_elements + elems_per_block - 1ULL) / elems_per_block;
    const cudaStream_t stream = use_async_stream ? cache->stream : static_cast<cudaStream_t>(0);
    const bool graph_enabled = use_cuda_graph && use_async_stream && cache->stream != nullptr;
    const bool use_graph_exec =
        graph_enabled && ensure_kernel_graph(state, cache, use_bf16, threads, blocks, kernel_variant, graph_build_time_ms);

    if (!check_cuda(cudaEventRecord(start, stream), "cudaEventRecord(start)")) {
        destroy_events();
        return false;
    }

    if (use_graph_exec) {
        if (!check_cuda(cudaGraphLaunch(cache->graph_exec, stream), "cudaGraphLaunch")) {
            destroy_events();
            return false;
        }
    } else {
        launch_nf4_kernel(state, cache, use_bf16, threads, blocks, stream, kernel_variant);
        if (!check_cuda(cudaGetLastError(), "cudaGetLastError(kernel launch)")) {
            destroy_events();
            return false;
        }
    }

    if (!check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord(stop)") ||
        !check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)")) {
        destroy_events();
        return false;
    }

    float elapsed_ms = 0.0f;
    if (!check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime")) {
        destroy_events();
        return false;
    }

    if (copy_output_to_host) {
        const size_t output_bytes = state.num_elements * sizeof(uint16_t);
        if (use_async_stream) {
            if (!check_cuda(
                    cudaMemcpyAsync(output, cache->d_output, output_bytes, cudaMemcpyDeviceToHost, stream),
                    "cudaMemcpyAsync(output)") ||
                !check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(output copy)")) {
                destroy_events();
                return false;
            }
        } else if (!check_cuda(
                       cudaMemcpy(output, cache->d_output, output_bytes, cudaMemcpyDeviceToHost),
                       "cudaMemcpy(output)")) {
            destroy_events();
            return false;
        }
    }

    if (kernel_time_ms != nullptr) {
        *kernel_time_ms = elapsed_ms;
    }
    destroy_events();
    return true;
}

}  // namespace

void cuda_release_nf4_device_cache() {
    release_device_cache(&g_device_cache);
}

bool cuda_dequant_nf4(
    const NF4QuantState& state,
    void* output,
    bool use_bf16,
    float* kernel_time_ms,
    int block_dim,
    bool copy_output_to_host,
    bool reuse_device_buffers,
    bool use_cuda_graph,
    NF4KernelVariant kernel_variant,
    float* end_to_end_ms,
    float* graph_build_time_ms) {
    if (end_to_end_ms != nullptr) {
        *end_to_end_ms = 0.0f;
    }
    if (graph_build_time_ms != nullptr) {
        *graph_build_time_ms = 0.0f;
    }
    if (copy_output_to_host && output == nullptr) {
        std::fprintf(stderr, "[cuda_dequant_nf4] output pointer is null.\n");
        return false;
    }
    if (state.h_packed_weights == nullptr || state.h_absmax_q == nullptr ||
        state.h_absmax2 == nullptr || state.h_code2 == nullptr) {
        std::fprintf(stderr, "[cuda_dequant_nf4] quantized buffers are not initialized.\n");
        return false;
    }
    if (state.num_elements == 0 || state.blocksize <= 0) {
        std::fprintf(stderr, "[cuda_dequant_nf4] invalid tensor metadata.\n");
        return false;
    }

    const size_t packed_bytes = state.num_packed_bytes;
    const size_t absmax_q_bytes = state.num_blocks * sizeof(uint8_t);
    const size_t absmax2_bytes = state.num_groups * sizeof(__half);
    const size_t code2_bytes = kCode2Entries * sizeof(__half);
    const size_t output_bytes = state.num_elements * sizeof(uint16_t);
    const bool use_async_stream = reuse_device_buffers;
    const auto wall_start = std::chrono::steady_clock::now();
    auto finalize = [&](const bool ok) {
        if (end_to_end_ms != nullptr) {
            const auto wall_end = std::chrono::steady_clock::now();
            *end_to_end_ms = std::chrono::duration<float, std::milli>(wall_end - wall_start).count();
        }
        return ok;
    };

    if (reuse_device_buffers) {
        if (!ensure_cache_capacity(
                &g_device_cache,
                packed_bytes,
                absmax_q_bytes,
                absmax2_bytes,
                code2_bytes,
                output_bytes,
                use_async_stream)) {
            return finalize(false);
        }
        if (!upload_inputs_if_needed(
                state,
                &g_device_cache,
                packed_bytes,
                absmax_q_bytes,
                absmax2_bytes,
                code2_bytes,
                false,
                use_async_stream)) {
            return finalize(false);
        }
        const bool ok = run_kernel_once(
            state,
            &g_device_cache,
            output,
            use_bf16,
            kernel_time_ms,
            block_dim,
            copy_output_to_host,
            use_async_stream,
            use_cuda_graph,
            kernel_variant,
            graph_build_time_ms);
        return finalize(ok);
    }

    DeviceCache local_cache{};
    if (!ensure_cache_capacity(
            &local_cache, packed_bytes, absmax_q_bytes, absmax2_bytes, code2_bytes, output_bytes, false)) {
        release_device_cache(&local_cache);
        return finalize(false);
    }
    if (!upload_inputs_if_needed(
            state, &local_cache, packed_bytes, absmax_q_bytes, absmax2_bytes, code2_bytes, true, false)) {
        release_device_cache(&local_cache);
        return finalize(false);
    }

    const bool ok = run_kernel_once(
        state,
        &local_cache,
        output,
        use_bf16,
        kernel_time_ms,
        block_dim,
        copy_output_to_host,
        false,
        false,
        kernel_variant,
        graph_build_time_ms);
    release_device_cache(&local_cache);
    return finalize(ok);
}
