#include "quantize.cuh"

namespace hadamard {
namespace {

constexpr int MAX_WARPS_PER_BLOCK = 8;

__device__ inline float warp_reduce_max(float value, unsigned mask) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(mask, value, offset));
    }
    return value;
}

__device__ inline float block_reduce_max(float value, float* warp_max_buffer, float* block_max_buffer) {
    const unsigned warp_mask = __activemask();
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int warp_count = (blockDim.x + 31) >> 5;

    value = warp_reduce_max(value, warp_mask);
    if (lane == 0) {
        warp_max_buffer[warp_id] = value;
    }
    __syncthreads();

    if (warp_id == 0) {
        const unsigned leader_mask = __ballot_sync(0xffffffffu, lane < warp_count);
        const float warp_value = lane < warp_count ? warp_max_buffer[lane] : 0.0f;
        const float block_value = warp_reduce_max(warp_value, leader_mask);
        if (lane == 0) {
            *block_max_buffer = block_value;
        }
    }
    __syncthreads();
    return *block_max_buffer;
}

__device__ inline void load_half2_to_shared(const half* input, int row, int tid, int head_dim, float* row_values) {
    const half2 packed = reinterpret_cast<const half2*>(input + static_cast<size_t>(row) * head_dim)[tid];
    const int base = tid << 1;
    row_values[base] = __half2float(__low2half(packed));
    row_values[base + 1] = __half2float(__high2half(packed));
}

__device__ inline void load_bfloat162_to_shared(const __nv_bfloat16* input,
                                                int row,
                                                int tid,
                                                int head_dim,
                                                float* row_values) {
    const __nv_bfloat162 packed =
        reinterpret_cast<const __nv_bfloat162*>(input + static_cast<size_t>(row) * head_dim)[tid];
    const float2 unpacked = __bfloat1622float2(packed);
    const int base = tid << 1;
    row_values[base] = unpacked.x;
    row_values[base + 1] = unpacked.y;
}

template <typename T>
__global__ void quantize_int4_kernel(const T* input,
                                     uint8_t* output,
                                     float* scales,
                                     int total_rows,
                                     int head_dim) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= total_rows) {
        return;
    }

    __shared__ float warp_max_buffer[MAX_WARPS_PER_BLOCK];
    __shared__ float block_max_buffer;
    extern __shared__ float shared_values[];
    float* row_values = shared_values;

    float local_absmax = 0.0f;
    if (tid < head_dim) {
        const float value = scalar_to_float(input[static_cast<size_t>(row) * head_dim + tid]);
        row_values[tid] = value;
        local_absmax = fabsf(value);
    }
    __syncthreads();
    const float absmax = block_reduce_max(local_absmax, warp_max_buffer, &block_max_buffer);
    const float scale = absmax > 0.0f ? absmax / 7.0f : 1.0f;
    if (tid == 0) {
        scales[row] = scale;
    }

    if (tid < head_dim / 2) {
        const int base = tid << 1;
        const int8_t low = quantize_int4_scalar(row_values[base], scale);
        const int8_t high = quantize_int4_scalar(row_values[base + 1], scale);
        output[static_cast<size_t>(row) * (head_dim / 2) + tid] = pack_int4_values(low, high);
    }
}

__global__ void quantize_int4_kernel_half2(const half* input,
                                           uint8_t* output,
                                           float* scales,
                                           int total_rows,
                                           int head_dim) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int pair_count = head_dim >> 1;
    if (row >= total_rows) {
        return;
    }

    __shared__ float warp_max_buffer[MAX_WARPS_PER_BLOCK];
    __shared__ float block_max_buffer;
    extern __shared__ float shared_values[];
    float* row_values = shared_values;

    float local_absmax = 0.0f;
    if (tid < pair_count) {
        load_half2_to_shared(input, row, tid, head_dim, row_values);
        const int base = tid << 1;
        local_absmax = fmaxf(fabsf(row_values[base]), fabsf(row_values[base + 1]));
    }
    __syncthreads();
    const float absmax = block_reduce_max(local_absmax, warp_max_buffer, &block_max_buffer);
    const float scale = absmax > 0.0f ? absmax / 7.0f : 1.0f;
    if (tid == 0) {
        scales[row] = scale;
    }

    if (tid < pair_count) {
        const int base = tid << 1;
        const int8_t low = quantize_int4_scalar(row_values[base], scale);
        const int8_t high = quantize_int4_scalar(row_values[base + 1], scale);
        output[static_cast<size_t>(row) * pair_count + tid] = pack_int4_values(low, high);
    }
}

template <typename T>
__global__ void quantize_fp8_kernel(const T* input,
                                    uint8_t* output,
                                    float* scales,
                                    int total_rows,
                                    int head_dim) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= total_rows) {
        return;
    }

    __shared__ float warp_max_buffer[MAX_WARPS_PER_BLOCK];
    __shared__ float block_max_buffer;
    extern __shared__ float shared_values[];
    float* row_values = shared_values;

    float local_absmax = 0.0f;
    if (tid < head_dim) {
        const float value = scalar_to_float(input[static_cast<size_t>(row) * head_dim + tid]);
        row_values[tid] = value;
        local_absmax = fabsf(value);
    }
    __syncthreads();
    const float absmax = block_reduce_max(local_absmax, warp_max_buffer, &block_max_buffer);
    const float scale = absmax > 0.0f ? absmax / FP8_E4M3_MAX : 1.0f;
    if (tid == 0) {
        scales[row] = scale;
    }

    if (tid < head_dim) {
        output[static_cast<size_t>(row) * head_dim + tid] =
            float_to_fp8_e4m3(row_values[tid] / scale);
    }
}

__global__ void quantize_fp8_kernel_half2(const half* input,
                                          uint8_t* output,
                                          float* scales,
                                          int total_rows,
                                          int head_dim) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int pair_count = head_dim >> 1;
    if (row >= total_rows) {
        return;
    }

    __shared__ float warp_max_buffer[MAX_WARPS_PER_BLOCK];
    __shared__ float block_max_buffer;
    extern __shared__ float shared_values[];
    float* row_values = shared_values;

    float local_absmax = 0.0f;
    if (tid < pair_count) {
        load_half2_to_shared(input, row, tid, head_dim, row_values);
        const int base = tid << 1;
        local_absmax = fmaxf(fabsf(row_values[base]), fabsf(row_values[base + 1]));
    }
    __syncthreads();
    const float absmax = block_reduce_max(local_absmax, warp_max_buffer, &block_max_buffer);
    const float scale = absmax > 0.0f ? absmax / FP8_E4M3_MAX : 1.0f;
    if (tid == 0) {
        scales[row] = scale;
    }

    if (tid < pair_count) {
        const int base = tid << 1;
        uint8_t* out = output + static_cast<size_t>(row) * head_dim + base;
        out[0] = float_to_fp8_e4m3(row_values[base] / scale);
        out[1] = float_to_fp8_e4m3(row_values[base + 1] / scale);
    }
}

__global__ void quantize_int4_kernel_bfloat162(const __nv_bfloat16* input,
                                               uint8_t* output,
                                               float* scales,
                                               int total_rows,
                                               int head_dim) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int pair_count = head_dim >> 1;
    if (row >= total_rows) {
        return;
    }

    __shared__ float warp_max_buffer[MAX_WARPS_PER_BLOCK];
    __shared__ float block_max_buffer;
    extern __shared__ float shared_values[];
    float* row_values = shared_values;

    float local_absmax = 0.0f;
    if (tid < pair_count) {
        load_bfloat162_to_shared(input, row, tid, head_dim, row_values);
        const int base = tid << 1;
        local_absmax = fmaxf(fabsf(row_values[base]), fabsf(row_values[base + 1]));
    }
    __syncthreads();
    const float absmax = block_reduce_max(local_absmax, warp_max_buffer, &block_max_buffer);
    const float scale = absmax > 0.0f ? absmax / 7.0f : 1.0f;
    if (tid == 0) {
        scales[row] = scale;
    }

    if (tid < pair_count) {
        const int base = tid << 1;
        const int8_t low = quantize_int4_scalar(row_values[base], scale);
        const int8_t high = quantize_int4_scalar(row_values[base + 1], scale);
        output[static_cast<size_t>(row) * pair_count + tid] = pack_int4_values(low, high);
    }
}

__global__ void quantize_fp8_kernel_bfloat162(const __nv_bfloat16* input,
                                              uint8_t* output,
                                              float* scales,
                                              int total_rows,
                                              int head_dim) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int pair_count = head_dim >> 1;
    if (row >= total_rows) {
        return;
    }

    __shared__ float warp_max_buffer[MAX_WARPS_PER_BLOCK];
    __shared__ float block_max_buffer;
    extern __shared__ float shared_values[];
    float* row_values = shared_values;

    float local_absmax = 0.0f;
    if (tid < pair_count) {
        load_bfloat162_to_shared(input, row, tid, head_dim, row_values);
        const int base = tid << 1;
        local_absmax = fmaxf(fabsf(row_values[base]), fabsf(row_values[base + 1]));
    }
    __syncthreads();
    const float absmax = block_reduce_max(local_absmax, warp_max_buffer, &block_max_buffer);
    const float scale = absmax > 0.0f ? absmax / FP8_E4M3_MAX : 1.0f;
    if (tid == 0) {
        scales[row] = scale;
    }

    if (tid < pair_count) {
        const int base = tid << 1;
        uint8_t* out = output + static_cast<size_t>(row) * head_dim + base;
        out[0] = float_to_fp8_e4m3(row_values[base] / scale);
        out[1] = float_to_fp8_e4m3(row_values[base + 1] / scale);
    }
}

template <typename T>
__global__ void fused_hadamard_quantize_int4_kernel(const T* input,
                                                    uint8_t* output,
                                                    float* scales,
                                                    int total_rows,
                                                    int head_dim) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= total_rows) {
        return;
    }

    __shared__ float warp_max_buffer[MAX_WARPS_PER_BLOCK];
    __shared__ float block_max_buffer;
    extern __shared__ float shared_values[];
    float* row_values = shared_values;

    if (tid < head_dim) {
        row_values[tid] = scalar_to_float(input[static_cast<size_t>(row) * head_dim + tid]);
    }
    __syncthreads();

    const int pair_count = head_dim >> 1;
    for (int len = 1; len < head_dim; len <<= 1) {
        if (tid < pair_count) {
            const int first_index = butterfly_index(tid, len);
            const float u = row_values[first_index];
            const float v = row_values[first_index + len];
            row_values[first_index] = u + v;
            row_values[first_index + len] = u - v;
        }
        __syncthreads();
    }

    float local_absmax = 0.0f;
    if (tid < head_dim) {
        const T rounded_value = float_to_scalar<T>(row_values[tid] * rsqrtf(static_cast<float>(head_dim)));
        row_values[tid] = scalar_to_float(rounded_value);
        local_absmax = fabsf(row_values[tid]);
    }
    const float absmax = block_reduce_max(local_absmax, warp_max_buffer, &block_max_buffer);
    const float scale = absmax > 0.0f ? absmax / 7.0f : 1.0f;
    if (tid == 0) {
        scales[row] = scale;
    }

    if (tid < head_dim / 2) {
        const int base = tid << 1;
        const int8_t low = quantize_int4_scalar(row_values[base], scale);
        const int8_t high = quantize_int4_scalar(row_values[base + 1], scale);
        output[static_cast<size_t>(row) * (head_dim / 2) + tid] = pack_int4_values(low, high);
    }
}

__global__ void fused_hadamard_quantize_int4_kernel_half2(const half* input,
                                                          uint8_t* output,
                                                          float* scales,
                                                          int total_rows,
                                                          int head_dim) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int pair_count = head_dim >> 1;
    if (row >= total_rows) {
        return;
    }

    __shared__ float warp_max_buffer[MAX_WARPS_PER_BLOCK];
    __shared__ float block_max_buffer;
    extern __shared__ float shared_values[];
    float* row_values = shared_values;

    if (tid < pair_count) {
        load_half2_to_shared(input, row, tid, head_dim, row_values);
    }
    __syncthreads();

    for (int len = 1; len < head_dim; len <<= 1) {
        if (tid < pair_count) {
            const int first_index = butterfly_index(tid, len);
            const float u = row_values[first_index];
            const float v = row_values[first_index + len];
            row_values[first_index] = u + v;
            row_values[first_index + len] = u - v;
        }
        __syncthreads();
    }

    float local_absmax = 0.0f;
    if (tid < pair_count) {
        const int base = tid << 1;
        const half2 rounded =
            __floats2half2_rn(row_values[base] * rsqrtf(static_cast<float>(head_dim)),
                              row_values[base + 1] * rsqrtf(static_cast<float>(head_dim)));
        row_values[base] = __half2float(__low2half(rounded));
        row_values[base + 1] = __half2float(__high2half(rounded));
        local_absmax = fmaxf(fabsf(row_values[base]), fabsf(row_values[base + 1]));
    }
    const float absmax = block_reduce_max(local_absmax, warp_max_buffer, &block_max_buffer);
    const float scale = absmax > 0.0f ? absmax / 7.0f : 1.0f;
    if (tid == 0) {
        scales[row] = scale;
    }

    if (tid < pair_count) {
        const int base = tid << 1;
        const int8_t low = quantize_int4_scalar(row_values[base], scale);
        const int8_t high = quantize_int4_scalar(row_values[base + 1], scale);
        output[static_cast<size_t>(row) * pair_count + tid] = pack_int4_values(low, high);
    }
}

template <typename T>
__global__ void fused_hadamard_quantize_fp8_kernel(const T* input,
                                                   uint8_t* output,
                                                   float* scales,
                                                   int total_rows,
                                                   int head_dim) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= total_rows) {
        return;
    }

    __shared__ float warp_max_buffer[MAX_WARPS_PER_BLOCK];
    __shared__ float block_max_buffer;
    extern __shared__ float shared_values[];
    float* row_values = shared_values;

    if (tid < head_dim) {
        row_values[tid] = scalar_to_float(input[static_cast<size_t>(row) * head_dim + tid]);
    }
    __syncthreads();

    const int pair_count = head_dim >> 1;
    for (int len = 1; len < head_dim; len <<= 1) {
        if (tid < pair_count) {
            const int first_index = butterfly_index(tid, len);
            const float u = row_values[first_index];
            const float v = row_values[first_index + len];
            row_values[first_index] = u + v;
            row_values[first_index + len] = u - v;
        }
        __syncthreads();
    }

    float local_absmax = 0.0f;
    if (tid < head_dim) {
        const T rounded_value = float_to_scalar<T>(row_values[tid] * rsqrtf(static_cast<float>(head_dim)));
        row_values[tid] = scalar_to_float(rounded_value);
        local_absmax = fabsf(row_values[tid]);
    }
    const float absmax = block_reduce_max(local_absmax, warp_max_buffer, &block_max_buffer);
    const float scale = absmax > 0.0f ? absmax / FP8_E4M3_MAX : 1.0f;
    if (tid == 0) {
        scales[row] = scale;
    }

    if (tid < head_dim) {
        output[static_cast<size_t>(row) * head_dim + tid] =
            float_to_fp8_e4m3(row_values[tid] / scale);
    }
}

__global__ void fused_hadamard_quantize_fp8_kernel_half2(const half* input,
                                                         uint8_t* output,
                                                         float* scales,
                                                         int total_rows,
                                                         int head_dim) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int pair_count = head_dim >> 1;
    if (row >= total_rows) {
        return;
    }

    __shared__ float warp_max_buffer[MAX_WARPS_PER_BLOCK];
    __shared__ float block_max_buffer;
    extern __shared__ float shared_values[];
    float* row_values = shared_values;

    if (tid < pair_count) {
        load_half2_to_shared(input, row, tid, head_dim, row_values);
    }
    __syncthreads();

    for (int len = 1; len < head_dim; len <<= 1) {
        if (tid < pair_count) {
            const int first_index = butterfly_index(tid, len);
            const float u = row_values[first_index];
            const float v = row_values[first_index + len];
            row_values[first_index] = u + v;
            row_values[first_index + len] = u - v;
        }
        __syncthreads();
    }

    float local_absmax = 0.0f;
    if (tid < pair_count) {
        const int base = tid << 1;
        const half2 rounded =
            __floats2half2_rn(row_values[base] * rsqrtf(static_cast<float>(head_dim)),
                              row_values[base + 1] * rsqrtf(static_cast<float>(head_dim)));
        row_values[base] = __half2float(__low2half(rounded));
        row_values[base + 1] = __half2float(__high2half(rounded));
        local_absmax = fmaxf(fabsf(row_values[base]), fabsf(row_values[base + 1]));
    }
    const float absmax = block_reduce_max(local_absmax, warp_max_buffer, &block_max_buffer);
    const float scale = absmax > 0.0f ? absmax / FP8_E4M3_MAX : 1.0f;
    if (tid == 0) {
        scales[row] = scale;
    }

    if (tid < pair_count) {
        const int base = tid << 1;
        uint8_t* out = output + static_cast<size_t>(row) * head_dim + base;
        out[0] = float_to_fp8_e4m3(row_values[base] / scale);
        out[1] = float_to_fp8_e4m3(row_values[base + 1] / scale);
    }
}

__global__ void fused_hadamard_quantize_int4_kernel_bfloat162(const __nv_bfloat16* input,
                                                              uint8_t* output,
                                                              float* scales,
                                                              int total_rows,
                                                              int head_dim) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int pair_count = head_dim >> 1;
    if (row >= total_rows) {
        return;
    }

    __shared__ float warp_max_buffer[MAX_WARPS_PER_BLOCK];
    __shared__ float block_max_buffer;
    extern __shared__ float shared_values[];
    float* row_values = shared_values;

    if (tid < pair_count) {
        load_bfloat162_to_shared(input, row, tid, head_dim, row_values);
    }
    __syncthreads();

    for (int len = 1; len < head_dim; len <<= 1) {
        if (tid < pair_count) {
            const int first_index = butterfly_index(tid, len);
            const float u = row_values[first_index];
            const float v = row_values[first_index + len];
            row_values[first_index] = u + v;
            row_values[first_index + len] = u - v;
        }
        __syncthreads();
    }

    float local_absmax = 0.0f;
    if (tid < pair_count) {
        const int base = tid << 1;
        const __nv_bfloat162 rounded =
            __floats2bfloat162_rn(row_values[base] * rsqrtf(static_cast<float>(head_dim)),
                                  row_values[base + 1] * rsqrtf(static_cast<float>(head_dim)));
        const float2 rounded_f = __bfloat1622float2(rounded);
        row_values[base] = rounded_f.x;
        row_values[base + 1] = rounded_f.y;
        local_absmax = fmaxf(fabsf(row_values[base]), fabsf(row_values[base + 1]));
    }
    const float absmax = block_reduce_max(local_absmax, warp_max_buffer, &block_max_buffer);
    const float scale = absmax > 0.0f ? absmax / 7.0f : 1.0f;
    if (tid == 0) {
        scales[row] = scale;
    }

    if (tid < pair_count) {
        const int base = tid << 1;
        const int8_t low = quantize_int4_scalar(row_values[base], scale);
        const int8_t high = quantize_int4_scalar(row_values[base + 1], scale);
        output[static_cast<size_t>(row) * pair_count + tid] = pack_int4_values(low, high);
    }
}

__global__ void fused_hadamard_quantize_fp8_kernel_bfloat162(const __nv_bfloat16* input,
                                                             uint8_t* output,
                                                             float* scales,
                                                             int total_rows,
                                                             int head_dim) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int pair_count = head_dim >> 1;
    if (row >= total_rows) {
        return;
    }

    __shared__ float warp_max_buffer[MAX_WARPS_PER_BLOCK];
    __shared__ float block_max_buffer;
    extern __shared__ float shared_values[];
    float* row_values = shared_values;

    if (tid < pair_count) {
        load_bfloat162_to_shared(input, row, tid, head_dim, row_values);
    }
    __syncthreads();

    for (int len = 1; len < head_dim; len <<= 1) {
        if (tid < pair_count) {
            const int first_index = butterfly_index(tid, len);
            const float u = row_values[first_index];
            const float v = row_values[first_index + len];
            row_values[first_index] = u + v;
            row_values[first_index + len] = u - v;
        }
        __syncthreads();
    }

    float local_absmax = 0.0f;
    if (tid < pair_count) {
        const int base = tid << 1;
        const __nv_bfloat162 rounded =
            __floats2bfloat162_rn(row_values[base] * rsqrtf(static_cast<float>(head_dim)),
                                  row_values[base + 1] * rsqrtf(static_cast<float>(head_dim)));
        const float2 rounded_f = __bfloat1622float2(rounded);
        row_values[base] = rounded_f.x;
        row_values[base + 1] = rounded_f.y;
        local_absmax = fmaxf(fabsf(row_values[base]), fabsf(row_values[base + 1]));
    }
    const float absmax = block_reduce_max(local_absmax, warp_max_buffer, &block_max_buffer);
    const float scale = absmax > 0.0f ? absmax / FP8_E4M3_MAX : 1.0f;
    if (tid == 0) {
        scales[row] = scale;
    }

    if (tid < pair_count) {
        const int base = tid << 1;
        uint8_t* out = output + static_cast<size_t>(row) * head_dim + base;
        out[0] = float_to_fp8_e4m3(row_values[base] / scale);
        out[1] = float_to_fp8_e4m3(row_values[base + 1] / scale);
    }
}

template <typename T>
void launch_rowwise_kernel(const T* d_input,
                           uint8_t* d_output,
                           float* d_scales,
                           int total_rows,
                           int head_dim,
                           cudaStream_t stream,
                           bool fused_int4,
                           bool fused_fp8) {
    if (!is_power_of_two(head_dim) || head_dim > 256) {
        throw std::invalid_argument("Quantization kernels expect head_dim to be a power of two in [1, 256]");
    }

    const dim3 grid(total_rows);
    const dim3 block(head_dim);
    const size_t shared_bytes = static_cast<size_t>(head_dim) * sizeof(float);

    if (fused_int4) {
        fused_hadamard_quantize_int4_kernel<<<grid, block, shared_bytes, stream>>>(
            d_input, d_output, d_scales, total_rows, head_dim);
    } else if (fused_fp8) {
        fused_hadamard_quantize_fp8_kernel<<<grid, block, shared_bytes, stream>>>(
            d_input, d_output, d_scales, total_rows, head_dim);
    } else {
        quantize_fp8_kernel<<<grid, block, shared_bytes, stream>>>(
            d_input, d_output, d_scales, total_rows, head_dim);
    }

    CUDA_CHECK(cudaGetLastError());
}

template <>
void launch_rowwise_kernel<half>(const half* d_input,
                                 uint8_t* d_output,
                                 float* d_scales,
                                 int total_rows,
                                 int head_dim,
                                 cudaStream_t stream,
                                 bool fused_int4,
                                 bool fused_fp8) {
    if (!is_power_of_two(head_dim) || head_dim > 256) {
        throw std::invalid_argument("Quantization kernels expect head_dim to be a power of two in [1, 256]");
    }

    const dim3 grid(total_rows);
    const dim3 block(head_dim >> 1);
    const size_t shared_bytes = static_cast<size_t>(head_dim) * sizeof(float);

    if (fused_int4) {
        fused_hadamard_quantize_int4_kernel_half2<<<grid, block, shared_bytes, stream>>>(
            d_input, d_output, d_scales, total_rows, head_dim);
    } else if (fused_fp8) {
        fused_hadamard_quantize_fp8_kernel_half2<<<grid, block, shared_bytes, stream>>>(
            d_input, d_output, d_scales, total_rows, head_dim);
    } else {
        quantize_fp8_kernel_half2<<<grid, block, shared_bytes, stream>>>(
            d_input, d_output, d_scales, total_rows, head_dim);
    }

    CUDA_CHECK(cudaGetLastError());
}

template <>
void launch_rowwise_kernel<__nv_bfloat16>(const __nv_bfloat16* d_input,
                                          uint8_t* d_output,
                                          float* d_scales,
                                          int total_rows,
                                          int head_dim,
                                          cudaStream_t stream,
                                          bool fused_int4,
                                          bool fused_fp8) {
    if (!is_power_of_two(head_dim) || head_dim > 256) {
        throw std::invalid_argument("Quantization kernels expect head_dim to be a power of two in [1, 256]");
    }

    const dim3 grid(total_rows);
    const dim3 block(head_dim >> 1);
    const size_t shared_bytes = static_cast<size_t>(head_dim) * sizeof(float);

    if (fused_int4) {
        fused_hadamard_quantize_int4_kernel_bfloat162<<<grid, block, shared_bytes, stream>>>(
            d_input, d_output, d_scales, total_rows, head_dim);
    } else if (fused_fp8) {
        fused_hadamard_quantize_fp8_kernel_bfloat162<<<grid, block, shared_bytes, stream>>>(
            d_input, d_output, d_scales, total_rows, head_dim);
    } else {
        quantize_fp8_kernel_bfloat162<<<grid, block, shared_bytes, stream>>>(
            d_input, d_output, d_scales, total_rows, head_dim);
    }

    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

template <typename T>
void quantize_int4_gpu(const T* d_input,
                       uint8_t* d_output,
                       float* d_scales,
                       int total_rows,
                       int head_dim,
                       cudaStream_t stream) {
    if (!is_power_of_two(head_dim) || head_dim > 256) {
        throw std::invalid_argument("INT4 kernel expects head_dim to be a power of two in [1, 256]");
    }

    const dim3 grid(total_rows);
    const dim3 block(head_dim);
    const size_t shared_bytes = static_cast<size_t>(head_dim) * sizeof(float);
    quantize_int4_kernel<<<grid, block, shared_bytes, stream>>>(d_input, d_output, d_scales, total_rows, head_dim);
    CUDA_CHECK(cudaGetLastError());
}

template <>
void quantize_int4_gpu<half>(const half* d_input,
                             uint8_t* d_output,
                             float* d_scales,
                             int total_rows,
                             int head_dim,
                             cudaStream_t stream) {
    if (!is_power_of_two(head_dim) || head_dim > 256) {
        throw std::invalid_argument("INT4 kernel expects head_dim to be a power of two in [1, 256]");
    }

    const dim3 grid(total_rows);
    const dim3 block(head_dim >> 1);
    const size_t shared_bytes = static_cast<size_t>(head_dim) * sizeof(float);
    quantize_int4_kernel_half2<<<grid, block, shared_bytes, stream>>>(d_input, d_output, d_scales, total_rows, head_dim);
    CUDA_CHECK(cudaGetLastError());
}

template <>
void quantize_int4_gpu<__nv_bfloat16>(const __nv_bfloat16* d_input,
                                      uint8_t* d_output,
                                      float* d_scales,
                                      int total_rows,
                                      int head_dim,
                                      cudaStream_t stream) {
    if (!is_power_of_two(head_dim) || head_dim > 256) {
        throw std::invalid_argument("INT4 kernel expects head_dim to be a power of two in [1, 256]");
    }

    const dim3 grid(total_rows);
    const dim3 block(head_dim >> 1);
    const size_t shared_bytes = static_cast<size_t>(head_dim) * sizeof(float);
    quantize_int4_kernel_bfloat162<<<grid, block, shared_bytes, stream>>>(d_input, d_output, d_scales, total_rows, head_dim);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void hadamard_quantize_int4_gpu(const T* d_input,
                                uint8_t* d_output,
                                float* d_scales,
                                int total_rows,
                                int head_dim,
                                cudaStream_t stream) {
    launch_rowwise_kernel(d_input, d_output, d_scales, total_rows, head_dim, stream, true, false);
}

template <typename T>
void quantize_fp8_gpu(const T* d_input,
                      uint8_t* d_output,
                      float* d_scales,
                      int total_rows,
                      int head_dim,
                      cudaStream_t stream) {
    launch_rowwise_kernel(d_input, d_output, d_scales, total_rows, head_dim, stream, false, false);
}

template <typename T>
void hadamard_quantize_fp8_gpu(const T* d_input,
                               uint8_t* d_output,
                               float* d_scales,
                               int total_rows,
                               int head_dim,
                               cudaStream_t stream) {
    launch_rowwise_kernel(d_input, d_output, d_scales, total_rows, head_dim, stream, false, true);
}

template void quantize_int4_gpu<half>(const half*, uint8_t*, float*, int, int, cudaStream_t);
template void quantize_int4_gpu<__nv_bfloat16>(const __nv_bfloat16*, uint8_t*, float*, int, int, cudaStream_t);
template void hadamard_quantize_int4_gpu<half>(const half*, uint8_t*, float*, int, int, cudaStream_t);
template void hadamard_quantize_int4_gpu<__nv_bfloat16>(const __nv_bfloat16*, uint8_t*, float*, int, int, cudaStream_t);
template void quantize_fp8_gpu<half>(const half*, uint8_t*, float*, int, int, cudaStream_t);
template void quantize_fp8_gpu<__nv_bfloat16>(const __nv_bfloat16*, uint8_t*, float*, int, int, cudaStream_t);
template void hadamard_quantize_fp8_gpu<half>(const half*, uint8_t*, float*, int, int, cudaStream_t);
template void hadamard_quantize_fp8_gpu<__nv_bfloat16>(const __nv_bfloat16*, uint8_t*, float*, int, int, cudaStream_t);

}  // namespace hadamard
