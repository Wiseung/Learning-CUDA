#include "hadamard.cuh"

#include <mma.h>

#include <vector>

namespace hadamard {
namespace {

using namespace nvcuda;

inline int host_popcount(unsigned int value) {
    int count = 0;
    while (value != 0U) {
        value &= (value - 1U);
        ++count;
    }
    return count;
}

__device__ inline void warp_butterfly_pair_stage(float& low, float& high, int offset, int tid) {
    const float peer_low = __shfl_xor_sync(0xffffffffu, low, offset);
    const float peer_high = __shfl_xor_sync(0xffffffffu, high, offset);
    if ((tid & offset) == 0) {
        low += peer_low;
        high += peer_high;
    } else {
        low = peer_low - low;
        high = peer_high - high;
    }
}

template <typename T>
__global__ void hadamard_transform_kernel_v1(T* data, int head_dim, int total_rows) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row >= total_rows) {
        return;
    }

    extern __shared__ float shared_buffer[];
    float* row_values = shared_buffer;

    if (tid < head_dim) {
        row_values[tid] = scalar_to_float(data[static_cast<size_t>(row) * head_dim + tid]);
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

    const float inv_sqrt_head_dim = rsqrtf(static_cast<float>(head_dim));
    if (tid < head_dim) {
        data[static_cast<size_t>(row) * head_dim + tid] =
            float_to_scalar<T>(row_values[tid] * inv_sqrt_head_dim);
    }
}

// The FP16 specialization keeps the same butterfly schedule as v1, but uses
// half2 for the global-memory load/store path so each thread moves two values.
__global__ void hadamard_transform_kernel_v1_half2(half* data, int head_dim, int total_rows) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int pair_count = head_dim >> 1;

    if (row >= total_rows) {
        return;
    }

    extern __shared__ float shared_buffer[];
    float* row_values = shared_buffer;

    if (tid < pair_count) {
        const half2 packed = reinterpret_cast<const half2*>(data + static_cast<size_t>(row) * head_dim)[tid];
        float low = __half2float(__low2half(packed));
        float high = __half2float(__high2half(packed));

        // Stage len=1 is fully local to the two values loaded by one thread.
        const float local_low = low + high;
        const float local_high = low - high;
        low = local_low;
        high = local_high;

        // Stages len=2..32 remain within a single warp when each thread owns 2 values.
        for (int offset = 1; offset < min(pair_count, 32); offset <<= 1) {
            warp_butterfly_pair_stage(low, high, offset, tid);
        }

        const int base = tid << 1;
        if (pair_count > 32) {
            row_values[base] = low;
            row_values[base + 1] = high;
        } else {
            const float inv_sqrt_head_dim = rsqrtf(static_cast<float>(head_dim));
            reinterpret_cast<half2*>(data + static_cast<size_t>(row) * head_dim)[tid] =
                __floats2half2_rn(low * inv_sqrt_head_dim, high * inv_sqrt_head_dim);
            return;
        }
    }
    __syncthreads();

    for (int len = 64; len < head_dim; len <<= 1) {
        if (tid < pair_count) {
            const int first_index = butterfly_index(tid, len);
            const float u = row_values[first_index];
            const float v = row_values[first_index + len];
            row_values[first_index] = u + v;
            row_values[first_index + len] = u - v;
        }
        __syncthreads();
    }

    const float inv_sqrt_head_dim = rsqrtf(static_cast<float>(head_dim));
    if (tid < pair_count) {
        const int base = tid << 1;
        reinterpret_cast<half2*>(data + static_cast<size_t>(row) * head_dim)[tid] =
            __floats2half2_rn(row_values[base] * inv_sqrt_head_dim,
                              row_values[base + 1] * inv_sqrt_head_dim);
    }
}

// BF16 vectorized v1 path. It mirrors the FP16 half2 strategy but uses
// bfloat162 for the global-memory load/store path.
__global__ void hadamard_transform_kernel_v1_bfloat162(__nv_bfloat16* data, int head_dim, int total_rows) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int pair_count = head_dim >> 1;

    if (row >= total_rows) {
        return;
    }

    extern __shared__ float shared_buffer[];
    float* row_values = shared_buffer;

    if (tid < pair_count) {
        const __nv_bfloat162 packed =
            reinterpret_cast<const __nv_bfloat162*>(data + static_cast<size_t>(row) * head_dim)[tid];
        const float2 unpacked = __bfloat1622float2(packed);
        float low = unpacked.x;
        float high = unpacked.y;

        const float local_low = low + high;
        const float local_high = low - high;
        low = local_low;
        high = local_high;

        for (int offset = 1; offset < min(pair_count, 32); offset <<= 1) {
            warp_butterfly_pair_stage(low, high, offset, tid);
        }

        const int base = tid << 1;
        if (pair_count > 32) {
            row_values[base] = low;
            row_values[base + 1] = high;
        } else {
            const float inv_sqrt_head_dim = rsqrtf(static_cast<float>(head_dim));
            reinterpret_cast<__nv_bfloat162*>(data + static_cast<size_t>(row) * head_dim)[tid] =
                __floats2bfloat162_rn(low * inv_sqrt_head_dim, high * inv_sqrt_head_dim);
            return;
        }
    }
    __syncthreads();

    for (int len = 64; len < head_dim; len <<= 1) {
        if (tid < pair_count) {
            const int first_index = butterfly_index(tid, len);
            const float u = row_values[first_index];
            const float v = row_values[first_index + len];
            row_values[first_index] = u + v;
            row_values[first_index + len] = u - v;
        }
        __syncthreads();
    }

    const float inv_sqrt_head_dim = rsqrtf(static_cast<float>(head_dim));
    if (tid < pair_count) {
        const int base = tid << 1;
        reinterpret_cast<__nv_bfloat162*>(data + static_cast<size_t>(row) * head_dim)[tid] =
            __floats2bfloat162_rn(row_values[base] * inv_sqrt_head_dim,
                                  row_values[base + 1] * inv_sqrt_head_dim);
    }
}

// The kernel processes a 16-row tile. It first applies a normalized H16 to each
// contiguous 16-value segment with WMMA, then finishes the transform with
// butterfly reductions across the segment groups in shared memory.
__global__ void hadamard_transform_kernel_wmma_half_layered(half* data,
                                                            const half* hadamard16_matrix,
                                                            int head_dim,
                                                            int total_rows) {
    const int row_group = blockIdx.x * TILE_DIM;
    const int lane = threadIdx.x;
    const int rows_this_block = total_rows - row_group < TILE_DIM ? total_rows - row_group : TILE_DIM;
    const int group_count = head_dim / TILE_DIM;
    const int float_row_stride = head_dim;

    extern __shared__ unsigned char shared_bytes[];
    half* input_tile = reinterpret_cast<half*>(shared_bytes);
    float* stage_tile =
        reinterpret_cast<float*>(input_tile + static_cast<size_t>(TILE_DIM) * head_dim);

    for (int idx = lane; idx < TILE_DIM * head_dim; idx += blockDim.x) {
        const int row_offset = idx / head_dim;
        const int col = idx % head_dim;
        if (row_offset < rows_this_block) {
            input_tile[idx] = data[static_cast<size_t>(row_group + row_offset) * head_dim + col];
        } else {
            input_tile[idx] = __float2half(0.0f);
        }
    }
    __syncthreads();

    if (lane < 32) {
        for (int group = 0; group < group_count; ++group) {
            const half* a_ptr = input_tile + group * TILE_DIM;

            wmma::fragment<wmma::matrix_a, TILE_DIM, TILE_DIM, TILE_DIM, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, TILE_DIM, TILE_DIM, TILE_DIM, half, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, TILE_DIM, TILE_DIM, TILE_DIM, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);

            wmma::load_matrix_sync(a_frag, a_ptr, head_dim);
            wmma::load_matrix_sync(b_frag, hadamard16_matrix, TILE_DIM);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            wmma::store_matrix_sync(stage_tile + group * TILE_DIM, c_frag, float_row_stride, wmma::mem_row_major);
        }
    }
    __syncthreads();

    const int pair_count = rows_this_block * TILE_DIM * (group_count >> 1);
    for (int len = 1; len < group_count; len <<= 1) {
        for (int pair_linear = lane; pair_linear < pair_count; pair_linear += blockDim.x) {
            const int row_offset = pair_linear / (TILE_DIM * (group_count >> 1));
            const int rem = pair_linear % (TILE_DIM * (group_count >> 1));
            const int col = rem % TILE_DIM;
            const int pair_index = rem / TILE_DIM;
            const int first_group = butterfly_index(pair_index, len);
            float* first = stage_tile + static_cast<size_t>(row_offset) * float_row_stride + first_group * TILE_DIM + col;
            float* second = first + len * TILE_DIM;
            const float u = *first;
            const float v = *second;
            *first = u + v;
            *second = u - v;
        }
        __syncthreads();
    }

    const float inv_sqrt_group_count = rsqrtf(static_cast<float>(group_count));
    for (int idx = lane; idx < rows_this_block * head_dim; idx += blockDim.x) {
        const int row_offset = idx / head_dim;
        const int col = idx % head_dim;
        const float value =
            stage_tile[static_cast<size_t>(row_offset) * float_row_stride + col] *
            inv_sqrt_group_count;
        data[static_cast<size_t>(row_group + row_offset) * head_dim + col] = float_to_scalar<half>(value);
    }
}

__global__ void hadamard_transform_kernel_wmma_bf16_layered(__nv_bfloat16* data,
                                                            const __nv_bfloat16* hadamard16_matrix,
                                                            int head_dim,
                                                            int total_rows) {
#if __CUDA_ARCH__ >= 800
    const int row_group = blockIdx.x * TILE_DIM;
    const int lane = threadIdx.x;
    const int rows_this_block = total_rows - row_group < TILE_DIM ? total_rows - row_group : TILE_DIM;
    const int group_count = head_dim / TILE_DIM;
    const int float_row_stride = head_dim;

    extern __shared__ unsigned char shared_bytes[];
    __nv_bfloat16* input_tile = reinterpret_cast<__nv_bfloat16*>(shared_bytes);
    float* stage_tile =
        reinterpret_cast<float*>(input_tile + static_cast<size_t>(TILE_DIM) * head_dim);

    for (int idx = lane; idx < TILE_DIM * head_dim; idx += blockDim.x) {
        const int row_offset = idx / head_dim;
        const int col = idx % head_dim;
        if (row_offset < rows_this_block) {
            input_tile[idx] = data[static_cast<size_t>(row_group + row_offset) * head_dim + col];
        } else {
            input_tile[idx] = __float2bfloat16_rn(0.0f);
        }
    }
    __syncthreads();

    if (lane < 32) {
        for (int group = 0; group < group_count; ++group) {
            const __nv_bfloat16* a_ptr = input_tile + group * TILE_DIM;

            wmma::fragment<wmma::matrix_a, TILE_DIM, TILE_DIM, TILE_DIM, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, TILE_DIM, TILE_DIM, TILE_DIM, __nv_bfloat16, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, TILE_DIM, TILE_DIM, TILE_DIM, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);

            wmma::load_matrix_sync(a_frag, a_ptr, head_dim);
            wmma::load_matrix_sync(b_frag, hadamard16_matrix, TILE_DIM);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            wmma::store_matrix_sync(stage_tile + group * TILE_DIM, c_frag, float_row_stride, wmma::mem_row_major);
        }
    }
    __syncthreads();

    const int pair_count = rows_this_block * TILE_DIM * (group_count >> 1);
    for (int len = 1; len < group_count; len <<= 1) {
        for (int pair_linear = lane; pair_linear < pair_count; pair_linear += blockDim.x) {
            const int row_offset = pair_linear / (TILE_DIM * (group_count >> 1));
            const int rem = pair_linear % (TILE_DIM * (group_count >> 1));
            const int col = rem % TILE_DIM;
            const int pair_index = rem / TILE_DIM;
            const int first_group = butterfly_index(pair_index, len);
            float* first = stage_tile + static_cast<size_t>(row_offset) * float_row_stride + first_group * TILE_DIM + col;
            float* second = first + len * TILE_DIM;
            const float u = *first;
            const float v = *second;
            *first = u + v;
            *second = u - v;
        }
        __syncthreads();
    }

    const float inv_sqrt_group_count = rsqrtf(static_cast<float>(group_count));
    for (int idx = lane; idx < rows_this_block * head_dim; idx += blockDim.x) {
        const int row_offset = idx / head_dim;
        const int col = idx % head_dim;
        const float value =
            stage_tile[static_cast<size_t>(row_offset) * float_row_stride + col] *
            inv_sqrt_group_count;
        data[static_cast<size_t>(row_group + row_offset) * head_dim + col] = float_to_scalar<__nv_bfloat16>(value);
    }
#else
    (void)data;
    (void)hadamard16_matrix;
    (void)head_dim;
    (void)total_rows;
#endif
}

half* get_hadamard16_matrix_device() {
    static half* device_matrix = nullptr;
    if (device_matrix != nullptr) {
        return device_matrix;
    }

    constexpr int kMatrixDim = TILE_DIM;
    constexpr float kInvSqrt16 = 0.25f;
    std::vector<half> host_matrix(static_cast<size_t>(kMatrixDim) * kMatrixDim);
    for (int row = 0; row < kMatrixDim; ++row) {
        for (int col = 0; col < kMatrixDim; ++col) {
            const int parity = host_popcount(static_cast<unsigned int>(row & col)) & 0x1;
            const float value = parity == 0 ? kInvSqrt16 : -kInvSqrt16;
            host_matrix[static_cast<size_t>(col) * kMatrixDim + row] = __float2half(value);
        }
    }

    CUDA_CHECK(cudaMalloc(&device_matrix, host_matrix.size() * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(device_matrix,
                          host_matrix.data(),
                          host_matrix.size() * sizeof(half),
                          cudaMemcpyHostToDevice));
    return device_matrix;
}

__nv_bfloat16* get_hadamard16_bf16_matrix_device() {
    static __nv_bfloat16* device_matrix = nullptr;
    if (device_matrix != nullptr) {
        return device_matrix;
    }

    constexpr int kMatrixDim = TILE_DIM;
    constexpr float kInvSqrt16 = 0.25f;
    std::vector<__nv_bfloat16> host_matrix(static_cast<size_t>(kMatrixDim) * kMatrixDim);
    for (int row = 0; row < kMatrixDim; ++row) {
        for (int col = 0; col < kMatrixDim; ++col) {
            const int parity = host_popcount(static_cast<unsigned int>(row & col)) & 0x1;
            const float value = parity == 0 ? kInvSqrt16 : -kInvSqrt16;
            host_matrix[static_cast<size_t>(col) * kMatrixDim + row] = __float2bfloat16_rn(value);
        }
    }

    CUDA_CHECK(cudaMalloc(&device_matrix, host_matrix.size() * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemcpy(device_matrix,
                          host_matrix.data(),
                          host_matrix.size() * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));
    return device_matrix;
}

bool device_supports_bf16_wmma() {
    int device = 0;
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    return properties.major >= 8;
}

template <typename T>
void launch_hadamard_v1(T* d_data, int total_rows, int head_dim, cudaStream_t stream) {
    if (!is_power_of_two(head_dim) || head_dim > 256) {
        throw std::invalid_argument("v1 kernel expects head_dim to be a power of two in [1, 256]");
    }

    const dim3 grid(total_rows);
    const dim3 block(head_dim);
    const size_t shared_bytes = static_cast<size_t>(head_dim) * sizeof(float);
    hadamard_transform_kernel_v1<<<grid, block, shared_bytes, stream>>>(d_data, head_dim, total_rows);
    CUDA_CHECK(cudaGetLastError());
}

template <>
void launch_hadamard_v1<half>(half* d_data, int total_rows, int head_dim, cudaStream_t stream) {
    if (!is_power_of_two(head_dim) || head_dim > 256) {
        throw std::invalid_argument("v1 kernel expects head_dim to be a power of two in [1, 256]");
    }

    const dim3 grid(total_rows);
    const dim3 block(head_dim >> 1);
    const size_t shared_bytes = static_cast<size_t>(head_dim) * sizeof(float);
    hadamard_transform_kernel_v1_half2<<<grid, block, shared_bytes, stream>>>(d_data, head_dim, total_rows);
    CUDA_CHECK(cudaGetLastError());
}

template <>
void launch_hadamard_v1<__nv_bfloat16>(__nv_bfloat16* d_data,
                                       int total_rows,
                                       int head_dim,
                                       cudaStream_t stream) {
    if (!is_power_of_two(head_dim) || head_dim > 256) {
        throw std::invalid_argument("v1 kernel expects head_dim to be a power of two in [1, 256]");
    }

    const dim3 grid(total_rows);
    const dim3 block(head_dim >> 1);
    const size_t shared_bytes = static_cast<size_t>(head_dim) * sizeof(float);
    hadamard_transform_kernel_v1_bfloat162<<<grid, block, shared_bytes, stream>>>(d_data, head_dim, total_rows);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launch_hadamard_layered_wmma(T* d_data, int total_rows, int head_dim, cudaStream_t stream) {
    launch_hadamard_v1(d_data, total_rows, head_dim, stream);
}

template <>
void launch_hadamard_layered_wmma<half>(half* d_data, int total_rows, int head_dim, cudaStream_t stream) {
    if (head_dim < TILE_DIM || head_dim % TILE_DIM != 0 || !is_power_of_two(head_dim) || head_dim > 256) {
        launch_hadamard_v1(d_data, total_rows, head_dim, stream);
        return;
    }

    const dim3 grid((total_rows + TILE_DIM - 1) / TILE_DIM);
    const dim3 block(256);
    const size_t shared_bytes = static_cast<size_t>(TILE_DIM) * head_dim * sizeof(half) +
                                static_cast<size_t>(TILE_DIM) * head_dim * sizeof(float);

    hadamard_transform_kernel_wmma_half_layered<<<grid, block, shared_bytes, stream>>>(
        d_data, get_hadamard16_matrix_device(), head_dim, total_rows);
    CUDA_CHECK(cudaGetLastError());
}

template <>
void launch_hadamard_layered_wmma<__nv_bfloat16>(__nv_bfloat16* d_data,
                                                 int total_rows,
                                                 int head_dim,
                                                 cudaStream_t stream) {
    if (head_dim < TILE_DIM || head_dim % TILE_DIM != 0 || !is_power_of_two(head_dim) || head_dim > 256 ||
        !device_supports_bf16_wmma()) {
        launch_hadamard_v1(d_data, total_rows, head_dim, stream);
        return;
    }

    const dim3 grid((total_rows + TILE_DIM - 1) / TILE_DIM);
    const dim3 block(256);
    const size_t shared_bytes = static_cast<size_t>(TILE_DIM) * head_dim * sizeof(__nv_bfloat16) +
                                static_cast<size_t>(TILE_DIM) * head_dim * sizeof(float);

    hadamard_transform_kernel_wmma_bf16_layered<<<grid, block, shared_bytes, stream>>>(
        d_data, get_hadamard16_bf16_matrix_device(), head_dim, total_rows);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launch_hadamard_wmma(T* d_data, int total_rows, int head_dim, cudaStream_t stream) {
    launch_hadamard_layered_wmma(d_data, total_rows, head_dim, stream);
}

}  // namespace

template <typename T>
void hadamard_transform_gpu(T* d_data,
                            int batch_size,
                            int seq_len,
                            int num_heads,
                            int head_dim,
                            cudaStream_t stream) {
    launch_hadamard_v1(d_data, checked_total_rows(batch_size, seq_len, num_heads), head_dim, stream);
}

template <typename T>
void hadamard_transform_gpu_wmma(T* d_data,
                                 int batch_size,
                                 int seq_len,
                                 int num_heads,
                                 int head_dim,
                                 cudaStream_t stream) {
    hadamard_transform_gpu(d_data, batch_size, seq_len, num_heads, head_dim, stream);
}

template <typename T>
void hadamard_transform_gpu_best(T* d_data,
                                 int batch_size,
                                 int seq_len,
                                 int num_heads,
                                 int head_dim,
                                 cudaStream_t stream) {
    hadamard_transform_gpu(d_data, batch_size, seq_len, num_heads, head_dim, stream);
}

template <>
void hadamard_transform_gpu_wmma<half>(half* d_data,
                                       int batch_size,
                                       int seq_len,
                                       int num_heads,
                                       int head_dim,
                                       cudaStream_t stream) {
    if (head_dim % TILE_DIM != 0 || head_dim > 256) {
        launch_hadamard_v1(d_data, checked_total_rows(batch_size, seq_len, num_heads), head_dim, stream);
        return;
    }

    const int total_rows = checked_total_rows(batch_size, seq_len, num_heads);
    launch_hadamard_wmma(d_data, total_rows, head_dim, stream);
}

template <>
void hadamard_transform_gpu_wmma<__nv_bfloat16>(__nv_bfloat16* d_data,
                                                int batch_size,
                                                int seq_len,
                                                int num_heads,
                                                int head_dim,
                                                cudaStream_t stream) {
    if (head_dim % TILE_DIM != 0 || head_dim > 256) {
        launch_hadamard_v1(d_data, checked_total_rows(batch_size, seq_len, num_heads), head_dim, stream);
        return;
    }

    const int total_rows = checked_total_rows(batch_size, seq_len, num_heads);
    launch_hadamard_wmma(d_data, total_rows, head_dim, stream);
}

template <>
void hadamard_transform_gpu_best<half>(half* d_data,
                                       int batch_size,
                                       int seq_len,
                                       int num_heads,
                                       int head_dim,
                                       cudaStream_t stream) {
    hadamard_transform_gpu(d_data, batch_size, seq_len, num_heads, head_dim, stream);
}

template <>
void hadamard_transform_gpu_best<__nv_bfloat16>(__nv_bfloat16* d_data,
                                                int batch_size,
                                                int seq_len,
                                                int num_heads,
                                                int head_dim,
                                                cudaStream_t stream) {
    hadamard_transform_gpu(d_data, batch_size, seq_len, num_heads, head_dim, stream);
}

template void hadamard_transform_gpu<half>(half*, int, int, int, int, cudaStream_t);
template void hadamard_transform_gpu<__nv_bfloat16>(__nv_bfloat16*, int, int, int, int, cudaStream_t);
template void hadamard_transform_gpu_wmma<half>(half*, int, int, int, int, cudaStream_t);
template void hadamard_transform_gpu_wmma<__nv_bfloat16>(__nv_bfloat16*, int, int, int, int, cudaStream_t);
template void hadamard_transform_gpu_best<half>(half*, int, int, int, int, cudaStream_t);
template void hadamard_transform_gpu_best<__nv_bfloat16>(__nv_bfloat16*, int, int, int, int, cudaStream_t);

}  // namespace hadamard
