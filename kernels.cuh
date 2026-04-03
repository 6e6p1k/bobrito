#pragma once


struct cuAdd { __device__ static float apply(float a, float b) { return a + b; } };
struct cuMul { __device__ static float apply(float a, float b) { return a * b; } };

// struct cubSum {
//     static cudaError_t reduce(void* tmp, size_t& tmp_bytes, float* in, float* out, int n, int* begin, int* end) {
//         return cub::DeviceSegmentedReduce::Sum(tmp, tmp_bytes, in, out, n, begin, end);
//     }
// };

// struct cubMax {
//     static cudaError_t reduce(void* tmp, size_t& tmp_bytes, float* in, float* out, int n, int* begin, int* end) {
//         return cub::DeviceSegmentedReduce::Max(tmp, tmp_bytes, in, out, n, begin, end);
//     }
// };

namespace kernels
{

template <typename Op>
__global__ void elemwise(float* a, float* b, float* out, BroadcastInfo info) 
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // decompose flat index into nd, then recompose for each input
    int a_idx = 0, b_idx = 0;
    int remaining = out_idx;
    for (int i = info.ndim - 1; i >= 0; i--) {
        int coord = remaining % info.out_shape[i];
        remaining /= info.out_shape[i];
        a_idx += coord * info.a_strides[i];
        b_idx += coord * info.b_strides[i];
    }
    
    out[out_idx] = Op::apply(a[a_idx], b[b_idx]);
}

__global__ void fill(float* data, float val, int n);
__global__ void copy(float* src, float* dst, int n);
__global__ void contiguous(float* src, float* dst, int size, ContiguousInfo info);
__global__ void reduction_offsets(int* offsets, int last_dim, int num_segments);

}

// namespace libs_ops
// {
// template <typename Op>
// void segmented_reduce(float* input, float* output, int* offsets, int num_segments) {
//     void* temp_storage = nullptr;
//     size_t temp_storage_bytes = 0;

//     Op::reduce(nullptr, temp_storage_bytes, input, output, num_segments, offsets, offsets + 1);

//     Block* temp_block;
//     MemoryPool::call().allocate(&temp_block, temp_storage_bytes);
    
//     Op::reduce(temp_block->d_ptr, temp_storage_bytes, input, output, num_segments, offsets, offsets + 1);
    
//     MemoryPool::call().free(temp_block);
// }

// }
