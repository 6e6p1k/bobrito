#include "utils.h"
#include "kernels.cuh"


namespace kernels
{
__global__ void elemwise(float* a, float* b, float* out, BroadcastInfo info);

__global__ void fill(float* data, float val, int n) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = val;
}

__global__ void copy(float* src, float* dst, int n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) dst[id] = src[id];
}

__global__ void contiguous(float* src, float* dst, int size, ContiguousInfo info)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= size) return;
    
    // decompose flat output index into nd coords
    int src_idx = 0;
    int remaining = out_idx;
    for (int i = info.ndim - 1; i >= 0; i--) {
        int coord = remaining % info.shape[i];
        remaining /= info.shape[i];
        src_idx += coord * info.strides[i];
    }
    
    dst[out_idx] = src[src_idx];
}

__global__ void reduction_offsets(int* offsets, int last_dim, int num_segments) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= num_segments) offsets[i] = i * last_dim;
}

}