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

__global__ void expand_last_dim(float* src, float* dst, int last_dim, int n) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= n) return;
    dst[out_idx] = src[out_idx / last_dim];
}

__global__ void relu(float* input, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = input[i] > 0 ? input[i] : 0;
}

__global__ void bwd_relu(float* fwd_data, float* prev_grad, float* new_grad, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) new_grad[i] = fwd_data[i] > 0 ? prev_grad[i] : 0;
}

__global__ void sgd(float* weight_data, float* grad_data, float lr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) weight_data[i] -= grad_data[i] * lr;
}

__global__ void log(float* src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = std::log(src[i]);
}

}