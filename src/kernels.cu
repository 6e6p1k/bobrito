#include "cub/device/dispatch/dispatch_streaming_reduce.cuh"
#include "utils.h"
#include "kernels.cuh"


namespace kernels
{
__global__ void elemwise(float* a, float* b, float* out, BroadcastInfo info);

__global__ void fill(float* dst, float val, int n) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = val;
}

__global__ void arange(float* dst, float start, float step, int n) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = start + i * step;
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

__global__ void bwd_relu(float* fwd_data, float* prev_grad, float* new_grad, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) new_grad[i] = fwd_data[i] > 0 ? prev_grad[i] : 0;
}

__global__ void fused_bwd_pow(float* mantissa, float* exponent, float* prev_grad, float* new_grad, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) new_grad[i] = prev_grad[i] * exponent[i] * powf(mantissa[i], (exponent[i] - 1.0f));
}

__global__ void pow(float* in, float* exponent, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = powf(in[i], exponent[i] );
}

__global__ void sgd(float* weight_data, float* grad_data, float lr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) weight_data[i] -= grad_data[i] * lr;
}

__global__ void scale(float* data, float limit, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = data[i] * 2.0f * limit - limit;
}

__global__ void bwd_div(float* src, float* dst, float* prev_grad, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = prev_grad[i] * 1.0f / src[i];
}
}