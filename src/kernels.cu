#include "utils.h"
#include "kernels.cuh"


namespace kernels
{
__global__ void elemwise(float* a, float* b, float* out, int n, BroadcastInfo info);

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
// HAVE TO THINK ABOUT POW
// SHOULD ADD ONE FOR CONSTANT

__global__ void sgd(float* weight_data, float* grad_data, float lr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) weight_data[i] -= grad_data[i] * lr;
}

__global__ void scale(float* data, float limit, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = data[i] * 2.0f * limit - limit;
}

__global__ void bwd_div(float* A, float* B, float* prev_grad, float* dst, int n, BroadcastInfo info) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= n) return;
    // decompose flat index into nd, then recompose for each input
    int a_idx = 0, b_idx = 0;
    int remaining = out_idx;
    for (int i = info.ndim - 1; i >= 0; i--) {
        int coord = remaining % info.out_shape[i];
        remaining /= info.out_shape[i];
        a_idx += coord * info.a_strides[i];
        b_idx += coord * info.b_strides[i];
    }
    // C = A / B
    // This function computes grad for B
    // dB = -dL/dC * A / B^2 = - prev_grad * A / pow(B, 2)
    dst[out_idx] = -prev_grad[out_idx] * A[a_idx] / powf(B[b_idx], 2.0f);
}

__global__ void max_min_bwd_last_dim(
    const float* fwd_data,  // Shape [N, D]
    const float* target,    // Shape [N] (The max/min values)
    const float* prev_grad, // Shape [N] (The incoming gradients)
    float* new_grad,        // Shape [N, D]
    int N,                  // Number of rows (leading dims)
    int D                   // Size of the last dimension
) {
    // Standard global index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Total elements is N * D
    if (i < N * D) {
        // Determine which row (leading dim index) this thread belongs to
        int row_idx = i / D; 
        
        // Compare current element to the target of its specific row
        // If it matches, it gets the gradient; otherwise, 0.
        if (fwd_data[i] == target[row_idx]) {
            new_grad[i] = prev_grad[row_idx];
        } else {
            new_grad[i] = 0.0f;
        }
    }
}
}