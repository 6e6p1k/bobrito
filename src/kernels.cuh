#pragma once

#include "utils.h"

// elemwise 
struct cuAdd { __device__ static float apply(float a, float b) { return a + b; } };
struct cuMul { __device__ static float apply(float a, float b) { return a * b; } };
struct cuSub { __device__ static float apply(float a, float b) { return a - b; } };
struct cuDiv { __device__ static float apply(float a, float b) { return a / b; } };
struct cuPow { __device__ static float apply(float a, float b) { return powf(a, b); } };

// unary
struct cuReLU  { __device__ static float apply(float x) { return x > 0 ? x : 0; } };
struct cuLog   { __device__ static float apply(float x) { return logf(fmaxf(x, 1e-7f)); } };
struct cuExp   { __device__ static float apply(float x) { return expf(x); } };
struct cuNeg   { __device__ static float apply(float x) { return -x; } };
struct cuSqrt  { __device__ static float apply(float x) { return sqrtf(x); } };

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

template <typename Op>
__global__ void unary(float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = Op::apply(in[i]);
}

__global__ void fill(float* data, float val, int n);
__global__ void arange(float* data, float start, float step, int n);
__global__ void copy(float* src, float* dst, int n);
__global__ void contiguous(float* src, float* dst, int size, ContiguousInfo info);
__global__ void reduction_offsets(int* offsets, int last_dim, int num_segments);
__global__ void expand_last_dim(float* src, float* dst, int last_dim, int n);
__global__ void relu(float* src, float* dst, int n);
__global__ void bwd_relu(float* fwd_data, float* prev_grad, float* new_grad, int n);
__global__ void fused_bwd_pow(float* mantissa, float exponent, float* prev_grad, float* new_grad, int n);
__global__ void sgd(float* weight_data, float* grad_data, float lr, int n);
}