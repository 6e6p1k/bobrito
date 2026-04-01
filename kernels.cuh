#pragma once


struct cuAdd { __device__ static float apply(float a, float b) { return a + b; } };
struct cuMul { __device__ static float apply(float a, float b) { return a * b; } };

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



}
