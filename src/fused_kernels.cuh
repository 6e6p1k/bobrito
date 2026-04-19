#pragma once
// using single header file to reduce the work of nvrtc compiler 

#define MAX_DIMS 8

struct BroadcastInfo {
    int out_shape[MAX_DIMS];
    int a_strides[MAX_DIMS];
    int b_strides[MAX_DIMS];
    int ndim;
};

__device__ void elemwise_idx(int out_idx, int& a_idx, int& b_idx, BroadcastInfo info);

#ifdef ELEMWISE_IDX_IMPL
__device__ void elemwise_idx(int out_idx, int& a_idx, int& b_idx, BroadcastInfo info)
{
    a_idx = 0; b_idx = 0;
    int remaining = out_idx;
    for (int i = info.ndim - 1; i >= 0; i--) {
        int coord = remaining % info.out_shape[i];
        remaining /= info.out_shape[i];
        a_idx += coord * info.a_strides[i];
        b_idx += coord * info.b_strides[i];
    }
}
#endif