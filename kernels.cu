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


}
