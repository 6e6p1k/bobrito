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

__global__ void bwd_relu(float* fwd_data, float* prev_grad, float* new_grad, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) new_grad[i] = fwd_data[i] > 0 ? prev_grad[i] : 0;
}

__global__ void fused_bwd_pow(float* mantissa, float* exponent, float* prev_grad, float* new_grad, int n) {
    // C = A ^ B
    // dC/dA = dC * B * A ^ (B - 1) 
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) new_grad[i] = prev_grad[i] * exponent[i] * powf(mantissa[i], (exponent[i] - 1.0f));
}

__global__ void sgd(float* weight_data, float* grad_data, float lr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) weight_data[i] -= grad_data[i] * lr;
}

}

namespace lib_ops
{

template <typename ReduceOp>
void reduce_last_dim(InternalNode* res_node) {
    float* res_data = res_node->get_data();
    float* input_data = res_node->inputs[0]->get_data();
    int* shape = res_node->inputs[0]->h_meta->shape();
    int ndim = res_node->inputs[0]->h_meta->ndim;
    int size = res_node->inputs[0]->h_meta->size;
    
    int last_dim = shape[ndim - 1];
    int num_segments = size / last_dim;
    
    Block* b_ptr;
    MemoryPool::call().allocate(&b_ptr, (num_segments + 1) * sizeof(int));
    int* offsets = (int*)b_ptr->d_ptr;
    compute_offset(offsets, last_dim, num_segments);

    size_t temp_storage_bytes = 0;

    ReduceOp::reduce(nullptr, temp_storage_bytes, input_data, res_data, num_segments, offsets, offsets + 1);

    Block* temp_block;
    MemoryPool::call().allocate(&temp_block, temp_storage_bytes);
    
    ReduceOp::reduce(temp_block->d_ptr, temp_storage_bytes, input_data, res_data, num_segments, offsets, offsets + 1);
    
    MemoryPool::call().free(temp_block);
    MemoryPool::call().free(b_ptr);
    // CUDA_CHECK();
}
}