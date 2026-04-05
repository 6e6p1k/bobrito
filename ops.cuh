#pragma once
#include "node.h"
#include "utils.h"
#include <cub/cub.cuh>


struct cubSum {
    static cudaError_t reduce(void* tmp, size_t& tmp_bytes, float* in, float* out, int n, int* begin, int* end) {
        return cub::DeviceSegmentedReduce::Sum(tmp, tmp_bytes, in, out, n, begin, end);
    }
};

struct cubMax {
    static cudaError_t reduce(void* tmp, size_t& tmp_bytes, float* in, float* out, int n, int* begin, int* end) {
        return cub::DeviceSegmentedReduce::Max(tmp, tmp_bytes, in, out, n, begin, end);
    }
};

void compute_offset(int* offsets, int last_dim, int num_segments);

namespace cu_ops
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

void add(InternalNode* res_node);
void sub(InternalNode* res_node);
void mul(InternalNode* res_node);
void div(InternalNode* res_node);
void matmul(InternalNode* res_node);
void skip(InternalNode* res_node);
void fill(Node* node, float val);
void fill(SourceNode* node, float val);
void fill(InternalNode* node, float val);
void contiguous(InternalNode* res_node);
void reduce_last_dim(InternalNode* res_node);
void expand_last_dim(InternalNode* res_node);
void relu(InternalNode* res_node);
void bwd_relu(InternalNode* res_node);
void sum_last_dim(InternalNode* res_node);
void max_last_dim(InternalNode* res_node);
void min_last_dim(InternalNode* res_node);
void argmax_last_dim(InternalNode* res_node);

// float* matmul_cuda_launch(const float* A, const float* B, int batch, int M, int N, int K);





//  float* src, float* dst, int* shape, int ndim, int n)

}
