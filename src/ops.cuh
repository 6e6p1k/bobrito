#pragma once
#include "node.h"
#include "utils.h"

void compute_offset(int* offsets, int last_dim, int num_segments);

namespace cu_ops
{
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
void sgd(SourceNode* snode, float lr);

// float* matmul_cuda_launch(const float* A, const float* B, int batch, int M, int N, int K);





//  float* src, float* dst, int* shape, int ndim, int n)

}
