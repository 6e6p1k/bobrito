#pragma once
#include "node.h"
#include "utils.h"


namespace cu_ops
{
void add(InternalNode* res_node);
void gadd(InternalNode* res_node);
void mul(InternalNode* res_node);
void matmul(InternalNode* res_node);
void skip(InternalNode* res_node);
void fill(Node* node, float val);
void fill(SourceNode* node, float val);
void fill(InternalNode* node, float val);
// float* matmul_cuda_launch(const float* A, const float* B, int batch, int M, int N, int K);
}
