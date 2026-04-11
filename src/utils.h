#pragma once
#include <iostream>
#include <exception>
#include "nodes.h"

namespace utils 
{

void compute_strides(int* dst, const int ndim, const int* shape);
int numel(const int ndim, const int* shape);

}


#define MAX_DIMS 8

struct BroadcastInfo {
    int out_shape[MAX_DIMS];
    int a_strides[MAX_DIMS];   // 0 for broadcast dims
    int b_strides[MAX_DIMS];   // 0 for broadcast dims
    int ndim;
};

struct MatmulDims {
    int M, K, N, batch;
    int* res_shape;
};

struct ContiguousInfo {
    int shape[MAX_DIMS];
    int strides[MAX_DIMS];
    int ndim;
};

namespace broadcast
{
BroadcastInfo elemwise(int* a_shape, int* b_shape, int* a_strides, int* b_strides, int* out_shape, int out_ndim, int a_ndim, int b_ndim);
BroadcastInfo elemwise(Node* node);
MatmulDims matmul(int* a_shape, int* b_shape, int* res_shape, int ndim);
MatmulDims matmul(Node* node);
ContiguousInfo contiguous(int* res_shape, int* input_strides, int ndim);
ContiguousInfo contiguous(Node* node);
}


namespace check_shape_return_out_shape 
{
std::pair<int*, int> elemwise(const int* a_shape, const int* b_shape, int a_size, int b_size);
std::pair<int*, int> elemwise(std::shared_ptr<Node> A, std::shared_ptr<Node> B);
std::pair<int*, int> reduction(const int* a_shape, const int ndim, const int dim, const bool keepdims);
std::pair<int*, int> reduction(std::shared_ptr<Node> A, const int dim, const bool keepdims);
std::pair<int*, int> matmul(int* a_shape, int* b_shape, int a_ndim, int b_ndim);
std::pair<int*, int> matmul(std::shared_ptr<Node> A, std::shared_ptr<Node> B);
}