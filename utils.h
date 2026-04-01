#pragma once
#include <iostream>
#include <exception>
#include "node.h"

using namespace std;
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

namespace broadcast
{
BroadcastInfo elemwise(int* a_shape, int* b_shape, int* a_strides, int* b_strides, int* out_shape, int out_ndim);
BroadcastInfo elemwise(InternalNode* node);
}

struct MatmulDims {
    int M, K, N, batch;
    int* res_shape;
};

namespace check_shape_return_out_shape 
{
pair<int*, int> elemwise(const int* a_shape, const int* b_shape, int a_size, int b_size);
pair<int*, int> elemwise(shared_ptr<Node> A, shared_ptr<Node> B);
pair<int*, int> reduction(const int* a_shape, const int ndim, const int dim, const bool keepdims);
pair<int*, int> reduction(shared_ptr<Node> A, const int dim, const bool keepdims);
pair<int*, int> matmul(int* a_shape, int* b_shape, int a_ndim, int b_ndim);
pair<int*, int> matmul(shared_ptr<Node> A, shared_ptr<Node> B);
}