#pragma once
#include <iostream>
#include <exception>
#include "grad.h"

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


namespace check_shape_return_out_shape 
{
pair<int*, int> elemwise(const int* a_shape, const int* b_shape, int a_size, int b_size);
pair<int*, int> elemwise(Node* A, Node* B);
pair<int*, int> reduction(const int* a_shape, const int a_size, const int ndim, const int dim, const bool keepdims);
pair<int*, int> reduction(Node* A, const int dim, const bool keepdims);
}