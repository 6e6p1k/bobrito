#include <iostream>
#include <memory>
#include <cstring>
#include "utils.h"
#include "node.h"

using namespace std;
namespace utils 
{

void compute_strides(int* dst, const int ndim, const int* shape) {
    int stride = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        dst[i] = stride;
        stride *= shape[i];
    }
}

int numel(const int ndim, const int* shape) {
    int n = 1;
    for (int i = 0; i < ndim; i++) n *= shape[i];
    return n;
}

}

namespace broadcast
{
BroadcastInfo elemwise(int* a_shape, int* b_shape, int* a_strides, int* b_strides, int* out_shape, int out_ndim)
{
    BroadcastInfo info;
    info.ndim = out_ndim;
    for (int i = 0; i < out_ndim; i++) {
        info.out_shape[i] = out_shape[i];
        // if input dim == 1 (broadcast), stride = 0, else real stride
        info.a_strides[i] = (a_shape[i] == 1) ? 0 : a_strides[i];
        info.b_strides[i] = (b_shape[i] == 1) ? 0 : b_strides[i];
    }
    return info;
}

BroadcastInfo elemwise(InternalNode* node)
{
    // input node MUST have exactly 2 inputs
    // a_info
    int* a_shape = node->inputs[0]->h_meta->shape();
    int* a_strides = node->inputs[0]->h_meta->strides();
    // b_info
    int* b_shape = node->inputs[1]->h_meta->shape();
    int* b_strides = node->inputs[1]->h_meta->strides();
    // out_info
    int* out_shape = node->h_meta->shape();
    int out_ndim = node->h_meta->ndim;

    return elemwise(a_shape, b_shape, a_strides, b_strides, out_shape, out_ndim);
}

MatmulDims matmul(int* a_shape, int* b_shape, int* res_shape, int ndim) {
    int M = a_shape[ndim - 2];
    int K = a_shape[ndim - 1];
    int N = b_shape[ndim - 1];
    int batch = 1;
    for (int i = 0; i < ndim - 2; i++) batch *= a_shape[i];

    return {M, K, N, batch, res_shape};
}

MatmulDims matmul(InternalNode* node) {
    // a_info
    int* a_shape = node->inputs[0]->h_meta->shape();
    // b_info
    int* b_shape = node->inputs[1]->h_meta->shape();
    // out_info
    int* out_shape = node->h_meta->shape();
    int out_ndim = node->h_meta->ndim;

    return matmul(a_shape, b_shape, out_shape, out_ndim);
}

ContiguousInfo contiguous(int* res_shape, int* input_strides, int ndim)
{
    ContiguousInfo info;
    for (int i = 0; i < ndim; ++i)
    {
        info.shape[i] = res_shape[i];
        info.strides[i] = input_strides[i];
    }
    info.ndim = ndim;

    return info;
}

ContiguousInfo contiguous(InternalNode* node)
{
    int* res_shape = node->h_meta->shape();
    int* input_strides = node->inputs[0]->h_meta->strides();
    int ndim = node->h_meta->ndim;
    return contiguous(res_shape, input_strides, ndim);
}
}

namespace check_shape_return_out_shape 
{

pair<int*, int> elemwise(const int* a_shape, const int* b_shape, int a_size, int b_size)
    {
        int ndim = max(a_size, b_size);
        int* res_shape = new int[ndim];
        for (int i = 0; i < ndim; i++)
        {
            int a_dim = i < a_size ? a_shape[a_size - 1 - i] : 1;  // pad with 1s from left
            int b_dim = i < b_size ? b_shape[b_size - 1 - i] : 1;
            if (a_dim != b_dim && a_dim != 1 && b_dim != 1)
                throw logic_error("Shapes are not broadcastable.");
            res_shape[ndim - 1 - i] = max(a_dim, b_dim);
        }
        return {res_shape, ndim};
    }

pair<int*, int> elemwise(shared_ptr<Node> A, shared_ptr<Node> B)
{
    int* a_shape = A->h_meta->shape();
    int* b_shape = B->h_meta->shape();
    int a_size = A->h_meta->ndim;
    int b_size = B->h_meta->ndim;

    return elemwise(a_shape, b_shape, a_size, b_size);
}

pair<int*, int> reduction(const int* a_shape, const int ndim, const int dim, const bool keepdims)
{
    int d = (dim + ndim) % ndim; // handle negative indexing
    int res_ndim = keepdims ? ndim : ndim - 1;
    int* res_shape = new int[res_ndim];
    int j = 0;
    for (int i = 0; i < ndim; i++)
    {
        if (i == d) { if (keepdims) res_shape[j++] = 1; }
        else res_shape[j++] = a_shape[i];
    }
    return {res_shape, res_ndim};
}

pair<int*, int> reduction(shared_ptr<Node> A, const int dim, const bool keepdims)
{
    // retrieve data
    int ndim = A->h_meta->ndim;
    int* a_shape = A->h_meta->shape();

    return reduction(a_shape, ndim, dim, keepdims);
}

pair<int*, int> matmul(int* a_shape, int* b_shape, int a_ndim, int b_ndim) {
    if (!(a_ndim >= 2 && b_ndim >= 2))
        throw std::logic_error("Tensors rank should be at least 2.");
    if (a_shape[a_ndim - 1] != b_shape[b_ndim - 2])
        throw std::logic_error("k dimensions don't match.");

    int out_ndim = max(a_ndim, b_ndim);
    int* res_shape = new int[out_ndim];

    // check/broadcast batch dims (pad shorter one with 1s from left)
    int a_off = out_ndim - a_ndim;
    int b_off = out_ndim - b_ndim;
    for (int i = 0; i < out_ndim - 2; i++) {
        int a_dim = (i < a_off) ? 1 : a_shape[i - a_off];
        int b_dim = (i < b_off) ? 1 : b_shape[i - b_off];
        if (a_dim != b_dim && a_dim != 1 && b_dim != 1)
            throw std::logic_error("Batch dimensions should match.");
        res_shape[i] = max(a_dim, b_dim);
    }

    res_shape[out_ndim - 2] = a_shape[a_ndim - 2];
    res_shape[out_ndim - 1] = b_shape[b_ndim - 1];
    return {res_shape, out_ndim};
}

pair<int*, int> matmul(shared_ptr<Node> A, shared_ptr<Node> B) {
    int a_ndim = A->h_meta->ndim;
    int b_ndim = B->h_meta->ndim;
    int* a_shape;
    int* b_shape;

    // handle tranposed
    if (A->transposed) {
        a_shape = new int[a_ndim];
        memcpy(a_shape, A->h_meta->shape(), A->h_meta->ndim * sizeof(int));
        int tmp = a_shape[a_ndim - 1];
        a_shape[a_ndim - 1] = a_shape[a_ndim - 2];
        a_shape[a_ndim - 2] = tmp; 
    } else a_shape = A->h_meta->shape();

    if (B->transposed) {
        b_shape = new int[b_ndim];
        memcpy(b_shape, B->h_meta->shape(), B->h_meta->ndim * sizeof(int));
        int tmp = b_shape[b_ndim - 1];
        b_shape[b_ndim - 1] = b_shape[b_ndim - 2];
        b_shape[b_ndim - 2] = tmp; 
    } else b_shape = B->h_meta->shape();

    return matmul(a_shape, b_shape, a_ndim, b_ndim);
}

// future


// MatmulDims matmul(shared_ptr<Node> A, shared_ptr<Node> B) {
//     int* a_shape = A->h_meta->shape();
//     int* b_shape = B->h_meta->shape();
//     int a_ndim = A->h_meta->ndim;

//     return matmul(a_shape, b_shape, a_ndim);
// }

}