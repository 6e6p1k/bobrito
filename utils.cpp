#include <iostream>
#include "utils.h"
#include "grad.h"

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

pair<int*, int> elemwise(Node* A, Node* B)
{
    int* a_shape = A->h_meta->shape();
    int* b_shape = B->h_meta->shape();
    int a_size = A->h_meta->ndim;
    int b_size = B->h_meta->ndim;

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

pair<int*, int> reduction(const int* a_shape, const int a_size, const int ndim, const int dim, const bool keepdims)
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

pair<int*, int> reduction(Node* A, const int dim, const bool keepdims)
{
    // retrieve data
    int ndim = A->h_meta->ndim;
    int* a_shape = A->h_meta->shape();

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


}