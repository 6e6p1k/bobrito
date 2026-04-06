#include "node_helpers.h"
#include "tensor.cu"
#include "utils.h"

class Bobr {
public:
    static Tensor ones(std::vector<int> shape, bool requires_grad)
    { return make_filled_source_node(shape, shape.size(), 1.0f); }

    static Tensor zeros(std::vector<int> shape, bool requires_grad)
    { return make_filled_source_node(shape, shape.size(), 0.0f); }

    static Tensor ones_like(Tensor t)
    { 
        int* shape = t.node->h_meta->shape();
        int ndim = t.node->h_meta->ndim;
        return make_filled_source_node(shape, ndim, 1.0f);
    }

    static Tensor zeros_like(Tensor t)
    { 
        int* shape = t.node->h_meta->shape();
        int ndim = t.node->h_meta->ndim;
        return make_filled_source_node(shape, ndim, 0.0f);
    }

};