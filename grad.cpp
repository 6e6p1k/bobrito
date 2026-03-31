#include <cstddef>
#include <iostream>
#include "utils.h"
#include "allocator.cuh"
#include "grad.h"

using namespace std;


// <ShapeData>
// static initialisation
ShapeData* ShapeData::init(int ndim)
{   
    size_t total_bytes = sizeof(ShapeData) + sizeof(int) * ndim * 2;

    ShapeData* sd = (ShapeData*)malloc(total_bytes);
    if (sd) sd->ndim = ndim;
    return sd;
}

int* ShapeData::shape() { return &data[0]; }
int* ShapeData::strides() { return &data[ndim]; }

void ShapeData::print() {
    std::cout << "Shape: ";
    for(int i = 0; i < ndim; i++) std::cout << shape()[i] << " ";
    std::cout << "\nStrides: ";
    for(int i = 0; i < ndim; i++) std::cout << strides()[i] << " ";
    std::cout << std::endl;
}
// </ShapeData>


// <ShapeData helper functions>
ShapeData* make_shape_data(int ndim, int* shape, int* strides)
{
    ShapeData* sd = ShapeData::init(ndim);
    // case 1: strides are not specified 
    if (strides == nullptr)
    {
        utils::compute_strides(sd->strides(), ndim, shape);
        for (int i = 0; i < ndim; i++) sd->shape()[i] = shape[i];
        return sd;
    }
    // case 2: strides are specified 
    for (int i = 0; i < ndim; i++)
    {
        sd->shape()[i] = shape[i];
        sd->strides()[i] = strides[i];
    }
    return sd;
}
// </ShapeData helper functions>


// <Node>
// source node initialisation
SourceNode::SourceNode(float* data, int ndim, int* shape, int* strides, bool requires_grad)
{
    kind = NodeKind::Source;
    // assign general parameters
    h_requiers_grad = requires_grad;
    
    // transfer array content to device
    MemoryPool::call().data_to_device(&data_block, data, utils::numel(ndim, shape));
    
    // store metadata
    h_meta = make_shape_data(ndim, shape, strides);
}

Node* make_internal_node(OpType op, Node* inputs[], int num_inputs, int* shape, int ndim, int* strides)
{
    InternalNode* node = new InternalNode();
    node->kind = NodeKind::Internal;
    node->op = op;
    node->num_inputs = num_inputs;
    node->h_meta = make_shape_data(ndim, shape, strides);
    for (int i = 0; i < num_inputs; i++) node->inputs[i] = inputs[i];
    return node;
}
// </Node>


// <nodes interaction>
//      <general operations> 
Node* elemwise(Node* a, Node* b, OpType op) 
{
    auto [res_shape, res_ndim] = check_shape_return_out_shape::elemwise(
        a, b
    );
    Node* inputs[] = {a, b};
    Node* node = make_internal_node(
        op, inputs, 2, res_shape, res_ndim
    );

    return node;
}

Node* reduction(Node* a, int dim, bool keepdims, OpType op)
{
    auto [res_shape, res_ndim] = check_shape_return_out_shape::reduction(
        a, dim, keepdims
    );
    Node* inputs[] = {a};
    Node* node = make_internal_node(
        op, inputs, 1, res_shape, res_ndim
    );
    return node;
}
//      </general operations> 


Node* add(Node* a, Node* b) 
{
    return elemwise(a, b, OpType::Add);
}

Node* mul(Node* a, Node* b) 
{
    return elemwise(a, b, OpType::Mul);
}

Node* sum(Node* a, int dim, bool keepdims)
{
    return reduction(a, dim, keepdims, OpType::Sum);
}

Node* max(Node* a, int dim, bool keepdims)
{
    return reduction(a, dim, keepdims, OpType::Max);
}

Node* min(Node* a, int dim, bool keepdims)
{
    return reduction(a, dim, keepdims, OpType::Min);
}
// </nodes interaction>