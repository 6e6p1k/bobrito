#pragma once
#include "allocator.cuh"
#include <cstddef>


enum class OpType { Add, Mul, MatMul, Sum, Max, Min, ReLU};
enum class NodeKind { Source, Internal };


// ShapeData
struct ShapeData 
{
    int ndim;
    int data[0];

    static ShapeData* init(int ndim);
    int* shape();
    int* strides();
    void print();
};

ShapeData* make_shape_data(int ndim, int* shape, int* strides = nullptr);


// Node
struct Node
{
    NodeKind kind;
    ShapeData* h_meta;  // shape, strides, ndim on host
    float* d_grad = nullptr;  // grad on device
};

struct SourceNode : Node
{   
    Block* data_block = nullptr;  // data on device
    bool h_requiers_grad = false; 

    SourceNode() = default;
    SourceNode(float* data, int ndim, int* shape, int* strides = nullptr, bool requires_grad = false);
};

struct InternalNode : Node
{
    OpType op;  // operation that was used to create this node 
    Node* inputs[4];
    int num_inputs;
};

Node* make_internal_node(OpType op, Node* inputs[], int num_inputs, int* shape, int ndim, int* strides = nullptr);

// <nodes interaction>
//      <general operations> 
Node* elemwise(Node* a, Node* b, OpType op);
Node* reduction(Node* a, int dim, OpType op, bool keepdims);
//      </general operations> 

Node* add(Node* a, Node* b);
Node* mul(Node* a, Node* b);
Node* sum(Node* a, int dim, bool keepdims = false);
Node* max(Node* a, int dim, bool keepdims = false);
Node* min(Node* a, int dim, bool keepdims = false);


// </nodes interaction>