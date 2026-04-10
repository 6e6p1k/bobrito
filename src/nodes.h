#pragma once
#include <memory>

#include "shape_data.h"
#include "primitives.h"


struct Node
{
    // Primitive prim;
    std::string label = "";

    ShapePtr h_meta;    
    ContentPtr d_data;  
    bool transposed = false;

    std::vector<Node> outputs;
    std::vector<Node> inputs;
    int num_inputs;
    int num_outputs;

    std::shared_ptr<Node> grad_buffer;
    std::shared_ptr<Node> owned_grad;
    std::weak_ptr<Node> grad_node;
    bool requires_grad;

    bool is_h_meta();
    float* get_data();
    int get_size();
    int get_ndim();
    int* get_shape();
    int* get_strides();
    bool is_data_allocated();
    void set_label(char*);
    ~Node() = default;
};

std::shared_ptr<Node> make_node(Primitive* prim, std::initializer_list<std::shared_ptr<Node>> inputs, int* shape, int ndim);
