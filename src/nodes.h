#pragma once
#include <string.h>

#include <memory>
#include <vector>

#include "allocator.cuh"
#include "shape_data.h"

#define DEBUG(__node__, space_name)                                 \
    cout << "\n=============================" << endl;              \
    cout << "[" << space_name << "]" << endl;                       \
    cout << "LABEL ========> " << __node__->label << endl;          \
    cout << "PRIMITIVE ====> " << __node__->prim->name() << endl;   \
    cout << (__node__->is_data() ? "ContentData ==> exists."        \
                                 : "ContentData ==> doesnt exist.") \
         << endl;                                                   \
    cout << (__node__->is_data_allocated()                          \
                 ? "ContentData ==> is allocated."                  \
                 : "ContentData ==> is not allocated.")             \
         << endl;                                                   \
    cout << "=============================\n" << endl;

struct Node;

struct Primitive {
    virtual void eval(Node* node) = 0;
    virtual void vjp(std::shared_ptr<Node> fwd, std::shared_ptr<Node> dC,
                     std::shared_ptr<Node> out[]) {}
    virtual const std::string name() = 0;
    virtual const std::string symbol() = 0;
    virtual ~Primitive() = default;
};

using ShapePtr = std::shared_ptr<ShapeData>;
using ContentPtr = std::shared_ptr<ContentData>;

struct Node {
    Primitive* prim;
    std::string label = "";

    ShapePtr h_meta;
    ContentPtr d_data;
    bool transposed = false;

    std::vector<std::shared_ptr<Node>> outputs;
    std::vector<std::shared_ptr<Node>> inputs;
    int num_inputs;
    int num_outputs;

    std::shared_ptr<Node> grad_buffer;
    std::shared_ptr<Node> owned_grad;
    std::weak_ptr<Node> grad_node;
    bool requires_grad;

    bool is_label();
    bool is_meta();
    bool is_data();
    bool is_data_allocated();

    float* get_data();
    int get_size();
    int get_ndim();
    int* get_shape();
    int* get_strides();

    void set_label(std::string);
    ~Node() = default;
};

struct ScalarNode : Node {
    float value;
};

std::shared_ptr<Node> make_node(Primitive* prim,
                                std::vector<std::shared_ptr<Node>> inputs,
                                int* shape, int ndim);
std::shared_ptr<Node> copy_node_no_prim(std::shared_ptr<Node> node);
std::shared_ptr<ScalarNode> make_scalar_node(float value,
                                             std::string label = "");
void print_node(std::shared_ptr<Node> node);
