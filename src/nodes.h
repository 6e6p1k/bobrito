#pragma once
#include <string.h>

#include <cstdlib>
#include <memory>
#include <vector>

#include "allocator.cuh"
#include "shape_data.h"

#define DEBUG(__node__, space_name)                                            \
    cout << "\n=============================" << endl;                         \
    cout << "[" << space_name << "]" << endl;                                  \
    if (!(__node__)) {                                                         \
        cout << "NODE IS NULLPTR" << endl;                                     \
        cout << "=============================\n" << endl;                     \
    } else {                                                                   \
        cout << "LABEL ========> " << (__node__)->label << endl;               \
        cout << "PRIMITIVE ====> "                                             \
             << ((__node__)->prim ? (__node__)->prim->name() : "<no prim>")    \
             << endl;                                                          \
        if ((__node__)->is_meta()) {                                           \
            cout << "NDIM ==========> " << (__node__)->get_ndim() << endl;     \
            cout << "SHAPE =========> [";                                      \
            for (int _i = 0; _i < (__node__)->get_ndim(); _i++)                \
                cout << (__node__)->get_shape()[_i]                            \
                     << (_i < (__node__)->get_ndim() - 1 ? ", " : "");         \
            cout << "]" << endl;                                               \
        } else {                                                               \
            cout << "NDIM ==========> no meta" << endl;                        \
            cout << "SHAPE =========> no meta" << endl;                        \
        }                                                                      \
        cout << ((__node__)->is_data() ? "DATA ===========> exists."           \
                                       : "DATA ===========> doesnt exist.")    \
             << endl;                                                          \
        cout << ((__node__)->is_data_allocated()                               \
                     ? "DATA ===========> is allocated."                       \
                     : "DATA ===========> is not allocated.")                  \
             << endl;                                                          \
        {                                                                      \
            auto _gn = (__node__)->owned_grad;                                 \
            cout << (_gn ? "GRAD ===========> exists."                         \
                         : "GRAD ===========> doesnt exist.")                  \
                 << endl;                                                      \
            if (_gn)                                                           \
                cout << (_gn->is_data_allocated()                              \
                             ? "GRAD ===========> is allocated."               \
                             : "GRAD ===========> is not allocated.")          \
                     << endl;                                                  \
        }                                                                      \
        cout << "INPUTS (" << (__node__)->inputs.size() << "):" << endl;       \
        for (auto& _in : (__node__)->inputs) {                                 \
            if (!_in)                                                          \
                cout << "  <nullptr input>" << endl;                           \
            else                                                               \
                cout << "  [" << (_in->prim ? _in->prim->name() : "<no prim>") \
                     << "] "                                                   \
                     << (_in->label.empty() ? "<no label>" : _in->label)       \
                     << endl;                                                  \
        }                                                                      \
        cout << "OUTPUTS (" << (__node__)->outputs.size() << "):" << endl;     \
        for (auto& _out : (__node__)->outputs) {                               \
            auto _sp = _out.lock();                                            \
            if (!_sp)                                                          \
                cout << "  [expired]" << endl;                                 \
            else                                                               \
                cout << "  [" << (_sp->prim ? _sp->prim->name() : "<no prim>") \
                     << "] "                                                   \
                     << (_sp->label.empty() ? "<no label>" : _sp->label)       \
                     << endl;                                                  \
        }                                                                      \
        cout << "=============================\n" << endl;                     \
    }

struct Primitive;

using ShapePtr = std::shared_ptr<ShapeData>;
using ContentPtr = std::shared_ptr<ContentData>;

struct Node : public std::enable_shared_from_this<Node>{
    Primitive* prim;
    std::string label = "";

    ShapePtr h_meta;
    ContentPtr d_data;
    bool transposed = false;

    std::vector<std::weak_ptr<Node>> outputs;
    std::vector<std::shared_ptr<Node>> inputs;

    std::shared_ptr<Node> owned_grad;
    bool requires_grad = false;
    bool is_bwd = false;

    std::shared_ptr<Node> getPtr() 
    { return shared_from_this(); }

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
    void export_graph(std::string filename = "graph");
    ~Node() = default;
};

struct ScalarNode : Node {
    float value;
};

std::shared_ptr<Node> make_node(Primitive* prim,
                                std::vector<std::shared_ptr<Node>> inputs,
                                int* shape, int ndim);
std::shared_ptr<Node> copy_node_no_prim_before_eval(std::shared_ptr<Node> node);
std::shared_ptr<ScalarNode> make_scalar_node(float value, std::string label = "");
std::shared_ptr<Node> copy_node_no_prim_with_link(std::shared_ptr<Node> node);
std::string make_node_string(std::shared_ptr<Node> node);
std::string make_node_string(Node* node);
void print_node(std::shared_ptr<Node> node);

struct Primitive {
    virtual void eval(Node* node) = 0;
    virtual void vjp(std::shared_ptr<Node> fwd, std::shared_ptr<Node> dC,
                     std::shared_ptr<Node> out[]) {}
    virtual void allocate(Node* node) {
        if (!node->is_data()) node->d_data = std::make_shared<ContentData>();
        if (!node->is_data_allocated()) node->d_data->allocate(node->get_size());
    }
    virtual const std::string name() = 0;
    virtual const std::string symbol() = 0;
    virtual ~Primitive() = default;
};

static struct : Primitive {
    void eval(Node* node) override {}
    void vjp(std::shared_ptr<Node> fwd, std::shared_ptr<Node> dC,
             std::shared_ptr<Node> out[]) override {}
    const std::string name() override { return "Leaf"; }
    const std::string symbol() override { return "leaf"; }
} prim_leaf;




