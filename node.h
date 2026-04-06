#pragma once
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <vector>
#include <string>

#include "allocator.cuh"

enum class OpType {null, Add, Mul, MatMul, Sub, Div, SumLD, MaxLD, MinLD, ArgMaxLD, ReLU, BwdReLU, Skip, Contiguous, Reshape, ExpandLD, BwdStart, BwdHelper};
std::ostream& operator<<(std::ostream& os, const OpType& op);
std::string opToString(const OpType& op);

enum class NodeKind { Source, Internal };

// ShapeData
struct ShapeData
{
    int ndim;
    int size;
    int data[0];

    static ShapeData* init(int ndim);
    int* shape();
    int* strides();
    void print();
};

using ShapePtr = std::shared_ptr<ShapeData>;
ShapePtr make_shape_data(int ndim, int* shape, int* strides = nullptr);

using ContentPtr = std::shared_ptr<ContentData>;


// Node
struct Node
{
    NodeKind kind;
    ShapePtr h_meta;       // shape, strides, ndim on host (shareable between fwd/bwd nodes)
    ContentPtr d_data;     // forward values on device
    std::shared_ptr<Node> grad_node;     // gradient on device
    bool transposed = false;
    int param_count = 0;
    std::string label; // just to test, remove later
    std::weak_ptr<Node> outputs[4]; // for grad

    float* get_data();
    bool is_data_allocated();
    virtual ~Node() = default;
};

struct SourceNode : Node
{
    bool h_requiers_grad = false;
    SourceNode() = default;
    SourceNode(float* data, int ndim, int* shape, int* strides = nullptr, bool requires_grad = false);
};

struct InternalNode : Node
{
    OpType op;  // operation that was used to create this node
    std::shared_ptr<Node> inputs[4];
    int num_inputs;
};

// <nodes interaction>

//      <general operations>
std::shared_ptr<Node> elemwise(std::shared_ptr<Node> a, std::shared_ptr<Node> b, OpType op, std::shared_ptr<Node> dst = nullptr);
std::shared_ptr<Node> reduction(std::shared_ptr<Node> a, int dim, OpType op, bool keepdims, std::shared_ptr<Node> dst = nullptr);
//      </general operations>

std::shared_ptr<Node> add(std::shared_ptr<Node> a, std::shared_ptr<Node> b);
std::shared_ptr<Node> mul(std::shared_ptr<Node> a, std::shared_ptr<Node> b);
std::shared_ptr<Node> max(std::shared_ptr<Node> a, int dim = -1, bool keepdims = false);
std::shared_ptr<Node> min(std::shared_ptr<Node> a, int dim = -1, bool keepdims = false);
std::shared_ptr<Node> argmax(std::shared_ptr<Node> a, int dim = -1, bool keepdims = false);
std::shared_ptr<Node> matmul(std::shared_ptr<Node> a, std::shared_ptr<Node> b, std::shared_ptr<Node> dst = nullptr);
std::shared_ptr<Node> skip(std::shared_ptr<Node> a);
std::shared_ptr<Node> copy(std::shared_ptr<Node> a, std::shared_ptr<Node> dst);
std::shared_ptr<Node> contiguous(std::shared_ptr<Node> a, std::shared_ptr<Node> dst = nullptr);
std::shared_ptr<Node> reshape(std::shared_ptr<Node> a, int* res_shape, int res_ndim);
std::shared_ptr<Node> sum_last_dim(std::shared_ptr<Node> a, std::shared_ptr<Node> dst = nullptr);
std::shared_ptr<Node> sum(std::shared_ptr<Node> a, int dim, bool keepdims = false);
std::shared_ptr<Node> expand_last_dim(std::shared_ptr<Node> a, int* desired_shape, int ndim, std::shared_ptr<Node> dst = nullptr);
std::shared_ptr<Node> reduce_to_shape(std::shared_ptr<Node> node, int* target_shape, int target_ndim);
std::shared_ptr<Node> relu(std::shared_ptr<Node> a);

// </nodes interaction>

// <Topological sort>
void _build_topo(Node* node, std::unordered_set<Node*>& visited, std::vector<InternalNode*>& inodes, std::vector<SourceNode*>& snodes);
void build_topo(Node* node, std::unordered_set<Node*>& visited, std::vector<Node*>& order);
void build_topo_grad(InternalNode* node, std::unordered_set<InternalNode*>& visited, std::vector<InternalNode*>& order);
std::vector<Node*> topo_sort(std::shared_ptr<InternalNode> node, bool grad);
std::vector<Node*> topo_sort(std::shared_ptr<Node> node, bool grad);
// </Topological sort>

void op_based_eval(InternalNode* inode);
void eval(std::shared_ptr<InternalNode> root);
void eval(std::shared_ptr<Node> root);
void geval(std::shared_ptr<InternalNode> root);
void geval(std::shared_ptr<Node> root);
void _geval(std::shared_ptr<Node> root);