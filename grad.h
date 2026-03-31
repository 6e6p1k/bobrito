#pragma once
#include "allocator.cuh"
#include <cstddef>
#include <memory>
#include <vector>
#include <unordered_set>


enum class OpType { Add, Mul, MatMul, Sum, Max, Min, ReLU, Skip};
// <helper for OpType>
ostream& operator<<(std::ostream& os, const OpType& op);
string opToString(const OpType& op);
// </helper for OpType>

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

using ShapePtr = std::shared_ptr<ShapeData>;
ShapePtr make_shape_data(int ndim, int* shape, int* strides = nullptr);

using ContentPtr = std::shared_ptr<ContentData>;


// Node
struct Node
{
    NodeKind kind;
    ShapePtr h_meta;       // shape, strides, ndim on host (shareable between fwd/bwd nodes)
    ContentPtr d_content;    // values + grad on device (shareable between fwd/bwd nodes)
    int use_count = 0;
    string label; // just to test, remove later

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
    std::shared_ptr<Node> outputs[4]; // for grad
    int num_inputs;
};

std::shared_ptr<Node> make_internal_node(OpType op, std::shared_ptr<Node> inputs[], int num_inputs, int* shape, int ndim, int* strides = nullptr, int use_count = 0);
std::shared_ptr<InternalNode> make_internal_node_like(InternalNode* example);
std::shared_ptr<SourceNode> make_source_node_like(SourceNode* example);


// <nodes interaction>

//      <general operations>
std::shared_ptr<Node> elemwise(std::shared_ptr<Node> a, std::shared_ptr<Node> b, OpType op);
std::shared_ptr<Node> reduction(std::shared_ptr<Node> a, int dim, OpType op, bool keepdims);
//      </general operations>

std::shared_ptr<Node> add(std::shared_ptr<Node> a, std::shared_ptr<Node> b);
std::shared_ptr<Node> mul(std::shared_ptr<Node> a, std::shared_ptr<Node> b);
std::shared_ptr<Node> sum(std::shared_ptr<Node> a, int dim, bool keepdims = false);
std::shared_ptr<Node> max(std::shared_ptr<Node> a, int dim, bool keepdims = false);
std::shared_ptr<Node> min(std::shared_ptr<Node> a, int dim, bool keepdims = false);


// </nodes interaction>

namespace backward_node_ops 
{
void add(std::shared_ptr<InternalNode> self, std::shared_ptr<InternalNode> mirror);
}


// <Topological sort>
void build_topo(Node* node, unordered_set<Node*>& visited, vector<InternalNode*>& inodes, vector<SourceNode*>& snodes);
// </Topological sort>

void backward_pass(shared_ptr<Node> node);
