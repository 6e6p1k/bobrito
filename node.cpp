#include <atomic>
#include <cstddef>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "utils.h"
#include "allocator.cuh"
#include "ops.cuh"
#include "node.h"

// <helper for OpType>
ostream& operator<<(std::ostream& os, const OpType& op) 
{
    switch (op) {
        case OpType::Add:       return os << "+";
        case OpType::Mul:       return os << "*";
        case OpType::Sum:       return os << "∑";
        case OpType::Min:       return os << ".min()";
        case OpType::Max:       return os << ".max()";
        case OpType::MatMul:    return os << "@";
        case OpType::Skip:      return os << ".skip()";
        default:                return os << "Unknown";
    }
}

string opToString(const OpType& op)
{
    switch (op) {
        case OpType::Add:       return "+";
        case OpType::Mul:       return "*";
        case OpType::Sum:       return ".sum()";
        case OpType::Min:       return ".min()";
        case OpType::Max:       return ".max()";
        case OpType::MatMul:    return "@";
        case OpType::Skip:      return ".skip()";
        default:                return "Unknown";
    }
}
// </helper for OpType>


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
// </ShapeData>


// <ShapeData helper functions>
ShapePtr make_shape_data(int ndim, int* shape, int* strides)
{
    ShapeData* sd = ShapeData::init(ndim);
    // case 1: strides are not specified
    if (strides == nullptr)
    {
        utils::compute_strides(sd->strides(), ndim, shape);
        for (int i = 0; i < ndim; i++) sd->shape()[i] = shape[i];
        return ShapePtr(sd, ::free);
    }
    // case 2: strides are specified
    for (int i = 0; i < ndim; i++)
    {
        sd->shape()[i] = shape[i];
        sd->strides()[i] = strides[i];
    }
    return ShapePtr(sd, ::free);
}
// </ShapeData helper functions>


// <Node>
//      SourceNode Initializer
SourceNode::SourceNode(float* data, int ndim, int* shape, int* strides, bool requires_grad)
{
    kind = NodeKind::Source;
    // assign general parameters
    h_requiers_grad = requires_grad;
    
    // transfer array content to device
    d_content = std::make_shared<ContentData>();
    d_content->ContentData::h_values_to_d(data, utils::numel(ndim, shape));
    
    // store metadata
    h_meta = make_shape_data(ndim, shape, strides);
}

//      Makes internal node,
//      mostly utilised by node operations
std::shared_ptr<InternalNode> make_internal_node(OpType op, std::shared_ptr<Node> inputs[], int num_inputs, int* shape, int ndim, int* strides, int param_count)
{
    std::shared_ptr<InternalNode> node = std::make_shared<InternalNode>();
    node->kind = NodeKind::Internal;
    node->op = op;
    node->num_inputs = num_inputs;
    node->h_meta = make_shape_data(ndim, shape, strides);
    node->d_content = make_shared<ContentData>();
    for (int i = 0; i < num_inputs; i++)
    {
        node->inputs[i] = inputs[i];
        node->param_count += inputs[i]->param_count;
        // cout<<"Node ["<<inputs[i]->label<<"] has "<<inputs[i]->param_count<<" labels"<<endl;
        if (inputs[i]->outputs[0] == nullptr) inputs[i]->outputs[0] = node;
        else inputs[i]->outputs[1] = node;
    }
    return node;
}

// 
std::shared_ptr<InternalNode> make_internal_node_like(InternalNode* example)
{
    auto node = std::make_shared<InternalNode>();
    node->kind = example->kind;
    node->op = example->op;
    node->num_inputs = example->num_inputs;
    node->h_meta = example->h_meta;
    node->d_content = example->d_content;
    node->param_count = example->param_count;
    return node;
}

std::shared_ptr<InternalNode> copy_to_internal_node(std::shared_ptr<Node> example, int param_count)
{
    if (example->kind == NodeKind::Internal)
    {
        SourceNode* snode = static_cast<SourceNode*>(&*example);
        if (snode->h_requiers_grad) param_count -= 1;
    }
    auto node = std::make_shared<InternalNode>();
    node->kind = NodeKind::Internal;
    node->h_meta = example->h_meta;
    node->d_content = example->d_content;
    node->param_count = param_count;
    node->label = example->label;
    return node;
}

std::shared_ptr<InternalNode> copy_to_internal_node(std::shared_ptr<Node> example)
{
    auto node = std::make_shared<InternalNode>();
    node->kind = NodeKind::Internal;
    node->h_meta = example->h_meta;
    node->d_content = example->d_content;
    node->label = example->label;
    return node;
}

std::shared_ptr<InternalNode> gcopy_to_internal_node(std::shared_ptr<Node> example)
{
    auto node = std::make_shared<InternalNode>();
    node->kind = NodeKind::Internal;
    node->h_meta = example->h_meta;
    node->d_content = example->d_content;
    node->label = example->label;
    return node;
}

std::shared_ptr<SourceNode> make_source_node_like(SourceNode* example)
{
    auto node = std::make_shared<SourceNode>();
    node->kind = example->kind;
    node->h_meta = example->h_meta;
    node->d_content = example->d_content;
    node->h_requiers_grad = example->h_requiers_grad;
    node->param_count = example->param_count;
    return node;
}

void print_internal_node(InternalNode* node)
{
    int* shape = node->h_meta->shape();
    float* h_data = node->d_content->d_values_to_h();
    int ndim = node->h_meta->ndim;
    
    // output shape
    cout<<"Shape: [";
    for (int i = 0; i < ndim; ++i) { if (i > 0) cout<<", "; cout<<shape[i]; }
    cout<<"]"<<endl;

    // output elements (truncated)
    int total = node->d_content->size();
    int limit = total > 20 ? 20 : total;

    cout<<"Op: "<<opToString(node->op)<<endl;
    cout<<"Data: [";
    for (int i = 0; i < limit; ++i)
    {
        if (i > 0) cout<<", ";
        cout<<h_data[i];
    }
    if (total > limit) cout<<", ... ("<<total<<" total)";
    cout<<"]"<<endl;

    delete[] h_data;
}
// </Node>


// <nodes interaction>
//      <general operations>
std::shared_ptr<Node> elemwise(std::shared_ptr<Node> a, std::shared_ptr<Node> b, OpType op)
{
    auto [res_shape, res_ndim] = check_shape_return_out_shape::elemwise(
        a, b
    );
    std::shared_ptr<Node> inputs[] = {a, b};
    auto node = make_internal_node(
        op, inputs, 2, res_shape, res_ndim
    );
    // <remove later>
    node->label = "( " + a->label + " " + opToString(op) + " " + b->label + " )"; 
    // </remove later>
    return node;
}

std::shared_ptr<Node> reduction(std::shared_ptr<Node> a, int dim, OpType op, bool keepdims)
{
    auto [res_shape, res_ndim] = check_shape_return_out_shape::reduction(
        a, dim, keepdims
    );
    std::shared_ptr<Node> inputs[] = {a};
    auto node = make_internal_node(
        op, inputs, 1, res_shape, res_ndim
    );
    node->label = a->label + opToString(op); // remove later
    return node;
}
//      </general operations>

//      <ops>
std::shared_ptr<Node> add(std::shared_ptr<Node> a, std::shared_ptr<Node> b) { return elemwise(a, b, OpType::Add); }
std::shared_ptr<Node> mul(const std::shared_ptr<Node> a, const std::shared_ptr<Node> b) { return elemwise(a, b, OpType::Mul); }
std::shared_ptr<Node> sum(std::shared_ptr<Node> a, int dim, bool keepdims) { return reduction(a, dim, OpType::Sum, keepdims); }
std::shared_ptr<Node> max(std::shared_ptr<Node> a, int dim, bool keepdims) { return reduction(a, dim, OpType::Max, keepdims); }
std::shared_ptr<Node> min(std::shared_ptr<Node> a, int dim, bool keepdims) { return reduction(a, dim, OpType::Min, keepdims); }
std::shared_ptr<Node> matmul(std::shared_ptr<Node> a, std::shared_ptr<Node> b)
{
    auto [res_shape, res_ndim] = check_shape_return_out_shape::matmul(a, b);
    std::shared_ptr<Node> inputs[] = {a, b};
    auto node = make_internal_node(
        OpType::MatMul, inputs, 2, res_shape, res_ndim
    );
    node->label = a->label + opToString(OpType::MatMul); // remove later
    return node;
}
std::shared_ptr<Node> skip(std::shared_ptr<Node> a)
{
    std::shared_ptr<Node> inputs[] = {a};
    auto node = make_internal_node(
        OpType::Skip, inputs, 1, a->h_meta->shape(), a->h_meta->ndim
    );
    return node;
}
//      </ops>
// </nodes interaction>


// <Topological sort>
void _build_topo(Node* node, unordered_set<Node*>& visited, vector<InternalNode*>& inodes, vector<SourceNode*>& snodes)
{
    if (visited.count(node)) return;
    visited.insert(node);

    if (node->kind == NodeKind::Internal) 
    {
        InternalNode* inode = static_cast<InternalNode*>(node);
        for (int i = 0; i < inode->num_inputs; i++)
            _build_topo(inode->inputs[i].get(), visited, inodes, snodes);
        inodes.push_back(inode);
    }
    else {
        snodes.push_back(static_cast<SourceNode*>(node));
    }
}

void build_topo(Node* node, unordered_set<Node*>& visited, vector<Node*>& order)
{
    if (visited.count(node)) return;
    visited.insert(node);

    if (node->kind == NodeKind::Internal) {
        InternalNode* inode = static_cast<InternalNode*>(node);
        for (int i = 0; i < inode->num_inputs; i++)
            build_topo(inode->inputs[i].get(), visited, order);
    }
    order.push_back(node);

}

void build_topo_grad(Node* node, unordered_set<Node*>& visited, vector<Node*>& order)
{
    if (visited.count(node)) return;
    visited.insert(node);

    if (node->kind == NodeKind::Internal) {
        InternalNode* inode = static_cast<InternalNode*>(node);
        // compute num_outputs
        int num_outputs = 0;
        while (num_outputs < 4 && inode->outputs[num_outputs]) num_outputs++;
        // same but with outputs
        for (int i = 0; i < num_outputs; i++)
            build_topo(inode->outputs[i].get(), visited, order);
    }
    order.push_back(node);
}

vector<Node*> topo_sort(shared_ptr<InternalNode> node, bool grad)
{
    unordered_set<Node*> visited;
    vector<Node*> order;
    if (grad) build_topo_grad(node.get(), visited, order);
    else build_topo(node.get(), visited, order);
    return order;
}
// </Topological sort>

void eval(std::shared_ptr<InternalNode> root)
{
    cout<<endl;
    cout<<"FROM 'node.cpp' FUNCTION: void evaluate"<<endl;
    // perform topological sort
    vector<Node*> order = topo_sort(root, false);
    cout<<"order.size() = "<<order.size()<<endl;

    
    // walk in order
    for (Node* node : order)
    {   
        cout<<"Current node: "<<node->label<<endl;
        if (node->kind == NodeKind::Source) continue;

        InternalNode* inode = static_cast<InternalNode*>(node);

        // allocate memory
        int size = utils::numel(inode->h_meta->ndim, inode->h_meta->shape());
        inode->d_content->allocate_values(size);
        if (inode->d_content->values()) cout<<"values are allocated"<<endl;
        
        // dispatch
        switch (inode->op)
        {
            case OpType::Add:
                cu_ops::add(inode);
                break;
            case OpType::Mul:
                cu_ops::mul(inode);
                break;
            default:
                break;
        }
    }
}

void geval(std::shared_ptr<InternalNode> root)
{
    cout<<endl;
    cout<<"FROM 'node.cpp' FUNCTION: void geval"<<endl;
    // perform topological sort
    vector<Node*> order = topo_sort(root, false);
    cout<<"order.size() = "<<order.size()<<endl;

    
    // walk in order
    for (Node* node : order)
    {   
        cout<<"Current node: "<<node->label<<endl;
        if (node->kind == NodeKind::Source) continue;

        InternalNode* inode = static_cast<InternalNode*>(node);

        // allocate memory
        int size = utils::numel(inode->h_meta->ndim, inode->h_meta->shape());
        inode->d_content->allocate_grad(size);
        if (inode->d_content->values()) cout<<"grad was allocated"<<endl;
        
        // dispatch
        switch (inode->op)
        {
            case OpType::Mul:
                cu_ops::gmul(inode);
                break;
            case OpType::Skip:
                cu_ops::skip(inode);
            default:
                break;
        }
    }
}