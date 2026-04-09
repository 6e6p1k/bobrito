#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <unordered_set>
#include <vector>
#include <cstring>

#include "utils.h"
#include "allocator.cuh"
#include "ops.cuh"
#include "node_helpers.h"
#include "node.h"

using namespace std;

// <helper for OpType>
ostream& operator<<(std::ostream& os, const OpType& op) 
{
    switch (op) {
        case OpType::Add:               return os << "+";
        case OpType::Sub:               return os << "-";
        case OpType::Mul:               return os << "*";
        case OpType::Div:               return os << "/";
        case OpType::SumLD:             return os << "∑";
        case OpType::MinLD:             return os << ".min()";
        case OpType::MaxLD:             return os << ".max()";
        case OpType::MatMul:            return os << "@";
        case OpType::Skip:              return os << ".skip()";
        case OpType::Contiguous:        return os << "~";
        case OpType::Reshape:           return os << "RR";
        case OpType::ExpandLD:          return os << "[ ]";
        case OpType::ReLU:              return os << "relu";
        case OpType::BwdReLU:           return os << "bwd_relu";
        default:                        return os << "Unknown";
    }
}

string opToString(const OpType& op)
{
    switch (op) {
        case OpType::Add:           return "+";
        case OpType::Sub:           return "-";
        case OpType::Mul:           return "*";
        case OpType::Div:           return "/";
        case OpType::SumLD:         return ".sum()";
        case OpType::MinLD:         return ".min()";
        case OpType::MaxLD:         return ".maxld()";
        case OpType::MatMul:        return "@";
        case OpType::Skip:          return ".skip()";
        case OpType::Contiguous:    return ".contiguous()";
        case OpType::Reshape:       return ".reshape()";
        case OpType::ExpandLD:      return ".expandld()";
        case OpType::ReLU:          return ".relu()";
        case OpType::BwdReLU:       return ".bwd_relu()";
        default:                    return "Unknown";
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
    sd->size = utils::numel(ndim, shape);
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

float* Node::get_data() { return d_data->data(); }
bool Node::is_data_allocated() { return d_data->is_allocated(); }

SourceNode::SourceNode(float* data, int ndim, int* shape, int* strides, bool requires_grad)
{
    kind = NodeKind::Source;
    // assign general parameters
    h_requiers_grad = requires_grad;
    
    // transfer array content to device
    d_data = std::make_shared<ContentData>();
    d_data->h_to_d(data, utils::numel(ndim, shape));
    
    // store metadata
    h_meta = make_shape_data(ndim, shape, strides);
}


// </Node>
// <nodes interaction>
//      <general operations>
std::shared_ptr<Node> elemwise(std::shared_ptr<Node> a, std::shared_ptr<Node> b, OpType op, std::shared_ptr<Node> dst)
{
    auto [res_shape, res_ndim] = check_shape_return_out_shape::elemwise(
        a, b
    );
    std::shared_ptr<Node> inputs[] = {a, b};
    auto node = make_internal_node({
        .op = op,
        .shape = res_shape,
        .ndim = res_ndim,
        .inputs = inputs,
        .num_inputs = 2,
        .dst = dst
    });
    // <remove later>
    node->label = "( " + a->label + " " + opToString(op) + " " + b->label + " )";
    // </remove later>
    return node;
}

std::shared_ptr<Node> reduce_last_dim(std::shared_ptr<Node> a, OpType op)
{
    int res_ndim = a->h_meta->ndim - 1;
    int res_shape[res_ndim];
    for (int i = 0; i < res_ndim; ++i) res_shape[i] = a->h_meta->shape()[i];
    std::shared_ptr<Node> inputs[] = {a};
    auto node = make_internal_node({
        .op = op,
        .shape = res_shape,
        .ndim = res_ndim,
        .inputs = inputs,
        .num_inputs = 1,
    });
    node->label = "( sum_last_dim" + a->label + " )";
    return node;
}

std::shared_ptr<Node> reduce(std::shared_ptr<Node> a, int dim, bool keepdims, OpType op)
{
    // handle negative indexing
    int ndim = a->h_meta->ndim;
    dim = (dim + ndim) % ndim;
    if (dim > ndim - 1) throw logic_error("Dim wasn't found.");

    if (dim == ndim - 1 && !keepdims) return reduce_last_dim(a, op);

    // 1. reshape, put the dim to the end
    int* shape_copy = new int[ndim];
    int val = a->h_meta->shape()[dim];
    int j = 0;
    for (int i = 0; i < ndim; ++i)
        if (i != dim) shape_copy[j++] = a->h_meta->shape()[i];
    shape_copy[ndim-1] = val;

    std::shared_ptr<Node> reshaped_node = reshape(a, shape_copy, ndim); 

    // 2. make contiguous
    std::shared_ptr<Node> contiguous_node = contiguous(reshaped_node);

    // 3. perform last dim reduction
    std::shared_ptr<Node> node_after_sum = reduce_last_dim(contiguous_node, op);

    // 4. if keepdims = True, insert another dim
    if (keepdims)
    {
        int* res_shape = new int[ndim];
        for (int i = 0; i < ndim; ++i)
            res_shape[i] = (i == dim) ? 1 : a->h_meta->shape()[i];
        std::shared_ptr<Node> node = reshape(node_after_sum, res_shape, ndim);
        return node;
    }
    return node_after_sum;
}
//      </general operations>

//      <ops>
std::shared_ptr<Node> add(const std::shared_ptr<Node> a, const std::shared_ptr<Node> b) 
{ 
    return elemwise(a, b, OpType::Add); 
}
std::shared_ptr<Node> mul(const std::shared_ptr<Node> a, const std::shared_ptr<Node> b) 
{ 
    return elemwise(a, b, OpType::Mul);
}
std::shared_ptr<Node> sub(const std::shared_ptr<Node> a, const std::shared_ptr<Node> b)
{
    return elemwise(a, b, OpType::Sub);
}
std::shared_ptr<Node> div(const std::shared_ptr<Node> a, const std::shared_ptr<Node> b)
{
    return elemwise(a, b, OpType::Div);
}

std::shared_ptr<Node> matmul(std::shared_ptr<Node> a, std::shared_ptr<Node> b, std::shared_ptr<Node> dst)
{
    auto [res_shape, res_ndim] = check_shape_return_out_shape::matmul(a, b);
    std::shared_ptr<Node> inputs[] = {a, b};
    auto node = make_internal_node({
        .op = OpType::MatMul,
        .shape = res_shape,
        .ndim = res_ndim,
        .inputs = inputs,
        .num_inputs = 2,
        .dst = dst
    });
    node->label = "(" + a->label + " " + opToString(OpType::MatMul) + " " + b->label + ")"; // remove later
    return node;
}
std::shared_ptr<Node> skip(std::shared_ptr<Node> a)
{
    InodeConfig cfg;
    cfg.op = OpType::Skip;
    cfg.src = a;
    cfg.num_inputs = 1;
    std::shared_ptr<Node> inputs[] = {a};
    cfg.inputs = inputs;
    cfg.copy_data = true;

    shared_ptr<Node> node = copy_to_internal_node(cfg);
    return node;
}
std::shared_ptr<Node> copy(std::shared_ptr<Node> a, std::shared_ptr<Node> dst)
{
    std::shared_ptr<Node> inputs[] = {a};
    auto node = make_internal_node({
        .op = OpType::Skip,
        .shape = a->h_meta->shape(),
        .ndim = a->h_meta->ndim,
        .inputs = inputs,
        .num_inputs = 1,
        .dst = dst
    });
    return node;
}
std::shared_ptr<Node> contiguous(std::shared_ptr<Node> a, std::shared_ptr<Node> dst)
{
    std::shared_ptr<Node> inputs[] = {a};
    int* res_strides = new int[a->h_meta->ndim];
    utils::compute_strides(res_strides, a->h_meta->ndim, a->h_meta->shape());
    auto node = make_internal_node({
        .op = OpType::Contiguous,
        .shape = a->h_meta->shape(),
        .strides = res_strides,
        .ndim = a->h_meta->ndim,
        .inputs = inputs,
        .num_inputs = 1,
        .dst = dst
    });
    node->label = "( contiguous" + a->label + " )";
    return node;
}
std::shared_ptr<Node> reshape(std::shared_ptr<Node> a, int* res_shape, int res_ndim)
{
    // quick check
    if (utils::numel(a->h_meta->ndim, a->h_meta->shape()) != utils::numel(res_ndim, res_shape))
        throw std::logic_error("Cannot broadcast this.");

    // compute new strides
    int* res_strides = new int[res_ndim];
    utils::compute_strides(res_strides, res_ndim, res_shape);
    
    // make output node 
    std::shared_ptr<Node> inputs[] = {a};
    std::shared_ptr<InternalNode> node = std::make_shared<InternalNode>();
    node->kind = NodeKind::Internal;
    node->op = OpType::Reshape;
    node->num_inputs = 1;
    node->h_meta = make_shape_data(res_ndim, res_shape, res_strides);
    node->d_data = a->d_data; // data isn't changed (reshape is a view)
    node->grad_node = a->grad_node;
    node->label = "( reshape" + a->label + " )";

    // connect inputs to outputs and vice versa
    for (int i = 0; i < 1; i++)
    {
        node->inputs[i] = inputs[i];
        node->param_count += inputs[i]->param_count;
        if (inputs[i]->outputs[0].lock() == nullptr) inputs[i]->outputs[0] = node;
        else inputs[i]->outputs[1] = node;
    }
    return node;
}

std::shared_ptr<Node> sum(std::shared_ptr<Node> a, int dim, bool keepdims)
{
    return reduce(a, dim, keepdims, OpType::SumLD);
}

std::shared_ptr<Node> min(std::shared_ptr<Node> a, int dim, bool keepdims)
{
    return reduce(a, dim, keepdims, OpType::MinLD);
}

std::shared_ptr<Node> max(std::shared_ptr<Node> a, int dim, bool keepdims)
{
    return reduce(a, dim, keepdims, OpType::MaxLD);
}

std::shared_ptr<Node> argmax(std::shared_ptr<Node> a, int dim, bool keepdims)
{
    return reduce(a, dim, keepdims, OpType::ArgMaxLD);
}

std::shared_ptr<Node> expand_last_dim(std::shared_ptr<Node> a, int* res_shape, int res_ndim, std::shared_ptr<Node> dst)
{
    // quick check
    bool same_ndim_case = (a->h_meta->ndim == res_ndim) && (a->h_meta->shape()[res_ndim - 1] == 1);
    bool different_ndim_case = (a->h_meta->ndim == res_ndim - 1);
    if (!(same_ndim_case || different_ndim_case)) throw std::logic_error("Cannot broadcast this.");
    for (int i = 0; i < res_ndim - 1; ++i)
        if (a->h_meta->shape()[i] != res_shape[i])
            throw std::logic_error("Cannot broadcast this.");

    std::shared_ptr<Node> inputs[] = {a};
    auto node = make_internal_node({
        .op = OpType::ExpandLD,
        .shape = res_shape,
        .ndim = res_ndim,
        .inputs = inputs,
        .num_inputs = 1,
        .dst = dst
    });
    node->label = "( expand_last_dim" + a->label + " )";
    return node;
}

std::shared_ptr<Node> reduce_to_shape(std::shared_ptr<Node> node, int* target_shape, int target_ndim)
{
    int* src_shape = node->h_meta->shape();
    int src_ndim = node->h_meta->ndim; 
    
    if (src_ndim == target_ndim) {
        bool match = true;
        for (int i = 0; i < src_ndim; i++)
            if (src_shape[i] != target_shape[i]) 
            {
                match = false; 
                break; 
            }
        if (match) return node;
    }

    // left-pad target with 1s
    int pad = src_ndim - target_ndim;
    
    // reduce dims from left (extra leading dims)
    // then reduce any internal broadcast dims
    auto current = node;
    
    // first collapse extra leading dims
    for (int i = 0; i < pad; i++) {
        current = sum(current, 0, false);
    }
    
    // now same rank — reduce any remaining broadcast dims (keepdims)
    for (int i = 0; i < target_ndim; i++) {
        if (target_shape[i] == 1 && current->h_meta->shape()[i] != 1) {
            current = sum(current, i, true);
        }
    }
    return current;
}

std::shared_ptr<Node> relu(std::shared_ptr<Node> a)
{
    std::shared_ptr<Node> inputs[] = {a};

    auto node = make_internal_node({
        .op = OpType::ReLU,
        .shape = a->h_meta->shape(),
        .ndim = a->h_meta->ndim,
        .inputs = inputs,
        .num_inputs = 1,
    });
    node->label = "( relu" + a->label + " )";
    return node;
}

std::shared_ptr<Node> log(std::shared_ptr<Node> a)
{
    std::shared_ptr<Node> inputs[] = {a};

    auto node = make_internal_node({
        .op = OpType::Log,
        .shape = a->h_meta->shape(),
        .ndim = a->h_meta->ndim,
        .inputs = inputs,
        .num_inputs = 1,
    });
    node->label = "( log" + a->label + " )";
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

void build_topo_grad(Node* node, unordered_set<Node*>& visited, vector<Node*>& order) {
    if (visited.count(node)) return;
    visited.insert(node);

    if (node->kind == NodeKind::Internal) {
        InternalNode* inode = static_cast<InternalNode*>(node);
        for (int i = 0; i < 4; i++) {
            auto out = inode->outputs[i].lock();
            if (!out) continue;
            build_topo_grad(out.get(), visited, order);
        }
    }
    order.push_back(node);
}

vector<Node*> topo_sort(shared_ptr<InternalNode> node, bool grad)
{
    unordered_set<Node*> visited;
    vector<Node*> order;
    if (grad) {
        build_topo_grad(node.get(), visited, order);
    } else {
        build_topo(node.get(), visited, order);
    }
    return order;
}

vector<Node*> topo_sort(shared_ptr<Node> node, bool grad)
{
    unordered_set<Node*> visited;
    vector<Node*> order;
    if (grad) {
        build_topo_grad(node.get(), visited, order);
    } else {
        build_topo(node.get(), visited, order);
    }
    return order;
}

void eval(std::shared_ptr<Node> root)
{
    // perform topological sort
    vector<Node*> order = topo_sort(root, false);

    // cout<<"EVAL LOOP ---------------------------\n";
    // walk in order
    for (Node* node : order)
    {
        // cout<<"Current node: "<<node->label<<endl;
        if (node->kind == NodeKind::Source) continue;

        InternalNode* inode = static_cast<InternalNode*>(node);

        // allocate memory
        int size = inode->h_meta->size;
        if (!inode->d_data) inode->d_data = make_shared<ContentData>();
        inode->d_data->allocate(size);

        // cout<<"Node: "<<inode->label<<endl;
        // cout<<"data allocated: "<<inode->d_data->is_allocated()<<endl;
        
        // dispatch
        op_based_eval(inode);
    }
}

void clear_grads(std::shared_ptr<Node> root)
{
    vector<Node*> order = topo_sort(root, false);
    // walk in order
    for (Node* node : order) 
    {
        node->grad_node.reset();
        if (node->kind == NodeKind::Source) static_cast<SourceNode*>(node)->owned_grad.reset();
    }
}

void _geval(std::shared_ptr<Node> root)
{
    // perform topological sort
    vector<Node*> order = topo_sort(root, true);
    // fill the root node grad with beer
    // prepare_starting_grad(root);

    // cout<<"GEVAL LOOP: ---------------------------------"<<endl;
    // walk in order
    for (int i = order.size() - 1; i >= 0; --i)
    {
        if (order[i]->kind == NodeKind::Source) continue;

        InternalNode* inode = static_cast<InternalNode*>(order[i]);

        // allocate grad
        int size = inode->h_meta->size;
        if (!(inode->d_data)) 
        {
            // cout<<"Not allocated\n";
            inode->d_data = make_shared<ContentData>();
        } 
        // else if (inode->d_data) cout<<"ContentData is already specified.";
        inode->d_data->allocate(size);

        
        // dispatch
        // cout<<"------------------------------------------------"<<endl;
        // cout<<"node: "<<inode->label<<endl;
        // cout << "Type is: " << typeid(order[i]).name() << std::endl;
        // cout<<"is_data: "<<(inode->d_data && inode->d_data->is_allocated())<<endl;
        // cout<<"size: "<<(inode->h_meta->size)<<endl;
        op_based_eval(inode);
        // print_node(inode);

    }
}

void op_based_eval(InternalNode* inode)
{
    switch (inode->op)
    {
        case OpType::Add:           cu_ops::add(inode);                 return;
        case OpType::Sub:           cu_ops::sub(inode);                 return;
        case OpType::Mul:           cu_ops::mul(inode);                 return;
        case OpType::Div:           cu_ops::div(inode);                 return;
        case OpType::MatMul:        cu_ops::matmul(inode);              return;
        case OpType::Contiguous:    cu_ops::contiguous(inode);          return;
        case OpType::SumLD:         cu_ops::sum_last_dim(inode);        return;
        case OpType::MinLD:         cu_ops::min_last_dim(inode);        return;
        case OpType::ExpandLD:      cu_ops::expand_last_dim(inode);     return;
        case OpType::BwdStart:      cu_ops::fill(inode, 1.0f);          return;
        case OpType::ReLU:          cu_ops::relu(inode);                return;
        case OpType::BwdReLU:       cu_ops::bwd_relu(inode);            return;
        case OpType::Log:           cu_ops::log(inode);                 return;
        default:                    return;
    }
}