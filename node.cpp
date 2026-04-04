#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <endian.h>
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
#include "node.h"

using namespace std;

// <helper for OpType>
ostream& operator<<(std::ostream& os, const OpType& op) 
{
    switch (op) {
        case OpType::Add:               return os << "+";
        case OpType::Mul:               return os << "*";
        case OpType::Sum:               return os << "∑";
        case OpType::Min:               return os << ".min()";
        case OpType::Max:               return os << ".max()";
        case OpType::MatMul:            return os << "@";
        case OpType::Skip:              return os << ".skip()";
        case OpType::Contiguous:        return os << "~";
        case OpType::Reshape:           return os << "RR";
        case OpType::ExpandLD:          return os << "[ ]";
        default:                        return os << "Unknown";
    }
}

string opToString(const OpType& op)
{
    switch (op) {
        case OpType::Add:           return "+";
        case OpType::Mul:           return "*";
        case OpType::Sum:           return ".sum()";
        case OpType::Min:           return ".min()";
        case OpType::Max:           return ".maxld()";
        case OpType::MatMul:        return "@";
        case OpType::Skip:          return ".skip()";
        case OpType::Contiguous:    return ".contiguous()";
        case OpType::Reshape:       return ".reshape()";
        case OpType::ExpandLD:      return ".expandld()";
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
float* BwdNode::get_data() { return d_grad->data(); }

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

//      Makes internal node,
//      mostly utilised by node operations
std::shared_ptr<InternalNode> make_internal_node(OpType op, std::shared_ptr<Node> inputs[], int num_inputs, int* shape, int ndim, int* strides, int param_count)
{
    std::shared_ptr<InternalNode> node = std::make_shared<InternalNode>();
    node->kind = NodeKind::Internal;
    node->op = op;
    node->num_inputs = num_inputs;
    node->h_meta = make_shape_data(ndim, shape, strides);
    node->d_data = make_shared<ContentData>();
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

std::shared_ptr<InternalNode> make_internal_node(InodeConfig cfg)
{
    if (!(cfg.is_op() && cfg.is_meta() && cfg.is_linking()))
        throw logic_error("Not enough data specified.");

    std::shared_ptr<InternalNode> node = std::make_shared<InternalNode>();
    node->kind = NodeKind::Internal;
    node->op = cfg.op;
    node->num_inputs = cfg.num_inputs;
    node->h_meta = make_shape_data(cfg.ndim, cfg.shape, cfg.strides);
    for (int i = 0; i < cfg.num_inputs; i++)
    {
        node->inputs[i] = cfg.inputs[i];
        node->param_count += cfg.inputs[i]->param_count;
        if (cfg.inputs[i]->outputs[0] == nullptr) cfg.inputs[i]->outputs[0] = node;
        else cfg.inputs[i]->outputs[1] = node;
    }
    if (cfg.is_param_count()) node->param_count = cfg.param_count;
    if (cfg.label != "") node->label = cfg.label;
    if (!cfg.dst) node->d_data = make_shared<ContentData>();
    else
    {
        node->d_data = cfg.dst->d_data;
        node->d_grad = cfg.dst->d_grad;
        cfg.dst->h_meta = node->h_meta;
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
    node->d_data = example->d_data;
    node->d_grad = example->d_grad;
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
    node->d_data = example->d_data;
    node->d_grad = example->d_grad;
    node->param_count = param_count;
    node->label = example->label + "_copy";
    return node;
}

std::shared_ptr<InternalNode> gcopy_to_internal_node(std::shared_ptr<Node> example)
{
    auto node = std::make_shared<InternalNode>();
    node->kind = NodeKind::Internal;
    node->h_meta = example->h_meta;
    node->d_data = example->d_data;
    node->d_grad = example->d_grad;
    node->label = example->label + "_grad";
    node->param_count = example->param_count;
    return node;
}

std::shared_ptr<InternalNode> copy_to_internal_node(std::shared_ptr<Node> example)
{
    auto node = std::make_shared<InternalNode>();
    node->kind = NodeKind::Internal;
    node->h_meta = example->h_meta;
    node->d_data = example->d_data;
    node->d_grad = example->d_grad;
    node->label = example->label + "_copy";
    node->param_count = example->param_count;
    return node;
}

std::shared_ptr<InternalNode> copy_to_internal_node(std::shared_ptr<Node> example, bool flipped, bool transposed, OpType op)
{
    auto node = std::make_shared<InternalNode>();
    node->kind = NodeKind::Internal;
    node->h_meta = example->h_meta;
    node->d_data = example->d_data;
    node->d_grad = example->d_grad;
    node->label = example->label + "_copy";
    if (flipped) node->flipped = !node->flipped;
    node->transposed = transposed;
    node->param_count = example->param_count;
    node->op = op;
    return node;
}

std::shared_ptr<BwdNode> copy_to_bwd_node(InodeConfig cfg)
{
    std::shared_ptr<BwdNode> node = std::make_shared<BwdNode>();

    if (cfg.data_copying_required())
    {
        node->kind = cfg.src->kind;
        node->h_meta = cfg.src->h_meta;
        node->d_data = cfg.src->d_data;
        node->d_grad = cfg.src->d_grad;
        node->label = cfg.src->label;
        node->param_count = cfg.src->param_count;
        cout<<"DATA IS COPIED!"<<endl;
    }

    if (cfg.is_op())
        node->op = cfg.op;

    if (cfg.is_label())
        node->label = cfg.label; cout<<"LABEL IS SET TO: "<<cfg.label<<endl;


    if (cfg.is_meta())
        node->h_meta = make_shape_data(cfg.ndim, cfg.shape, cfg.strides);

    if (cfg.is_linking())
    {
        node->num_inputs = cfg.num_inputs;
        for (int i = 0; i < cfg.num_inputs; i++)
        {
            node->inputs[i] = cfg.inputs[i];
            node->param_count += cfg.inputs[i]->param_count;

            if (cfg.inputs[i]->outputs[0] == nullptr)
                cfg.inputs[i]->outputs[0] = node;
            else
                cfg.inputs[i]->outputs[1] = node;
        }
    }

    if (cfg.is_param_count())
        node->param_count = cfg.param_count;

    if (cfg.data_transfer_required())
    {
        if (cfg.src_d_data) node->d_data = cfg.src_d_data;
        if (cfg.src_d_grad) node->d_grad = cfg.src_d_grad;
    }

    return node;
}

std::shared_ptr<BwdNode> internal_to_bwd(shared_ptr<InternalNode> src)
{
    std::shared_ptr<BwdNode> node = std::make_shared<BwdNode>();
    node->op = src->op;
    node->label = src->label;
    node->kind = src->kind;
    node->h_meta = src->h_meta;
    node->d_data = src->d_data;
    node->d_grad = src->d_grad;
    node->param_count = src->param_count;
    
    node->num_inputs = src->num_inputs;
    for (int i = 0; i < src->num_inputs; i++)
    {
        node->inputs[i] = src->inputs[i];

        for (int j = 0; j < 4; ++j)
            if (node->inputs[i]->outputs[j] == src)
                node->inputs[i]->outputs[j] = node;
    }
    return node;
}

std::shared_ptr<SourceNode> make_source_node_like(SourceNode* example)
{
    auto node = std::make_shared<SourceNode>();
    node->kind = example->kind;
    node->h_meta = example->h_meta;
    node->d_data = example->d_data;
    node->d_grad = example->d_grad;
    node->h_requiers_grad = example->h_requiers_grad;
    node->param_count = example->param_count;
    return node;
}

void prepare_starting_grad(std::shared_ptr<Node> node)
{
    int seed_size = utils::numel(node->h_meta->ndim, node->h_meta->shape());
    if (!node->d_grad) node->d_grad = make_shared<ContentData>();
    node->d_grad->allocate(seed_size);
    bool _buff = node->flipped;
    node->flipped = true;
    cu_ops::fill(node.get(), 1.0f);
    node->flipped = false;
    node->flipped = _buff;
}

void prepare_starting_grad(std::shared_ptr<InternalNode> node)
{
    int seed_size = utils::numel(node->h_meta->ndim, node->h_meta->shape());
    if (!node->d_grad) node->d_grad = make_shared<ContentData>();
    node->d_grad->allocate(seed_size);
    bool _buff = node->flipped;
    node->flipped = true;
    cu_ops::fill(node.get(), 1.0f);
    node->flipped = false;
    node->flipped = _buff;
}

void print_node(Node* node)
{
    int* shape = node->h_meta->shape();
    int ndim = node->h_meta->ndim;

    // output shape
    cout<<"Shape: [";
    for (int i = 0; i < ndim; ++i) { if (i > 0) cout<<", "; cout<<shape[i]; }
    cout<<"]"<<endl;

    // handle values
    if (!node->d_data || !node->d_data->is_allocated()) { cout<<"No values allocated yet."<<endl; }
    else
    {
        int total = node->h_meta->size;
        int limit = total > 20 ? 20 : total;
        float* h_data = node->d_data->d_to_h();

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

    // handle grad
    if (!node->d_grad || !node->d_grad->is_allocated()) { cout<<"No grad allocated yet."<<endl; }
    else
    {
        int total = node->h_meta->size;
        int limit = total > 20 ? 20 : total;
        float* h_grad = node->d_grad->d_to_h();

        cout<<"Grad: [";
        for (int i = 0; i < limit; ++i)
        {
            if (i > 0) cout<<", ";
            cout<<h_grad[i];
        }
        if (total > limit) cout<<", ... ("<<total<<" total)";
        cout<<"]"<<endl;

        delete[] h_grad;
    }
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

std::shared_ptr<Node> reduction(std::shared_ptr<Node> a, int dim, OpType op, bool keepdims, std::shared_ptr<Node> dst)
{
    auto [res_shape, res_ndim] = check_shape_return_out_shape::reduction(
        a, dim, keepdims
    );
    std::shared_ptr<Node> inputs[] = {a};
    auto node = make_internal_node({
        .op = op,
        .shape = res_shape,
        .ndim = res_ndim,
        .inputs = inputs,
        .num_inputs = 1,
        .dst = dst
    });
    node->label = a->label + opToString(op); // remove later
    return node;
}
//      </general operations>

//      <ops>
std::shared_ptr<Node> add(std::shared_ptr<Node> a, std::shared_ptr<Node> b) { return elemwise(a, b, OpType::Add); }
std::shared_ptr<Node> mul(const std::shared_ptr<Node> a, const std::shared_ptr<Node> b) { return elemwise(a, b, OpType::Mul); }
std::shared_ptr<Node> max(std::shared_ptr<Node> a, int dim, bool keepdims) { return reduction(a, dim, OpType::Max, keepdims); }
std::shared_ptr<Node> min(std::shared_ptr<Node> a, int dim, bool keepdims) { return reduction(a, dim, OpType::Min, keepdims); }
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
std::shared_ptr<Node> skip(std::shared_ptr<Node> a, std::shared_ptr<Node> dst)
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
    node->label = "contiguous(" + a->label + ")";
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
    node->d_grad = a->d_grad;
    node->label = "reshape(" + a->label + ")";

    // connect inputs to outputs and vice versa
    for (int i = 0; i < 1; i++)
    {
        node->inputs[i] = inputs[i];
        node->param_count += inputs[i]->param_count;
        if (inputs[i]->outputs[0] == nullptr) inputs[i]->outputs[0] = node;
        else inputs[i]->outputs[1] = node;
    }
    return node;
}

std::shared_ptr<Node> sum_last_dim(std::shared_ptr<Node> a, std::shared_ptr<Node> dst)
{
    int res_ndim = a->h_meta->ndim - 1;
    int res_shape[res_ndim];
    for (int i = 0; i < res_ndim; ++i) res_shape[i] = a->h_meta->shape()[i];
    std::shared_ptr<Node> inputs[] = {a};
    cout<<"NDIM="<<res_ndim<<endl;
    auto node = make_internal_node({
        .op = OpType::Sum,
        .shape = res_shape,
        .ndim = res_ndim,
        .inputs = inputs,
        .num_inputs = 1,
        .dst = dst
    });
    node->label = "sum_last_dim(" + a->label + ")";
    return node;
}

std::shared_ptr<Node> sum(std::shared_ptr<Node> a, int dim, bool keepdims)
{
    // handle negative indexing
    int ndim = a->h_meta->ndim;
    dim = (dim + ndim) % ndim;
    if (dim > ndim - 1) throw logic_error("Dim wasn't found.");
    // if (dim == 52)
    // {
    //     flatten
    // }

    if (dim == ndim - 1 && !keepdims) return sum_last_dim(a);

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
    std::shared_ptr<Node> node_after_sum = sum_last_dim(contiguous_node);

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
    node->label = "expand_last_dim(" + a->label + ")";
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
        for (int i = 0; i < num_outputs; i++) build_topo_grad(inode->outputs[i].get(), visited, order);
    }
    // cout<<node->label<<" | ";
    order.push_back(node);
}

void build_topo_grad(BwdNode* node, unordered_set<BwdNode*>& visited, vector<BwdNode*>& order)
{
    if (visited.count(node)) return;
    visited.insert(node);

    if (node->kind == NodeKind::Internal) {
        // compute num_outputs
        int num_outputs = 0;
        while (num_outputs < 4 && node->outputs[num_outputs]) num_outputs++;
        // same but with outputs
        for (int i = 0; i < num_outputs; i++) 
        {
            BwdNode* n = static_cast<BwdNode*>(node->outputs[i].get());
            build_topo_grad(n, visited, order);
            
        }
    }
    // cout<<node->label<<" | ";
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

vector<BwdNode*> topo_grad_sort(shared_ptr<BwdNode> node)
{
    unordered_set<BwdNode*> visited;
    vector<BwdNode*> order;
    build_topo_grad(node.get(), visited, order);
    return order;
}
// </Topological sort>


void eval(std::shared_ptr<InternalNode> root)
{
    // perform topological sort
    vector<Node*> order = topo_sort(root, false);
    
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

        // dispatch
        op_based_eval(inode);
    }
}

void eval(std::shared_ptr<Node> root)
{
    // perform topological sort
    vector<Node*> order = topo_sort(root, false);

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
        
        // dispatch
        op_based_eval(inode);
    }
}

void geval(std::shared_ptr<InternalNode> root)
{
    // perform topological sort
    vector<Node*> order = topo_sort(root, true);
    cout<<"order.size() = "<<order.size()<<endl;

    // fill the root node grad with beer
    prepare_starting_grad(root);
    
    // walk in order
    for (int i = order.size() - 2; i >= 0; --i)
    {   
        cout<<"Current node: "<<order[i]->label<<endl;
        if (order[i]->kind == NodeKind::Source) continue;

        InternalNode* inode = static_cast<InternalNode*>(order[i]);

        // allocate grad
        int size = utils::numel(inode->h_meta->ndim, inode->h_meta->shape());
        if (!inode->d_data) inode->d_data = make_shared<ContentData>();
        if (!inode->d_grad) inode->d_grad = make_shared<ContentData>();
        inode->d_data->allocate(size);
        inode->d_grad->allocate(size);
        inode->flipped = !inode->flipped;
        cout<<"flipped="<<inode->flipped<<endl;
        if (inode->d_grad->is_allocated()) cout<<"grad was allocated"<<endl;
        
        // dispatch
        op_based_eval(inode);
    }
}

void geval(std::shared_ptr<Node> root)
{
    // perform topological sort
    vector<Node*> order = topo_sort(root, true);

    // fill the root node grad with beer
    prepare_starting_grad(root);

    cout<<"GEVAL LOOP: "<<endl;
    // walk in order
    for (int i = order.size() - 2; i >= 0; --i)
    {   
        if (order[i]->kind == NodeKind::Source) continue;

        InternalNode* inode = static_cast<InternalNode*>(order[i]);
        
        // allocate grad
        int size = utils::numel(inode->h_meta->ndim, inode->h_meta->shape());
        if (auto* bwd = dynamic_cast<BwdNode*>(order[i])) {
            if (!inode->d_grad) inode->d_grad = make_shared<ContentData>();
            inode->d_grad->allocate(size);
        } else {
            if (!inode->d_data) inode->d_data = make_shared<ContentData>();
            inode->d_data->allocate(size);
        }

        // dispatch
        cout<<"node: "<<inode->label<<endl;
        cout << "Type is: " << typeid(order[i]).name() << std::endl;
        if constexpr (std::is_same_v<decltype(order[i]), BwdNode>) {
            std::cout << "type is bwd node.\n";
        }
        cout<<"flipped: "<<inode->flipped<<endl;
        cout<<"is_grad: "<<(inode->d_grad && inode->d_grad->is_allocated())<<endl;
        cout<<"size: "<<(inode->d_data ? inode->d_data->size() : inode->d_grad->size())<<endl;
        op_based_eval(inode);
        print_node(inode);
    }
}

void _geval(std::shared_ptr<Node> root)
{
    // perform topological sort
    vector<Node*> order = topo_sort(root, true);

    // fill the root node grad with beer
    // prepare_starting_grad(root);

    cout<<"GEVAL LOOP: "<<endl;
    // walk in order
    for (int i = order.size() - 2; i >= 0; --i)
    {
        if (order[i]->kind == NodeKind::Source) continue;

        InternalNode* inode = static_cast<InternalNode*>(order[i]);

        // allocate grad
        int size = inode->h_meta->size;
        if (auto* bwd = dynamic_cast<BwdNode*>(order[i])) {
            if (!inode->d_grad) inode->d_grad = make_shared<ContentData>();
            inode->d_grad->allocate(size);
        } else {
            if (!inode->d_data) inode->d_data = make_shared<ContentData>();
            inode->d_data->allocate(size);
        }

        // dispatch
        cout<<"------------------------------------------------"<<endl;
        cout<<"node: "<<inode->label<<endl;
        cout << "Type is: " << typeid(order[i]).name() << std::endl;
        if constexpr (std::is_same_v<decltype(order[i]), BwdNode>) {
            std::cout << "type is bwd node.\n";
        }
        cout<<"flipped: "<<inode->flipped<<endl;
        cout<<"is_grad: "<<(inode->d_grad && inode->d_grad->is_allocated())<<endl;
        cout<<"is_data: "<<(inode->d_data && inode->d_data->is_allocated())<<endl;
        cout<<"size: "<<(inode->h_meta->size)<<endl;
        op_based_eval(inode);
        print_node(inode);
    }
}

void op_based_eval(InternalNode* inode)
{
    switch (inode->op)
    {
        case OpType::Add:           cu_ops::add(inode);                 return;
        case OpType::Mul:           cu_ops::mul(inode);                 return;
        case OpType::MatMul:        cu_ops::matmul(inode);              return;
        case OpType::Contiguous:    cu_ops::contiguous(inode);          return;
        case OpType::Sum:           cu_ops::reduce_last_dim(inode);     return;
        case OpType::ExpandLD:      cu_ops::expand_last_dim(inode);     return;
        case OpType::BwdStart:      cu_ops::fill(inode, 1.0f);          return;
        case OpType::Reshape:       return; // no-op: shares d_content with input, data unchanged
        default:                    return;
    }
}

void op_based_eval(BwdNode* inode)
{
    switch (inode->op)
    {
        case OpType::Add:           cu_ops::add(inode);                 return;
        case OpType::Mul:           cu_ops::mul(inode);                 return;
        case OpType::MatMul:        cu_ops::matmul(inode);              return;
        case OpType::Contiguous:    cu_ops::contiguous(inode);          return;
        case OpType::Sum:           cu_ops::reduce_last_dim(inode);     return;
        case OpType::ExpandLD:      cu_ops::expand_last_dim(inode);     return;
        case OpType::Reshape:       return; // no-op: shares d_content with input, data unchanged
        default:                    return;
    }
}