#include <cstddef>
#include <iostream>
#include <memory>
#include <unordered_set>
#include <vector>

#include "utils.h"
#include "allocator.cuh"
#include "grad.h"

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

void ShapeData::print() {
    std::cout << "Shape: ";
    for(int i = 0; i < ndim; i++) std::cout << shape()[i] << " ";
    std::cout << "\nStrides: ";
    for(int i = 0; i < ndim; i++) std::cout << strides()[i] << " ";
    std::cout << std::endl;
}
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
//      source node initialisation
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

std::shared_ptr<Node> make_internal_node(OpType op, std::shared_ptr<Node> inputs[], int num_inputs, int* shape, int ndim, int* strides, int use_count)
{
    auto node = std::make_shared<InternalNode>();
    node->kind = NodeKind::Internal;
    node->op = op;
    node->num_inputs = num_inputs;
    node->h_meta = make_shape_data(ndim, shape, strides);
    for (int i = 0; i < num_inputs; i++)
    {
        node->inputs[i] = inputs[i];
        node->use_count += inputs[i]->use_count;
        cout<<"Node ["<<inputs[i]->label<<"] has "<<inputs[i]->use_count<<" labels"<<endl;
    }
    return node;
}

std::shared_ptr<InternalNode> make_internal_node_like(InternalNode* example)
{
    auto node = std::make_shared<InternalNode>();
    node->kind = example->kind;
    node->op = example->op;
    node->num_inputs = example->num_inputs;
    node->h_meta = example->h_meta;
    node->d_content = example->d_content;
    node->use_count = example->use_count;
    return node;
}

std::shared_ptr<InternalNode> copy_to_internal_node(std::shared_ptr<Node> example, int use_count)
{
    if (example->kind == NodeKind::Internal)
    {
        SourceNode* snode = static_cast<SourceNode*>(&*example);
        if (snode->h_requiers_grad) use_count -= 1;
    }
    auto node = std::make_shared<InternalNode>();
    node->kind = NodeKind::Internal;
    node->h_meta = example->h_meta;
    node->d_content = example->d_content;
    node->use_count = use_count;
    return node;
}

std::shared_ptr<SourceNode> make_source_node_like(SourceNode* example)
{
    auto node = std::make_shared<SourceNode>();
    node->kind = example->kind;
    node->h_meta = example->h_meta;
    node->d_content = example->d_content;
    node->h_requiers_grad = example->h_requiers_grad;
    node->use_count = example->use_count;
    return node;
}
// </Node>


// <nodes interaction>
//      <general operations>



std::shared_ptr<Node> elemwise(std::shared_ptr<Node> a, std::shared_ptr<Node> b, OpType op)
{
    auto [res_shape, res_ndim] = check_shape_return_out_shape::elemwise(
        a.get(), b.get()
    );
    std::shared_ptr<Node> inputs[] = {a, b};
    auto node = make_internal_node(
        op, inputs, 2, res_shape, res_ndim
    );
    node->label = "( " + a->label + " " + opToString(op) + " " + b->label + " )"; // remove later
    return node;
}

std::shared_ptr<Node> reduction(std::shared_ptr<Node> a, int dim, OpType op, bool keepdims)
{
    auto [res_shape, res_ndim] = check_shape_return_out_shape::reduction(
        a.get(), dim, keepdims
    );
    std::shared_ptr<Node> inputs[] = {a};
    auto node = make_internal_node(
        op, inputs, 1, res_shape, res_ndim
    );
    node->label = a->label + opToString(op); // remove later
    return node;
}
//      </general operations>


std::shared_ptr<Node> add(std::shared_ptr<Node> a, std::shared_ptr<Node> b)
{
    return elemwise(a, b, OpType::Add);
}

std::shared_ptr<Node> mul(const std::shared_ptr<Node> a, const std::shared_ptr<Node> b)
{
    return elemwise(a, b, OpType::Mul);
}

std::shared_ptr<Node> sum(std::shared_ptr<Node> a, int dim, bool keepdims)
{
    return reduction(a, dim, OpType::Sum, keepdims);
}

std::shared_ptr<Node> max(std::shared_ptr<Node> a, int dim, bool keepdims)
{
    return reduction(a, dim, OpType::Max, keepdims);
}

std::shared_ptr<Node> min(std::shared_ptr<Node> a, int dim, bool keepdims)
{
    return reduction(a, dim, OpType::Min, keepdims);
}
std::shared_ptr<Node> skip(std::shared_ptr<Node> a)
{
    std::shared_ptr<Node> inputs[] = {a};
    auto node = make_internal_node(
        OpType::Skip, inputs, 1, a->h_meta->shape(), a->h_meta->ndim
    );
    return node;
}


// </nodes interaction>

namespace backward_node_ops 
{
void add(
    std::shared_ptr<InternalNode> self, std::shared_ptr<InternalNode> mirror
)
{
    // just forward the gradient to each prev nodes
    if (self->use_count < 1 || self->kind == NodeKind::Source) return;
    for (int i = 0; i <= mirror->num_inputs; ++i)
    {
        if (mirror->inputs[i]->kind == NodeKind::Internal)
        {            
            InternalNode* inode = static_cast<InternalNode*>(&*(mirror->inputs[i]));
            auto new_node = make_internal_node_like(inode);
            new_node->use_count = self->use_count;
            self->outputs[i] = skip(new_node);
        }
        else 
        {
            SourceNode* snode = static_cast<SourceNode*>(&*(mirror->inputs[i]));
            auto new_node = make_source_node_like(snode);
            if (new_node->h_requiers_grad) self->use_count -= 1;
            new_node->use_count = self->use_count;
            self->outputs[i] = skip(new_node);
        }
    } 
}

void mul(
    std::shared_ptr<InternalNode> self, std::shared_ptr<InternalNode> mirror
)
{
    // self = node_1 * node_2
    // d(self)/d(node_1) = node_2 and vice versa
    // also * self as it contains grad
    
    int use_count = self->use_count;
    if (use_count < 1) return;
    

    // only 2 inputs always
    std::shared_ptr<InternalNode> node_1 = copy_to_internal_node(mirror->inputs[0], use_count);
    std::shared_ptr<InternalNode> node_2 = copy_to_internal_node(mirror->inputs[0], use_count);

    // handle op
    node_1->op = OpType::Mul;
    node_2->op = OpType::Mul;

    // handle inputs of each
    std::shared_ptr<InternalNode> inputs_1[2] = {node_2, self};
    std::shared_ptr<InternalNode> inputs_2[2] = {node_1, self};

    // connect both to the self
    std::shared_ptr<InternalNode> outputs_1[2] = {node_1};
}
}













// <Topological sort>
void build_topo(Node* node, unordered_set<Node*>& visited, vector<InternalNode*>& inodes, vector<SourceNode*>& snodes)
{
    if (visited.count(node)) return;
    visited.insert(node);

    if (node->kind == NodeKind::Internal) 
    {
        InternalNode* inode = static_cast<InternalNode*>(node);
        for (int i = 0; i < inode->num_inputs; i++)
            build_topo(inode->inputs[i].get(), visited, inodes, snodes);
        inodes.push_back(inode);
    }
    else {
        snodes.push_back(static_cast<SourceNode*>(node));
    }
    
}
// </Topological sort>


void backward_pass(shared_ptr<Node> node)
{
    unordered_set<Node*> visited;
    vector<InternalNode*> inodes;
    vector<SourceNode*> snodes;
    build_topo(node.get(), visited, inodes, snodes);

    cout<<"Overview:"<<endl;
    for (int i = 0; i < snodes.size(); ++i)
    {
        cout<<"************************************"<<endl;
        SourceNode* snode = snodes[i];
        cout<<"Source Node: "<<snode->label<<endl;
    } 

    for (int i = inodes.size() - 1; i >=0; --i)
    {
        InternalNode* inode = inodes[i];
        cout<<"************************************"<<endl;
        cout<<"Current op: "<<inode->op<<endl;
        cout<<"Inputs num: "<<inode->num_inputs<<endl;
        cout<<"Current expr: "<<inode->label<<endl;
    }

    for (int i = inodes.size() - 1; i >=0; --i)
    {
        InternalNode* inode = inodes[i];
        for (int i = 0; i < inode->num_inputs; i++)
        {
            SourceNode* curr = snodes.back();
            if (snodes.back()->label == inode->inputs[i]->label)
            {
                snodes.pop_back();

            }
        }
        cout<<"-----------------------------------"<<endl;
        cout<<"Current op: "<<inode->op<<endl;
        cout<<"Inputs num: "<<inode->num_inputs<<endl;
        cout<<"Current expr: "<<inode->label<<endl;
    }
}
