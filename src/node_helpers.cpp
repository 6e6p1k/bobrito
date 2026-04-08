#include <ostream>
#include <memory>
#include <string>
#include <vector>
#include "allocator.cuh"
#include "node.h"
#include "node_helpers.h"
#include "ops.cuh"

using namespace std;

std::shared_ptr<SourceNode> make_filled_source_node(int* shape, int ndim, float value, bool requires_grad, std::string label)
{
    std::shared_ptr<SourceNode> node = std::make_shared<SourceNode>();
    node->kind = NodeKind::Source;
    node->h_requiers_grad = requires_grad;
    node->d_data = std::make_shared<ContentData>();
    node->h_meta = make_shape_data(ndim, shape);
    node->label = label;
    // allocate & fill
    node->d_data->allocate(node->h_meta->size); 
    cu_ops::fill(node.get(), value);
    return node;
}

std::shared_ptr<SourceNode> make_filled_source_node(std::vector<int> vec_shape, int ndim, float value, bool requires_grad, std::string label)
{
    std::shared_ptr<SourceNode> node = std::make_shared<SourceNode>();
    node->kind = NodeKind::Source;
    node->h_requiers_grad = requires_grad;
    node->d_data = std::make_shared<ContentData>();

    // convert to array bc make_shape_data only receives cpp arrays
    int* shape = new int[vec_shape.size()];
    for (int i = 0; i < vec_shape.size(); ++i)
        shape[i] = vec_shape[i];

    node->h_meta = make_shape_data(ndim, shape);
    node->label = label;
    // allocate & fill
    node->d_data->allocate(node->h_meta->size); 
    cu_ops::fill(node.get(), value);
    return node;
}

std::shared_ptr<InternalNode> make_internal_node(OpType op, std::shared_ptr<Node> inputs[], int num_inputs, int* shape, int ndim, int* strides, int param_count)
{
    // Creates new internal node and assigns given parameters
    // Default: strides = nullptr, param_count = 0
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
        if (inputs[i]->outputs[0].lock() == nullptr) inputs[i]->outputs[0] = node;
        else inputs[i]->outputs[1] = node;
    }
    return node;
}

std::shared_ptr<InternalNode> make_internal_node(InodeConfig cfg)
{
    // Creates a new internal node and assigns data from config.
    // If cfg.dst is specified, copies h_meta to the new node
    // and cfg.copy_data = true or cfg.copy_grad = true, copies data or grad respectively to the new node.

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
        if (cfg.inputs[i]->outputs[0].lock() == nullptr) cfg.inputs[i]->outputs[0] = node;
        else cfg.inputs[i]->outputs[1] = node;
    }
    if (cfg.is_param_count()) node->param_count = cfg.param_count;
    if (cfg.label != "") node->label = cfg.label;
    if (!cfg.dst) node->d_data = make_shared<ContentData>();
    else
    {
        if (cfg.copy_data) node->d_data = cfg.dst->d_data;
        if (cfg.copy_grad) node->grad_node = cfg.dst->grad_node;
        cfg.dst->h_meta = node->h_meta;
    }
    return node;
}

std::shared_ptr<InternalNode> copy_to_internal_node(std::shared_ptr<Node> example)
{
    auto node = std::make_shared<InternalNode>();
    node->kind = NodeKind::Internal;
    node->h_meta = example->h_meta;
    node->d_data = example->d_data;
    node->grad_node = example->grad_node;
    node->label = example->label + "_copy";
    node->param_count = example->param_count;
    return node;
}

std::shared_ptr<InternalNode> copy_to_internal_node(std::shared_ptr<Node> example, bool transposed, OpType op)
{
    auto node = std::make_shared<InternalNode>();
    node->kind = NodeKind::Internal;
    node->h_meta = example->h_meta;
    node->d_data = example->d_data;
    node->grad_node = example->grad_node;
    node->label = example->label + "_copy";
    node->transposed = transposed;
    node->param_count = example->param_count;
    node->op = op;
    return node;
}

std::shared_ptr<InternalNode> copy_to_internal_node(InodeConfig cfg)
{
    std::shared_ptr<InternalNode> node = std::make_shared<InternalNode>();

    if (cfg.data_copying_required())
    {
        node->kind = cfg.src->kind;
        node->h_meta = cfg.src->h_meta;
        node->label = cfg.src->label;
        node->param_count = cfg.src->param_count;

        if (cfg.copy_data) node->d_data = cfg.src->d_data;
        if (cfg.copy_grad) node->grad_node = cfg.src->grad_node;
    }

    if (cfg.is_op())
        node->op = cfg.op;

    if (cfg.is_label())
        node->label = cfg.label;


    if (cfg.is_meta())
        node->h_meta = make_shape_data(cfg.ndim, cfg.shape, cfg.strides);

    if (cfg.is_linking())
    {
        node->num_inputs = cfg.num_inputs;
        for (int i = 0; i < cfg.num_inputs; i++)
        {
            node->inputs[i] = cfg.inputs[i];
            node->param_count += cfg.inputs[i]->param_count;

            if (cfg.inputs[i]->outputs[0].lock() == nullptr)
                cfg.inputs[i]->outputs[0] = node;
            else
                cfg.inputs[i]->outputs[1] = node;
        }
    } else node->num_inputs = 0;

    if (cfg.is_param_count())
        node->param_count = cfg.param_count;

    if (cfg.data_transfer_required())
    {
        if (cfg.src_d_data) node->d_data = cfg.src_d_data;
    }

    return node;
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
    auto grad = node->grad_node.lock();
    if (!grad || !grad->d_data || !grad->d_data->is_allocated()) { cout<<"No grad allocated yet."<<endl; }
    else
    {
        int total = node->h_meta->size;
        int limit = total > 20 ? 20 : total;
        float* h_grad = grad->d_data->d_to_h();

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