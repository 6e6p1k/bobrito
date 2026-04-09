#include "allocator.cuh"
#include "graph_viz.h"
#include "node.h"
#include "node_helpers.h"
#include "tensor.cuh"
#include "bobr.cuh"
#include <memory>

namespace bobr 
{
Tensor full(std::vector<int> shape, float val, bool requires_grad)
{
    std::shared_ptr<SourceNode> snode = make_filled_source_node(shape, shape.size(), val);
    snode->h_requiers_grad = requires_grad;
    return Tensor(snode);
}

Tensor full_like(Tensor t, float val)
{ 
    int* shape = t.node->h_meta->shape();
    int ndim = t.node->h_meta->ndim;
    bool requires_grad = false;
    std::shared_ptr<SourceNode> snode = make_filled_source_node(shape, ndim, val, requires_grad);
    Tensor out_t = Tensor(snode);
    if (t.node->kind == NodeKind::Source)
    {
        std::shared_ptr<SourceNode> snode = std::static_pointer_cast<SourceNode>(t.node);
        requires_grad = snode->h_requiers_grad;
    }
    return out_t;
}

Tensor ones(std::vector<int> shape, bool requires_grad)
{ return full(shape, 1.0f, requires_grad); }

Tensor zeros(std::vector<int> shape, bool requires_grad)
{ return full(shape, 0.0f, requires_grad); }

Tensor ones_like(Tensor t)
{ return full_like(t, 1.0f); }

Tensor zeros_like(Tensor t)
{ return full_like(t, 0.0f); }

std::string get_allocation_stats()
{ return MemoryPool::call().stats(); }

void reset_memory_pool()
{ MemoryPool::call().reset(); }

void clear_graph(Tensor t)
{ clear_grads(t.node); }

void draw_graph(Tensor t, const char* pdf_path)
{ 
    std::string dot_path = std::string(pdf_path) + ".dot";
    export_dot_full(t.node.get(), dot_path.c_str());
    std::string cmd = "dot -Tpdf " + dot_path + " -o " + pdf_path;
    system(cmd.c_str());
    std::remove(dot_path.c_str());
}
}
