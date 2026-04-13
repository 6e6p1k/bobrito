#include <memory>
#include <sstream>
#include <string>
#include "nodes.h"
#include "ops.h"
#include "evaluate.h"
#include "allocator.cuh"

using namespace std;

// ============================================================
//                        NODE INTERFACE
// ============================================================

namespace bobr {
shared_ptr<Node> get_ptr(shared_ptr<Node> ptr) {
    return ptr;
}

template <typename T> T* get_data(shared_ptr<Node> ptr) {
    if (!ptr->is_data_allocated()) throw logic_error("Data not allocated!");
    return static_cast<T*>(ptr->d_data->d_to_h());
}

template <typename T> T* get_grad(shared_ptr<Node> ptr) {
    if (!ptr->requires_grad || ptr->inputs.size() != 0) throw logic_error("This is not a leaf node!");
    if (!ptr->owned_grad || !ptr->owned_grad->is_data_allocated())
        throw logic_error("Grad is not allocated!");
    return static_cast<T*>(ptr->owned_grad->d_data->d_to_h());
}

int* get_shape(shared_ptr<Node> ptr) {
    if (!ptr->is_meta()) throw logic_error("Node has no meta!");
    return ptr->h_meta->shape();
}

int* get_strides(shared_ptr<Node> ptr) {
    if (!ptr->is_meta()) throw logic_error("Node has no meta!");
    return ptr->h_meta->strides();
}

int get_ndim(shared_ptr<Node> ptr) {
    if (!ptr->is_meta()) throw logic_error("Node has no meta!");
    return ptr->h_meta->ndim;
}

int get_size(shared_ptr<Node> ptr) {
    if (!ptr->is_meta()) throw logic_error("Node has no meta!");
    return ptr->h_meta->size;
}

bool get_requires_grad(shared_ptr<Node> ptr) {
    return ptr->requires_grad;
}

string get_label(shared_ptr<Node> ptr) {
    return ptr->label;
}

void set_label(shared_ptr<Node> ptr, string new_label) {
    ptr->set_label(new_label);
}

void get_op(shared_ptr<Node> ptr) {
    if (!ptr->prim) throw logic_error("Node has no primitive!");
    cout << "Primitive: " << ptr->prim->name() << "\n";
}

void get_inputs(shared_ptr<Node> ptr) {
    for (auto& inp : ptr->inputs) {
        if (!inp) cout << "<nullptr input>\n";
        else print_node(inp);
    }
}

void get_outputs(shared_ptr<Node> ptr) {
    for (auto& out : ptr->outputs) {
        auto sp = out.lock();
        if (!sp) cout << "<expired output>\n";
        else print_node(sp);
    }
}

void export_graph(shared_ptr<Node> ptr, string filename) {
    ptr->export_graph(filename);
}

void evaluate(shared_ptr<Node> ptr) {
    Graph g = Graph(ptr, true);
    g.eval();
}

void build_backward_graph(shared_ptr<Node> ptr) {
    Graph g = Graph(ptr, true);
    g.backward();
}


// ============================================================
//                        GRAPH INTERFACE
// ============================================================
class GraphAPI {
    shared_ptr<Graph> ptr;

    public:
    GraphAPI(shared_ptr<Graph> graph_ptr) : ptr(graph_ptr) {}

    void eval() {
        ptr->eval();
    }

    void build_backward_graph() {
        ptr->backward();
    }

    void sgd_step(float lr) {
        ptr->sgd_step(lr);
    }

    void detach_grads_from_bwd_graph() {
        ptr->detach_grads_from_bwd_graph();
    }

    void train(int n_iters, float lr) {
        if (ptr->disposable) throw logic_error("Graph is disposable, cannot train!");
        if (ptr->bwd_order.size() == 0) throw logic_error("Backward graph is empty, cannot train!");
        if (ptr->require_grad_nodes.size() == 0) throw logic_error("No leaf nodes that require grad, cannot train!");
        for (int i = 0; i < n_iters; i++) {
            eval();
            sgd_step(lr);
        }
        detach_grads_from_bwd_graph();
    }


};

void synchronize_device_and_check() {
    CUDA_CHECK();
}

void get_memory_pool_stats() {
    cout << MemoryPool::call().stats() << endl;
}

void get_memory_summary() {
    auto s = snapshot();
    print_snapshot("VRAM SUMMARY", s);
}
}