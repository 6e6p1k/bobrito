#include <iostream>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>

#include "allocator.cuh"
#include "nodes.h"
#include "ops.h"
#include "evaluate.h"

using namespace std;

// ============================================================
//                       BACKWARD HELPER
// ============================================================
void build_backward(shared_ptr<Node> fwd_node, shared_ptr<Node> bwd_node,
                    unordered_map<Node*, int>& remaining);

// Count how many times each node is consumed as an input.
// Indicates that all upstream grads have arrived.
void count_uses(shared_ptr<Node> node, unordered_map<Node*, int>& uses,
                unordered_set<Node*>& visited) {
    if (visited.count(node.get())) return;
    visited.insert(node.get());
    if (!node->is_meta()) return;

    for (auto input : node->inputs) {
        uses[input.get()]++;
        count_uses(input, uses, visited);
    }
}

// Accumulate gradient for a node. When all upstream grads
// have arrived, recurse into that node's vjp.
void set_grad_and_traverse(shared_ptr<Node> fwd_node, shared_ptr<Node> new_grad,
                           unordered_map<Node*, int>& remaining) {
    // accumulate
    shared_ptr<Node> final_grad;
    if (fwd_node->grad_buffer) {
        fwd_node->grad_buffer->set_label(fwd_node->label + "_partial_grad");
        final_grad = add(fwd_node->grad_buffer, new_grad);
    } else
        final_grad = new_grad;
    final_grad->set_label(fwd_node->label + "_grad");
    // final_grad->d_data = make_shared<ContentData>();

    remaining[fwd_node.get()]--;

    // if leaf node (only those can have requires_grad = true),
    // just store the grad
    if (fwd_node->requires_grad) {
        fwd_node->owned_grad = final_grad;
    }

    // not all upstream grads arrived yet — stash and wait
    if (remaining[fwd_node.get()] > 0) {
        fwd_node->grad_buffer = final_grad;
        return;
    }

    // all grads arrived — recurse
    build_backward(fwd_node, final_grad, remaining);
    fwd_node->grad_buffer = nullptr;
}

// The core: ask the primitive for raw grads, reduce for
// broadcasting, wire them up.
void build_backward(shared_ptr<Node> fwd_node, shared_ptr<Node> bwd_node,
                    unordered_map<Node*, int>& remaining) {
    if (fwd_node->inputs.size() <= 0 || fwd_node->requires_grad) return;
    if (!fwd_node->prim) return;

    // ask the op for raw gradients
    shared_ptr<Node> grads[4] = {};
    fwd_node->prim->vjp(fwd_node, bwd_node, grads);

    // go through grads and remove leaf nodes in which requires_grad = false

    // wire up each input
    for (int i = 0; i < fwd_node->inputs.size(); i++) {
        if (!grads[i]) continue;
        // check if node is leaf but doesn't require grad (eg x in x @ W + b)
        bool requires_grad = fwd_node->inputs[i]->requires_grad;
        bool is_leaf = fwd_node->inputs[i]->inputs.size() == 0;
        if (is_leaf && !requires_grad) {
            // remove useless grads from outputs to avoid computing them
            bwd_node->outputs[i].reset();
            continue;
        }
        auto input = fwd_node->inputs[i];

        // handle broadcasting: reduce grad to input's shape
        auto reduced =
            reduce_to_shape(grads[i], input->get_shape(), input->get_ndim());

        set_grad_and_traverse(input, reduced, remaining);
    }

    // go through bwdnode-> outputs vector
    // and remove all the nullptrs
    auto& v = bwd_node->outputs;
    for (int i = v.size() - 1; i >= 0; --i) {
        if (v[i].expired()) {
            v.erase(v.begin() + i); // Actually deletes it right now
        }
    }
}

// ============================================================
//                       TOPO HELPERS
// ============================================================
void build_topo_inputs(Node* node, unordered_set<Node*>& visited,
                       vector<Node*>& order) {
    if (visited.count(node)) return;
    visited.insert(node);
    for (auto& inp : node->inputs) build_topo_inputs(inp.get(), visited, order);
    order.push_back(node);
}

void build_topo_outputs(Node* node, unordered_set<Node*>& visited,
                        vector<Node*>& order) {
    // skip nodes that were freed (dead-end backward branches produce
    // expired weak_ptrs in outputs when no requires_grad path uses them)
    if (!node) return;
    if (visited.count(node)) return;
    visited.insert(node);
    // push before children (pre-order traversal)
    order.push_back(node);  // bc we need to go from root node to leafs
    for (auto& out : node->outputs)
        build_topo_outputs(out.lock().get(), visited, order);
}

void build_topo_inputs_bwd(Node* node, unordered_set<Node*>& visited,
                           vector<Node*>& order) {
    if (visited.count(node)) return;
    visited.insert(node);
    for (auto& inp : node->inputs) {
        if (inp->is_bwd)
            build_topo_inputs_bwd(inp.get(), visited, order);
    }
    order.push_back(node);
}

void eval_helper(vector<Node*>& fwd_order, vector<Node*>& bwd_order) {
    vector<Node*> require_grad;

    for (Node* node : fwd_order) {
        if (!node->is_meta()) continue;
        int size = node->get_size();
        if (!node->is_data()) node->d_data = make_shared<ContentData>();
        if (!node->is_data_allocated()) node->d_data->allocate(size);
        node->prim->eval(node);

        if (node->requires_grad) require_grad.push_back(node);
    }

    for (Node* node : bwd_order) {
        if (!node->is_meta()) continue;
        int size = node->get_size();

        if (!node->is_data()) node->d_data = make_shared<ContentData>();
        if (!node->is_data_allocated()) node->d_data->allocate(size);

        // DEBUG(node, "GEVAL LOOP");

        node->prim->eval(node);
    }
}

// ============================================================
//                       GRAPH
// ============================================================
Graph::Graph(shared_ptr<Node> loss, bool disposable) {
    root = loss;
    compute_fwd_order();
    locate_leaf_w_grad_nodes();
    this->disposable = disposable;
}

void Graph::backward() {
    if (need_backward()) {
        _backward();
        compute_bwd_order();
    }
}

void Graph::eval() {
    fwd_pass();
    if (need_backward()) {
        bwd_pass();
    }
    if (disposable) detach_grads_from_bwd_graph();
    CUDA_CHECK();
}

// CHECKS
bool Graph::is_root() { return root != nullptr; }

bool Graph::is_backward() {
    if (root->outputs.size() != 1) return false;
    if (root->outputs[0].expired()) return false;
    return (root->outputs[0].lock()->label == "BWD_START");
}

bool Graph::need_backward() {
    if (require_grad_nodes.size() == 0) locate_leaf_w_grad_nodes();
    return (require_grad_nodes.size() > 0);
}

// FUNCS
void Graph::compute_fwd_order() {
    if (!is_root()) throw logic_error("Root node is nullptr!");
    unordered_set<Node*> visited;
    build_topo_inputs(root.get(), visited, fwd_order);
}

void Graph::compute_bwd_order() {
    unordered_set<Node*> visited;
    for (Node* leaf : require_grad_nodes) {
        if (leaf->owned_grad) {
            build_topo_inputs_bwd(leaf->owned_grad.get(), visited, bwd_order);
        }
    }
}

void Graph::locate_leaf_w_grad_nodes() {
    if (require_grad_nodes.size() > 0)
        throw logic_error("require_grad_nodes is already filled.");
    for (Node* node : fwd_order) {
        if (!node->is_meta()) continue;
        if (node->requires_grad) require_grad_nodes.push_back(node);
    }
}

void Graph::pass(vector<Node*> order) {
    for (Node* node : order) {
        if (!node) continue;
        if (!node->is_meta()) continue;
        node->prim->allocate(node);
        node->prim->eval(node);
    }
}

void Graph::fwd_pass() { return pass(fwd_order); }
void Graph::bwd_pass() { return pass(bwd_order); }

void Graph::_backward() {
    if (root->requires_grad || root->inputs.size() == 0)
        throw logic_error("Cannot call backward() on a source node.");

    // seed: gradient of loss w.r.t. itself = 1
    auto seed = full(root->get_shape(), root->get_ndim(), 1.0f);
    seed->set_label("BWD_START");
    seed->is_bwd = true;
    root->outputs.push_back(seed);

    // count how many consumers each node has
    unordered_map<Node*, int> uses;
    unordered_set<Node*> visited;
    count_uses(root, uses, visited);

    build_backward(root, seed, uses);
}

void Graph::detach_grads_from_bwd_graph() {
    for (auto grad_leaf : require_grad_nodes)
        grad_leaf->owned_grad = copy_node_no_prim_before_eval(grad_leaf->owned_grad);
}
// ============================================================
//                       PUBLIC API
// ============================================================
shared_ptr<Graph> make_model(shared_ptr<Node> root) {
    auto g = make_shared<Graph>(root, false);
    return g;
}

void eval(shared_ptr<Node> root, bool export_graph) {
    Graph g = Graph(root, true);
    g.backward();
    if (export_graph) root->export_graph();
    g.eval();
}