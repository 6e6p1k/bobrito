#include <bits/stdc++.h>

#include <iostream>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "allocator.cuh"
#include "nodes.h"
#include "ops.h"

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
    if (fwd_node->grad_buffer)
        {
            fwd_node->grad_buffer->set_label(fwd_node->label + "_partial_grad");
            final_grad = add(fwd_node->grad_buffer, new_grad);
        }
    else
        final_grad = new_grad;
    final_grad->set_label(fwd_node->label + "_grad");
    // final_grad->d_data = make_shared<ContentData>();

    remaining[fwd_node.get()]--;

    // if leaf node (only those can have requires_grad = true),
    // just store the grad
    if (fwd_node->requires_grad) {
        if (fwd_node->requires_grad) fwd_node->owned_grad = final_grad;
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
    if (fwd_node->inputs.size() == 0) return;
    if (!fwd_node->prim) return;

    // ask the op for raw gradients
    shared_ptr<Node> grads[4] = {};
    fwd_node->prim->vjp(fwd_node, bwd_node, grads);

    // wire up each input
    for (int i = 0; i < fwd_node->inputs.size(); i++) {
        if (!grads[i]) continue;

        auto input = fwd_node->inputs[i];

        // handle broadcasting: reduce grad to input's shape
        auto reduced =
            reduce_to_shape(grads[i], input->get_shape(), input->get_ndim());

        set_grad_and_traverse(input, reduced, remaining);
    }
}

// ============================================================
//                       EVAL HELPER
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
    if (visited.count(node)) return;
    visited.insert(node);
    // push before children (pre-order traversal)
    order.push_back(node); // bc we need to go from root node to leafs
    for (auto& out : node->outputs) build_topo_outputs(out.get(), visited, order);
}

void eval_helper(vector<Node*>& fwd_order, vector<Node*>& bwd_order) {
    for (Node* node : fwd_order) {
        if (!node->is_meta()) continue;
        int size = node->get_size();
        if (!node->is_data()) node->d_data = make_shared<ContentData>();
        if (!node->is_data_allocated()) node->d_data->allocate(size);
        node->prim->eval(node);
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
//                       PUBLIC API
// ============================================================

void backward(shared_ptr<Node> loss) {
    if (loss->requires_grad || loss->inputs.size() == 0)
        throw logic_error("Cannot call backward() on a source node.");

    // seed: gradient of loss w.r.t. itself = 1
    auto seed = full(loss->get_shape(), loss->get_ndim(), 1.0f);
    seed->set_label("BWD_START");
    loss->outputs.push_back(seed);

    // count how many consumers each node has
    unordered_map<Node*, int> uses;
    unordered_set<Node*> visited;
    count_uses(loss, uses, visited);

    // kick off
    build_backward(loss, seed, uses);
}

void eval(shared_ptr<Node> root) {
    vector<Node*> fwd_order;
    vector<Node*> bwd_order;
    if (root->inputs.size() > 0) {
        unordered_set<Node*> visited;
        build_topo_inputs(root.get(), visited, fwd_order);
    }
    if (root->outputs.size() == 1 && root->outputs[0]->label == "BWD_START") {
        unordered_set<Node*> visited;
        build_topo_outputs(root->outputs[0].get(), visited, bwd_order);
    }
    eval_helper(fwd_order, bwd_order);
}