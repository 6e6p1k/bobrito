#include <cstddef>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "allocator.cuh"
#include "nodes.h"
#include "ops.h"
#include "evaluate.h"
#include "graph_viz.h"

using namespace std;

// ============================================================
//                       BACKWARD HELPER
// ============================================================
void build_backward(shared_ptr<Node> fwd_node, shared_ptr<Node> bwd_node,
                    unordered_map<Node*, int>& remaining,
                    unordered_map<Node*, shared_ptr<Node>>& grad_accum);

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
// grad_accum: temporary map holding partial gradient sums
//             for multi-consumer nodes (replaces the old grad_buffer field).
void set_grad_and_traverse(shared_ptr<Node> fwd_node, shared_ptr<Node> new_grad,
                           unordered_map<Node*, int>& remaining,
                           unordered_map<Node*, shared_ptr<Node>>& grad_accum) {
    // accumulate
    shared_ptr<Node> final_grad;
    auto it = grad_accum.find(fwd_node.get());
    if (it != grad_accum.end()) {
        it->second->set_label(fwd_node->label + "_partial_grad");
        final_grad = add(it->second, new_grad);
    } else
        final_grad = new_grad;
    final_grad->set_label(fwd_node->label + "_grad");

    remaining[fwd_node.get()]--;

    // if leaf node (only those can have requires_grad = true),
    // just store the grad
    if (fwd_node->requires_grad) {
        fwd_node->owned_grad = final_grad;
    }

    // not all upstream grads arrived yet — stash and wait
    if (remaining[fwd_node.get()] > 0) {
        grad_accum[fwd_node.get()] = final_grad;
        return;
    }

    // all grads arrived — clean up accumulator and recurse
    grad_accum.erase(fwd_node.get());
    build_backward(fwd_node, final_grad, remaining, grad_accum);
}

// The core: ask the primitive for raw grads, reduce for
// broadcasting, wire them up.
void build_backward(shared_ptr<Node> fwd_node, shared_ptr<Node> bwd_node,
                    unordered_map<Node*, int>& remaining,
                    unordered_map<Node*, shared_ptr<Node>>& grad_accum) {
    if (fwd_node->inputs.size() <= 0 || fwd_node->requires_grad) return;
    if (!fwd_node->prim) return;

    // ask the op for raw gradients
    shared_ptr<Node> grads[4] = {};
    fwd_node->prim->vjp(fwd_node, bwd_node, grads);

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

        set_grad_and_traverse(input, reduced, remaining, grad_accum);
    }

    // go through bwdnode-> outputs vector
    // and remove all the nullptrs
    auto& v = bwd_node->outputs;
    for (int i = v.size() - 1; i >= 0; --i) {
        if (v[i].expired()) {
            v.erase(v.begin() + i);
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

void Graph::eval(bool debug) {
    if (debug) debug_pass(fwd_order);
    else fwd_pass();

    if (need_backward()) {
        if (bwd_order.size() < 0) throw logic_error("Backward order is empty!");
        if (debug) debug_pass(bwd_order);
        else bwd_pass();
    }

    if (disposable)
        detach_grads_from_bwd_graph();
}

void Graph::fwd_bwd_pass() {
    fwd_pass();
    bwd_pass();
}

void Graph::fuse_fwd_order() {
    // compute fwd order uses
    unordered_map<Node*, int> uses;
    unordered_set<Node*> visited;
    count_uses(root, uses, visited);

    int i = fwd_order.size();
    while (i >= 0) {
        auto node = fwd_order[i];
        if (uses[node] == 1) {
            // TODO
        }
        i--;
    }
}

std::vector<std::shared_ptr<Node>> Graph::parameters() {
    return require_grad_nodes;
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
    for (auto leaf : require_grad_nodes) {
        if (leaf->owned_grad)
            build_topo_inputs_bwd(leaf->owned_grad.get(), visited, bwd_order);
    }
}

void Graph::locate_leaf_w_grad_nodes() {
    if (require_grad_nodes.size() > 0)
        throw logic_error("require_grad_nodes is already filled.");
    for (auto node : fwd_order) {
        if (!node || !node->is_meta()) continue;
        if (node->requires_grad) {
            require_grad_nodes.push_back(node->getPtr());
        }
    }
}

void Graph::pass(vector<Node*> order) {
    for (Node* node : order) {
        if (!node || !node->prim || !node->is_meta()) continue;
        node->prim->allocate(node);
        node->prim->eval(node);
    }
}

void Graph::debug_pass(vector<Node*> order) {
    if (order.size() == 0) cout << "<========ORDER IS EMPTY!======>\n";
    if (order[0] && order[0]->is_bwd) cout<<"\nINSIDE BWD PASS =================================>\n";
    if (order[0] && !order[0]->is_bwd) cout<<"\nINSIDE FWD PASS =================================>\n";
    for (Node* node : order) {
        if (!node || !node->prim || !node->is_meta()) continue;
        cout << "[" << node->prim->name() << "] " << node->label << endl;


        cout << "\tLABEL ========> " << (node)->label << endl;
        cout << "\tPRIMITIVE ====> "
                << ((node)->prim ? (node)->prim->name() : "<no prim>")
                << endl;
        if ((node)->is_meta()) {
            cout << "\tNDIM =========> " << (node)->get_ndim() << endl;
            cout << "\tSHAPE ========> [";
            for (int _i = 0; _i < (node)->get_ndim(); _i++)
                cout << (node)->get_shape()[_i]
                        << (_i < (node)->get_ndim() - 1 ? ", " : "");
            cout << "]" << endl;
        } else {
            cout << "\tNDIM =========> no meta" << endl;
            cout << "\tSHAPE ========> no meta" << endl;
        }

        auto _gn = (node)->owned_grad;
        cout << (_gn ? "\tGRAD =========> " + ptr_str(_gn.get())  + "."
                        : "\tGRAD =========> doesnt exist.")
                << endl;
        if (_gn)
            cout << (_gn->is_data_allocated()
                            ? "\tGRAD DATA ====> " + ptr_str(_gn->get_data()) + "."
                            : "\tGRAD DATA ====> is not allocated.")
                    << endl;

        cout << "\tINPUTS (" << (node)->inputs.size() << "):" << endl;
        for (auto& _in : (node)->inputs) {
            if (!_in)
                cout << "\t\t<nullptr input>" << endl;
            else
                cout << "\t\t[" << (_in->prim ? _in->prim->name() : "<no prim>")
                        << "] "
                        << (_in->label.empty() ? "<no label>" : _in->label)
                        << endl;
        }
        cout << "\tOUTPUTS (" << (node)->outputs.size() << "):" << endl;
        for (auto& _out : (node)->outputs) {
            auto _sp = _out.lock();
            if (!_sp)
                cout << "\t\t[expired]" << endl;
            else
                cout << "\t\t[" << (_sp->prim ? _sp->prim->name() : "<no prim>")
                        << "] "
                        << (_sp->label.empty() ? "<no label>" : _sp->label)
                        << endl;
        }

        node->prim->allocate(node);
        cout<<"After allocation:\n";
        cout << print_block(node->d_data->b_ptr);
        node->prim->eval(node);
        cout<<"After evaluation:\n";
        cout << print_block(node->d_data->b_ptr);
        if (order[order.size() - 1] == node) continue;
        cout<<"\n----------------------------------------------\n\n";

    }
    cout<<"INSIDE PASS =================================<\n\n";
}

void Graph::fwd_pass() { pass(fwd_order); }
void Graph::bwd_pass() { pass(bwd_order); }

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

    // temporary accumulator for multi-consumer gradient sums
    unordered_map<Node*, shared_ptr<Node>> grad_accum;
    build_backward(root, seed, uses, grad_accum);
}

bool check_node_before_detaching(shared_ptr<Node> node) {
    if (!node) return false;
    if (!node->is_meta()) return false;
    if (!node->is_data()) return false;
    if (!node->is_data_allocated()) return false;
    if (!node->prim) return false;
    return true;
}

std::shared_ptr<Node> detach_helper(shared_ptr<Node> node) {
    shared_ptr<Node> out_node = make_shared<Node>();
    out_node->h_meta = node->h_meta;
    out_node->d_data = node->d_data;
    out_node->is_bwd = node->is_bwd;
    out_node->set_label(node->label + ".copy()");
    return out_node;
}

void Graph::detach_grads_from_bwd_graph() {
    for (auto node : require_grad_nodes) {
        if (!node || !node->requires_grad || !node->owned_grad) continue;
        if (check_node_before_detaching(node->owned_grad))
            node->owned_grad = copy_node_no_prim_before_eval(node->owned_grad);
    }
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

void debug_eval(shared_ptr<Node> root, bool export_graph, bool disposable) {
    Graph g = Graph(root, disposable);
    g.backward();
    if (export_graph) root->export_graph();
    g.eval(true);
}
