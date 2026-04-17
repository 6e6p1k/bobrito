#pragma once

#include "nodes.h"
#include <memory>
#include <vector>

struct Graph {
    std::shared_ptr<Node> root;
    std::vector<Node*> fwd_order;
    std::vector<Node*> bwd_order;
    std::vector<std::shared_ptr<Node>> require_grad_nodes;
    std::vector<Node*> intermediates;

    bool disposable = true;
    std::vector<Node*> bwd_nodes;

    public: 
    Graph(std::shared_ptr<Node> loss, bool disposable = true);
    void backward();
    void eval(bool debug = false);
    void detach_grads_from_bwd_graph();
    void fwd_bwd_pass();
    void collect_intermidiate_nodes();
    void zero_nodes_data();
    void fwd_pass();
    void bwd_pass();
    std::vector<std::shared_ptr<Node>> parameters();

    private:
    // CHECKS
    bool is_root();
    bool is_backward();
    bool need_backward();

    // FUNCS
    void compute_fwd_order();
    void compute_bwd_order();
    void locate_leaf_w_grad_nodes();
    void locate_bwd_nodes();
    void pass(std::vector<Node*> order);
    void debug_pass(std::vector<Node*> order);
    void _backward();
};
std::shared_ptr<Graph> make_model(std::shared_ptr<Node> root);
void backward(std::shared_ptr<Node> loss);
void eval(std::shared_ptr<Node> root, bool export_graph = false);
void debug_eval(std::shared_ptr<Node> root, bool export_graph, bool disposable);
