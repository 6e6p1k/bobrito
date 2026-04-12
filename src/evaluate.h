#include "nodes.h"
#include <memory>

struct Graph {
    std::shared_ptr<Node> root;
    std::vector<Node*> fwd_order;
    std::vector<Node*> bwd_order;
    std::vector<Node*> require_grad_nodes;

    bool disposable = true;
    std::vector<Node*> bwd_nodes;

    public: 
    Graph(std::shared_ptr<Node> loss, bool disposable = true);
    void backward();
    void eval();
    void sgd_step(float lr);
    void detach_grads_from_bwd_graph();

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
    void fwd_pass();
    void bwd_pass();
    void _backward();
};
std::shared_ptr<Graph> make_model(std::shared_ptr<Node> root);
void backward(std::shared_ptr<Node> loss);
void eval(std::shared_ptr<Node> root, bool export_graph = false);
