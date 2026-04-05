
#include "node.h"
#include "backward.h"
#include "ops.cuh"
#include "utils.h"


class Tensor {
public:
    std::shared_ptr<Node> node;

    Tensor(std::vector<int> shape, bool requires_grad = false) {
        int ndim = shape.size();
        auto snode = std::make_shared<SourceNode>();
        snode->kind = NodeKind::Source;
        snode->h_meta = make_shape_data(ndim, shape.data());
        snode->h_requiers_grad = requires_grad;
        if (requires_grad) snode->param_count = 1;
        snode->d_data = std::make_shared<ContentData>();
        int size = utils::numel(ndim, shape.data());
        snode->d_data->allocate(size);
        node = snode;
    }

    Tensor(std::shared_ptr<Node> n) : node(n) {}

    void fill(float val) {
        cu_ops::fill(node.get(), val);
    }

    Tensor matmul(const Tensor& other) {
        return Tensor(::matmul(node, other.node));
    }

    Tensor operator+(const Tensor& other) {
        return Tensor(::add(node, other.node));
    }

    Tensor operator*(const Tensor& other) {
        return Tensor(::mul(node, other.node));
    }

    Tensor relu() {
        return Tensor(::relu(node));
    }

    Tensor sum(int dim) {
        return Tensor(::sum(node, dim));
    }

    void evaluate(void) {
        eval(node);
    }

    void geval() {
        _geval(node);
    }

    void build_backward(void) {
        _backward(node);
    }

    std::string get_label() {
        return node->label;
    }

    void set_label(std::string label) {
        node->label = label;
    }
};


int main(void)
{
    int batch = 64;
    int in_dim = 784;
    int hidden = 256;
    int out_dim = 10;

    Tensor x({batch, in_dim}, true); x.fill(0.5);
    Tensor W1({in_dim, hidden}, true); W1.fill(0.01);
    Tensor b1({1, hidden}, true); b1.fill(0.0);
    Tensor W2({hidden, out_dim}, true); W2.fill(0.01);
    Tensor b2({1, out_dim}, true); b2.fill(0.0);

    for (int i = 0; i < 1; i++) {
        std::cout << "iter " << i << std::endl;
        auto h = (x.matmul(W1) + b1).relu();
        auto out = h.matmul(W2) + b2;
        auto loss = out.sum(0);
        loss.build_backward();
        loss.evaluate();
        loss.geval();
        print_node(loss.node.get());
    }
    return 0;
}

// nvcc -std=c++17 -I.. tensor.cu ../ops.cu ../allocator.cu ../kernels.cu ../utils.cpp ../node.cpp ../backward.cpp ../graph_viz.cpp ../visualize_graph.cpp -o debug_test -lcublas && ./debug_test