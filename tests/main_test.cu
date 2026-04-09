#include "../src/tensor.cuh"
#include "../src/bobr.cuh"
#include "../src/node_helpers.h"
#include "../src/graph_viz.h"

int main(void)
{

    Tensor x({64, 784}, true); x.fill(2);
    x.set_label("x");

    Tensor power({64, 784}, false); power.fill(2);

    auto res = x.pow(power);
    res.evaluate();
    res.build_backward();
    export_dot_full(res.node.get(), "file.dot");
    print_node(res.node.get());
}

