#include "grad.h"
#include "graph_viz.h"
#include <memory>

std::shared_ptr<SourceNode> make_source_node(int ndim, int* shape, bool requires_grad) {
    auto node = std::make_shared<SourceNode>();
    node->kind = NodeKind::Source;
    node->h_meta = make_shape_data(ndim, shape);
    node->h_requiers_grad = requires_grad;
    node->use_count++;
    return node;
}

int main(void) {
    // f = (n1.sum(-1) * n2 + n3) * (512, 128, 512)

    // (128, 28*28)
    int s1[] = {128, 784};
    auto n1 = make_source_node(2, s1, true);
    n1->label = "N1";

    // (1, 512)
    int s2[] = {1, 512};
    auto n2 = make_source_node(2, s2, true);
    n2->label = "N2";

    // (1, 512)
    int s3[] = {1, 512};
    auto n3 = make_source_node(2, s3, true);
    n3->label = "N3";

    //
    int s4[] = {512, 128, 512};
    auto n4 = make_source_node(3, s4, false);
    n4->label = "N4";

    auto i1 = sum(n1, -1, true);
    auto i2 = mul(n2, i1);
    auto i3 = add(n3, i2);
    auto i4 = mul(n4, i3);

    // visualize(i4); // visualize forward
    export_dot(i4, "graph.dot"); // export to .dot
    // backward_viz(i4); // visualize backward
    return 0;
}

// visialize + export
// g++ -std=c++17 -o visual visualize_graph.cpp utils.cpp grad.cpp graph_viz.cpp allocator_stub.cpp 2>&1 && ./visual
// dot -Tpdf graph.dot -o graph.pdf

// just export to pdf:
// g++ -std=c++17 -o visual visualize_graph.cpp utils.cpp grad.cpp graph_viz.cpp allocator_stub.cpp 2>&1 && dot -Tpdf graph.dot -o graph.pdf
