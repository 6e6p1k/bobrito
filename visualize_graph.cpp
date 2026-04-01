#include "grad.h"
#include "graph_viz.h"
#include <algorithm>
#include <machine/limits.h>
#include <memory>


std::shared_ptr<SourceNode> make_source_node(int ndim, int* shape, bool requires_grad) {
    auto node = std::make_shared<SourceNode>();
    node->kind = NodeKind::Source;
    node->h_meta = make_shape_data(ndim, shape);
    node->h_requiers_grad = requires_grad;
    node->use_count++;
    return node;
}

void post_order_dfs(Node* node, unordered_set<Node*>& visited, vector<Node*>& order) {
    if (visited.count(node)) return;
    visited.insert(node);

    if (node->kind == NodeKind::Internal) {
        InternalNode* inode = static_cast<InternalNode*>(node);
        for (int i = 0; i < inode->num_inputs; i++)
            post_order_dfs(inode->inputs[i].get(), visited, order);
    }

    order.push_back(node);
}

int main(void) {
    // f = (n1.sum(-1) * n2 + n3) * (512, 128, 512)

    // (128, 28*28)
    int s1[] = {128, 512};
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
    auto n4 = make_source_node(3, s4, true);
    n4->label = "N4";

    // auto i1 = sum(n1, -1, true);
    auto i2 = mul(n2, n1);
    auto i3 = add(n3, i2);
    auto i4 = mul(n4, i3);

    // auto bwd_node = std::make_shared<InternalNode>();
    // auto node = static_cast<InternalNode*>(i4.get());
    // auto bwd_node = make_internal_node_like(node);
    // bwd_node->op = OpType::Empty;
    // bwd_node->label = "bwd_node";
    std::shared_ptr<InternalNode> q = std::dynamic_pointer_cast<InternalNode>(i4);
    build_backward(i4, q, 4);
    export_dot_full(i4.get(), "turbofile");
    std::shared_ptr<InternalNode> i4_out_1 = std::dynamic_pointer_cast<InternalNode>(q->outputs[0]);
    cout<<i4_out_1->label<<endl;
    cout<<i4_out_1->inputs[0]->label<<endl;
    // export_dot_backward(bwd_node, "turbofile");
    // unordered_set<Node*> visited;
    // vector<Node*> order;
    // post_order_dfs(i4.get(), visited, order);
    // visualize(i4); // visualize forward
    // export_dot(i4, "graph.dot"); // export to .dot
    // backward_viz(bwd_node); // visualize backward
    return 0;
}

// visialize + export
// g++ -std=c++17 -o visual visualize_graph.cpp utils.cpp grad.cpp graph_viz.cpp allocator_stub.cpp 2>&1 && ./visual
// dot -Tpdf graph.dot -o graph.pdf

// just export to pdf:
// g++ -std=c++17 -o visual visualize_graph.cpp utils.cpp grad.cpp graph_viz.cpp allocator_stub.cpp 2>&1 && dot -Tpdf graph.dot -o graph.pdf
