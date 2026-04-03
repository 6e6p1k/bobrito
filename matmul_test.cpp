#include "node.h"
#include "graph_viz.h"
#include "backward.h"
#include "allocator.cuh"
#include "ops.cuh"
#include "utils.h"
#include "visualize_graph.h"

using namespace std;

int main_(void) {
    // f = (n1 * n2 + n3) * n4

    // (128, 28*28)
    int s1[] = {128, 512};
    // auto n1 = make_source_node(2, s1, true);
    auto n1 = qq::fill(2, s1, 3.0, true);
    n1->label = "N1";

    // (1, 512)
    int s2[] = {1, 512};
    auto n2 = qq::fill(2, s2, 1.0, true);
    n2->label = "N2";

    // (1, 512)
    int s3[] = {1, 512};
    auto n3 = qq::fill(2, s3, 3.0, true);
    n3->label = "N3";

    //
    int s4[] = {512, 128, 512};
    auto n4 = qq::fill(3, s4, 3.0, true);
    n4->label = "N4";

    // auto i1 = sum(n1, -1, true);
    auto i2 = mul(n2, n1); i2->label="i2";
    auto i3 = add(n3, i2); i3->label="i3";
    auto i4 = mul(n4, i3); i4->label="i4";


    _backward(i4);

    // auto _buff = i4->flipped;
    // i4->d_content->allocate(utils::numel(i4->h_meta->ndim, i4->h_meta->shape()));
    // i4->flipped = true;
    // cu_ops::fill(i4.get(), 1.0f);
    // i4->flipped = _buff;

    eval(i4);
    geval(i4);
    cout<<"Node n4: "<<endl;
    print_node(n4.get());
    print_all_nodes_grads({n1.get(), n2.get(), n3.get() ,n4.get()});
    print_all_nodes_grads({i2.get(), i3.get(), i4.get()});
    export_dot_full(i4.get(), "parovozik");    

    // cout<<"i4 inputs[1]: "<<q->inputs[1]->label<<endl;

    // print_all_nodes_labels({n1.get(), n2.get(), n3.get() ,n4.get()});
    // print_all_nodes_labels({i2.get(), i3.get(), i4.get()});
    // auto next = i4->outputs[0]->outputs[0];
    // cout<<"HUUUUUY: "<<next->label<<endl;
    // if (next->kind == NodeKind::Internal) 
    // {
    //     auto _next = dynamic_pointer_cast<InternalNode>(next);
    //     cout<<"_next->label="<<_next->label<<endl;
    //     for (int i = 0; i < _next->num_inputs; ++i) 
    //     {
    //         if (_next->inputs[i] == nullptr) cout<<"_next->inputs["<<i<<"] = nulltpr"<<endl;
    //         else cout<<"label: "<<_next->inputs[i]->label<<endl;
    //     }
    // } else { cout<<"next->kind=Source"<<endl;}

    // auto topo_lst = topo_sort(q);
    // auto dfs_lst = get_roots(q);

    // cout<<"DFS size: "<<dfs_lst.size()<<endl;
    // cout<<"TOPO size: "<<topo_lst.size()<<endl;
    // for (int i = 0; i < min(topo_lst.size(), dfs_lst.size()); ++i)
    // {
    //     cout<<"DFS ["<<i<<"]: "<<dfs_lst[i]->label<<endl;
    //     cout<<"TOPO ["<<i<<"]: "<<topo_lst[i]->label<<endl;
    // }




    // print_internal_node(q.get());
    return 0;
}

// visialize + export
// g++ -std=c++17 -o visual visualize_graph.cpp utils.cpp grad.cpp graph_viz.cpp allocator_stub.cpp 2>&1 && ./visual
// dot -Tpdf graph.dot -o graph.pdf

// just export to pdf:
// g++ -std=c++17 -o visual visualize_graph.cpp utils.cpp grad.cpp graph_viz.cpp allocator_stub.cpp 2>&1 && dot -Tpdf graph.dot -o graph.pdf
 
// useful things:
    // export_dot_backward(bwd_node, "turbofile");
    // unordered_set<Node*> visited;
    // vector<Node*> order;
    // post_order_dfs(i4.get(), visited, order);
    // visualize(i4); // visualize forward
    // export_dot(i4, "graph.dot"); // export to .dot
    // backward_viz(bwd_node); // visualize backward


int main(void)
{   
    int A_shape[] =  {2, 3, 4};
    auto A = qq::fill(3, A_shape, 3, true);
    A->label = "A";

    int B_shape[] =  {4, 3};
    auto B = qq::fill(2, B_shape, 1, true);
    B->label = "B";

    auto C = matmul(A, B);
    eval(C);
    _backward(C);
    // geval(C);
    export_dot_full(C.get(), "parovozik.dot");
    // print_node(A.get());
    // print_node(B.get());
    // C = A @ B    | ([2, 3, 3] = [2, 3, 4] @ [4, 3])
    // A_g = ones @ B.T

}