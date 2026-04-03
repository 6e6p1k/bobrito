#include "node.h"
#include "graph_viz.h"
#include "backward.h"
#include "allocator.cuh"
#include "ops.cuh"
#include "utils.h"
#include "ops.cuh"
#include "visualize_graph.h"


// int _main(void)
// {   
//     int A_shape[] =  {2, 3, 4};
//     auto A = qq::fill(3, A_shape, 3, true);
//     A->label = "A";
//     print_node(A.get());

//     auto B = contiguous(A);

//     eval(B);
//     export_dot_full(A.get(), "parovozik.dot");
//     print_node(B.get());
//     return 0;
// }

int main(void)
{   
    int A_shape[] =  {2, 3, 4};
    auto A = qq::fill(3, A_shape, 3, true);
    A->label = "A";

    int B_shape[] =  {2, 3, 4};
    auto B = qq::fill(3, A_shape, 4, true);
    B->label = "B";
    
    auto C = add(A, B);

    auto D = sum(C, 1, true);
    eval(D);
    export_dot_full(D.get(), "parovozik.dot");
    print_node(C.get());
    print_node(D.get());
}