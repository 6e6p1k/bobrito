#include "node.h"
#include "graph_viz.h"
#include "backward.h"
#include "allocator.cuh"
#include "ops.cuh"
#include "utils.h"
#include "ops.cuh"
#include "visualize_graph.h"
#include <memory>

using namespace std;

// CONTIGUOUS TEST
// int main(void)
// {   
//     int A_shape[] =  {2, 3, 4};
//     auto A = qq::fill(3, A_shape, 3, true);
//     A->label = "A";
//     print_node(A.get());

//     auto B = contiguous(A);

//     _backward(B);
//     eval(B);
    

//     export_dot_full(A.get(), "parovozik.dot");
//     print_node(B.get());
//     return 0;
// }

// SUM TEST
// int main(void)
// {   
//     int A_shape[] =  {2, 3, 4};
//     auto A = qq::fill(3, A_shape, 3, true);
//     A->label = "A";

//     int B_shape[] =  {2, 3, 4};
//     auto B = qq::fill(3, A_shape, 4, true);
//     B->label = "B";
    
//     auto C = add(A, B);

//     auto D = sum(C, 1, true);
//     eval(D);
//     export_dot_full(D.get(), "parovozik.dot");
//     print_node(C.get());
//     print_node(D.get());
// }

// MATMUL TEST
int main(void)
{   
    int A_shape[] =  {3, 4};
    auto A = qq::fill(2, A_shape, 3, true);
    A->label = "A";
    print_node(A.get());

    int B_shape[] =  {4, 3};
    auto B = qq::fill(2, B_shape, 1, true);
    B->label = "B";
    print_node(B.get());

    auto C = matmul(A, B);
    // auto D = sum(C, 0);
    
    _backward(C);
    
    export_dot_full(C.get(), "parovozik.dot");

    eval(C);
    _geval(C);
   
    cout<<"A: "<<endl;
    print_node(A.get());
    cout<<"B: "<<endl;
    print_node(B.get());
    cout<<"C: "<<endl;
    print_node(C.get());
    // cout<<"D: "<<endl;
    // print_node(D.get());
    // C = A @ B    | ([2, 3, 3] = [2, 3, 4] @ [4, 3])
    // A_g = ones @ B.T
}


// int main(void)
// {   
//     int A_shape[] =  {3, 4};
//     auto A = qq::fill(2, A_shape, 3, true);
//     A->label = "A";
//     print_node(A.get());

//     int B_shape[] =  {4, 3};
//     auto B = qq::fill(2, B_shape, 1, true);
//     B->label = "B";
//     print_node(B.get());

//     auto C = matmul(A, B);
    
//     _backward(C);
//     eval(C);
//     geval(C);
//     export_dot_full(C.get(), "parovozik.dot");

//     cout<<"A: "<<endl;
//     print_node(A.get());
//     cout<<"B: "<<endl;
//     print_node(B.get());
//     cout<<"C: "<<endl;
//     print_node(C.get());
// }


// // GENERAL
// int main(void)
// {   
//     int A_shape[] =  {2, 3, 4};
//     auto A = qq::fill(3, A_shape, 3, true);
//     A->label = "A";

//     int B_shape[] =  {4, 3};
//     auto B = qq::fill(2, B_shape, 1, true);
//     B->label = "B";

//     std::shared_ptr<ContentData> content_data = std::make_shared<ContentData>();
//     std::shared_ptr<InternalNode> Q = std::make_shared<InternalNode>();
//     Q->d_content = content_data;
    
//     auto C = matmul(A, B);

//     eval(C);

//     export_dot_full(C.get(), "parovozik.dot");

    
//     print_node(C.get());
//     std::cout<<"Q->d_content == C->d_content: "<< ((Q->d_content) == (C->d_content)) <<std::endl;
//     // C = A @ B    | ([2, 3, 3] = [2, 3, 4] @ [4, 3])
//     // A_g = ones @ B.T

// }

