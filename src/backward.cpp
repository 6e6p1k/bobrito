#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include "backward.h"
#include <vector>
#include "node_helpers.h"
#include "node.h"

using namespace std;

void set_grad_and_travers(std::shared_ptr<Node> fwd_node, std::shared_ptr<Node> new_grad, int param_count) {
    if (fwd_node->grad_node) {
        // chagne old labels
        fwd_node->grad_node->label = fwd_node->label + ".partial_grad";
        new_grad->label = fwd_node->label + ".partial_grad";
        // accumulate grad
        fwd_node->grad_node = add(fwd_node->grad_node, new_grad);
        fwd_node->grad_node->label = fwd_node->label + ".grad";
    } else {
        fwd_node->grad_node = new_grad;
        build_backward(fwd_node, std::static_pointer_cast<InternalNode>(new_grad), param_count);
    }
}

void build_backward(shared_ptr<Node> fwd_node, shared_ptr<InternalNode> bwd_node, int param_count) {
    
    if (fwd_node->kind == NodeKind::Source) {
        SourceNode* snode = static_cast<SourceNode*>(fwd_node.get());
        if (!snode->h_requiers_grad) return;
        param_count--;
        // accumulate grad into snode's grad buffer
        return;
    }

    shared_ptr<InternalNode> inode = dynamic_pointer_cast<InternalNode>(fwd_node);
    if (inode->param_count < 0) return;
    
    // cout<<"INSIDE BUILD_BACKWARD ----------------------------------\n";
    // cout<<"Op: "<<inode->op<<endl;

    switch (inode->op) {
        case OpType::Add:
            {
                shared_ptr<Node> A = inode->inputs[0];
                shared_ptr<Node> B = inode->inputs[1];
                // C = inode    (dC = gradient of C, which is in bwd_node)
                // C = A + B
                auto dC = bwd_node;
                dC->label = "d" + inode->label;

                // dC/dA = dC
                shared_ptr<Node> dC_dA = skip(dC);
                dC_dA->label = dC->label + " / d("+ A->label + ")";
                shared_ptr<Node> dA = reduce_to_shape(dC_dA, A->h_meta->shape(), A->h_meta->ndim);
                shared_ptr<InternalNode> bwd_a = static_pointer_cast<InternalNode>(dA);
                bwd_a->label = A->label + ".grad";
                set_grad_and_travers(A, bwd_a, param_count);

                
                // dC/dB = dC
                shared_ptr<Node> dC_dB = skip(dC);
                dC_dB->label = dC->label + " / d("+ B->label + ")";
                shared_ptr<Node> dB = reduce_to_shape(dC_dB, B->h_meta->shape(), B->h_meta->ndim);
                shared_ptr<InternalNode> bwd_b = static_pointer_cast<InternalNode>(dB);
                bwd_b->label = B->label + ".grad";
                set_grad_and_travers(B, bwd_b, param_count);
            }
            break;
            
        case OpType::Mul:
            {
                shared_ptr<Node> A = inode->inputs[0];
                shared_ptr<Node> B = inode->inputs[1];
                // C = inode    (dC = gradient of C, which is in bwd_node)
                // C = A * B
                auto dC = bwd_node;
                dC->label = "d" + inode->label;

                // dC/dA = dC * B
                shared_ptr<Node> B_bwd = copy_to_internal_node(B, false, OpType::BwdHelper);
                auto dC_dA = mul(dC, B_bwd);
                dC_dA->label = dC->label + " / d("+ A->label + ")";

                // Check if shapes match, otherwise out = (dC/dA).sum(dim=0, keepdims=false)
                shared_ptr<Node> dA = reduce_to_shape(dC_dA, A->h_meta->shape(), A->h_meta->ndim);
                shared_ptr<InternalNode> bwd_a = static_pointer_cast<InternalNode>(dA);
                bwd_a->label = A->label + ".grad";
                set_grad_and_travers(A, bwd_a, param_count);


                // dC/dB = A * dC
                auto A_bwd = copy_to_internal_node(A, false, OpType::BwdHelper);
                auto dC_dB = mul(A_bwd, dC);
                dC_dB->label = dC->label + " / d("+ B->label + ")";

                // Check if shapes match, otherwise dC = (dC/dB).sum(dim=0, keepdims=false)
                shared_ptr<Node> dB = reduce_to_shape(dC_dB, B->h_meta->shape(), B->h_meta->ndim);
                shared_ptr<InternalNode> bwd_b = static_pointer_cast<InternalNode>(dB);
                bwd_b->label = B->label + ".grad";
                set_grad_and_travers(B, bwd_b, param_count);
            }
            break;

        case OpType::MatMul:
            {
                shared_ptr<Node> A = inode->inputs[0];
                shared_ptr<Node> B = inode->inputs[1];
                // C = inode    (dC = gradient of C, which is in bwd_node)
                // C = A @ B
                shared_ptr<Node> dC = bwd_node;
                dC->label = "d" + inode->label;

                // dC/dA = dC @ B_T
                shared_ptr<Node> B_T = copy_to_internal_node(B, true, OpType::BwdHelper);
                shared_ptr<Node> dC_dA = matmul(dC, B_T);
                B_T->label = B_T->label + ".T";
                dC_dA->label = dC->label + " / d("+ A->label + ")";

                // Check if shapes match, otherwise out = (dC/dA).sum(dim=0, keepdims=false)
                shared_ptr<Node> dA = reduce_to_shape(dC_dA, A->h_meta->shape(), A->h_meta->ndim);
                shared_ptr<InternalNode> bwd_a = static_pointer_cast<InternalNode>(dA);
                bwd_a->label = A->label + ".grad";
                set_grad_and_travers(A, bwd_a, param_count);


                // dC/dB = A_T @ dC
                shared_ptr<Node> A_T = copy_to_internal_node(A, true, OpType::BwdHelper);
                shared_ptr<Node> dC_dB = matmul(A_T, dC);
                A_T->label = A_T->label + ".T";
                dC_dB->label = dC->label + " / d("+ B->label + ")";

                // Check if shapes match, otherwise dC = (dC/dB).sum(dim=0, keepdims=false)
                shared_ptr<Node> dB = reduce_to_shape(dC_dB, B->h_meta->shape(), B->h_meta->ndim);
                shared_ptr<InternalNode> bwd_b = static_pointer_cast<InternalNode>(dB);
                bwd_b->label = B->label + ".grad";
                set_grad_and_travers(B, bwd_b, param_count);
            }
            break;

        case OpType::SumLD:
            {
                shared_ptr<Node> A = inode->inputs[0];
                // C = sum_last_dim(A)
                // dC/dA = expand_last_dim(dC) (where dC = bwd_node)
                auto dC = bwd_node;
                dC->label = "d" + inode->label;

                shared_ptr<Node> dA = expand_last_dim(dC, A->h_meta->shape(), A->h_meta->ndim);
                shared_ptr<InternalNode> bwd_a = static_pointer_cast<InternalNode>(dA);
                bwd_a->label = A->label + ".grad";
                set_grad_and_travers(A, bwd_a, param_count);
            }
            break;

        case OpType::Reshape:
            {
                shared_ptr<Node> A = inode->inputs[0];
                // C = reshape(A, dC.shape)
                // dC/dA = reshape(dC, A.shape) (where dC = bwd_node)
                auto dC = bwd_node;
                dC->label = "d" + inode->label;

                shared_ptr<Node> dA = reshape(dC, A->h_meta->shape(), A->h_meta->ndim);
                shared_ptr<InternalNode> bwd_a = static_pointer_cast<InternalNode>(dA);
                bwd_a->label = A->label + ".grad";
                set_grad_and_travers(A, bwd_a, param_count);
            }
            break;
        case OpType::Contiguous:
            {
                shared_ptr<Node> A = inode->inputs[0];
                // C = contiguous(A)
                // dC/dA = contiguous(dC) (where dC = bwd_node)
                auto dC = bwd_node;
                dC->label = "d" + inode->label;

                shared_ptr<Node> dA = contiguous(dC);
                shared_ptr<InternalNode> bwd_a = static_pointer_cast<InternalNode>(dA);
                bwd_a->label = A->label + ".grad";
                set_grad_and_travers(A, bwd_a, param_count);
            }
            break;
        case OpType::ReLU:
            {
                shared_ptr<Node> A = inode->inputs[0];
                // C = relu(A)
                // dC/dA = bwd_relu(dC) (where dC = bwd_node)
                auto dC = bwd_node;
                dC->label = "d" + inode->label;

                shared_ptr<Node> dA = relu(dC);
                shared_ptr<InternalNode> bwd_a = static_pointer_cast<InternalNode>(dA);
                auto A_copy = copy_to_internal_node(A, false, OpType::BwdHelper);
                bwd_a->num_inputs = 2;
                bwd_a->inputs[1] = A_copy;
                bwd_a->op = OpType::BwdReLU;

                set_grad_and_travers(A, bwd_a, param_count);
            }
            break;
        default:
            break;
    }
}

void _backward(shared_ptr<Node> node)
{
    // post_order_dfs(i4.get(), visited, order);
    if (node->kind == NodeKind::Source) throw logic_error("Cannot use .backward() on this tensor.");

    // construct cfg 
    InodeConfig cfg;
    cfg.op = OpType::BwdStart;
    cfg.src = node;
    cfg.num_inputs = 1;
    std::shared_ptr<Node> inputs[] = {node};
    cfg.inputs = inputs;
    cfg.label = "BWD GRAPH START";

    shared_ptr<InternalNode> first_grad_node = copy_to_internal_node(cfg);
    node->grad_node = first_grad_node; // new node is the grad of node
    build_backward(node, first_grad_node, first_grad_node->param_count);
}


void backward_pass(shared_ptr<Node> node)
{
    unordered_set<Node*> visited;
    vector<InternalNode*> inodes;
    vector<SourceNode*> snodes;
    _build_topo(node.get(), visited, inodes, snodes);

    cout<<"Overview:"<<endl;
    for (int i = 0; i < snodes.size(); ++i)
    {
        cout<<"************************************"<<endl;
        SourceNode* snode = snodes[i];
        cout<<"Source Node: "<<snode->label<<endl;
    } 

    for (int i = inodes.size() - 1; i >=0; --i)
    {
        InternalNode* inode = inodes[i];
        cout<<"************************************"<<endl;
        cout<<"Current op: "<<inode->op<<endl;
        cout<<"Inputs num: "<<inode->num_inputs<<endl;
        cout<<"Current expr: "<<inode->label<<endl;
    }

    for (int i = inodes.size() - 1; i >=0; --i)
    {
        InternalNode* inode = inodes[i];
        for (int i = 0; i < inode->num_inputs; i++)
        {
            SourceNode* curr = snodes.back();
            if (snodes.back()->label == inode->inputs[i]->label)
            {
                snodes.pop_back();

            }
        }
        cout<<"-----------------------------------"<<endl;
        cout<<"Current op: "<<inode->op<<endl;
        cout<<"Inputs num: "<<inode->num_inputs<<endl;
        cout<<"Current expr: "<<inode->label<<endl;
    }
}
