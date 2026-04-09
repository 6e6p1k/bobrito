#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include "backward.h"
#include <vector>
#include <unordered_map>
#include "node_helpers.h"
#include "node.h"

using namespace std;

void count_uses(shared_ptr<Node> node, unordered_map<Node*, int>& use_counts, unordered_set<Node*>& visited) {
    if (visited.count(node.get())) return;
    visited.insert(node.get());
    
    if (node->kind != NodeKind::Internal) return;
    auto inode = static_pointer_cast<InternalNode>(node);

    if (inode->op == OpType::MaxLD) return;  // don't count through max
    
    for (int i = 0; i < inode->num_inputs; i++) {
        use_counts[inode->inputs[i].get()]++;
        count_uses(inode->inputs[i], use_counts, visited);
    }
}

void set_grad_and_travers(shared_ptr<Node> fwd_node, shared_ptr<InternalNode> new_grad, unordered_map<Node*, int>& remaining) {

    shared_ptr<InternalNode> final_grad;
    if (fwd_node->grad_buffer) final_grad = static_pointer_cast<InternalNode>(add(fwd_node->grad_buffer, new_grad));
    else final_grad = new_grad;
    
    remaining[fwd_node.get()]--;
    
    if (fwd_node->kind == NodeKind::Source) {
        auto snode = static_cast<SourceNode*>(fwd_node.get());
        if (snode->h_requiers_grad) snode->owned_grad = final_grad;
    }
    // only continue when all upstream grads have arrived

    if (remaining[fwd_node.get()] > 0) fwd_node->grad_buffer = new_grad;
    else if (remaining[fwd_node.get()] <= 0) 
    {
        build_backward(fwd_node, final_grad, remaining);
        fwd_node->grad_buffer = nullptr;
    }
}

void build_backward(shared_ptr<Node> fwd_node, shared_ptr<InternalNode> bwd_node, unordered_map<Node*, int>& remaining) {
    
    if (fwd_node->kind == NodeKind::Source) return;

    shared_ptr<InternalNode> inode = dynamic_pointer_cast<InternalNode>(fwd_node);
    
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
                set_grad_and_travers(A, bwd_a, remaining);

                
                // dC/dB = dC
                shared_ptr<Node> dC_dB = skip(dC);
                dC_dB->label = dC->label + " / d("+ B->label + ")";
                shared_ptr<Node> dB = reduce_to_shape(dC_dB, B->h_meta->shape(), B->h_meta->ndim);
                shared_ptr<InternalNode> bwd_b = static_pointer_cast<InternalNode>(dB);
                bwd_b->label = B->label + ".grad";
                set_grad_and_travers(B, bwd_b, remaining);
            }
            break;

        case OpType::Sub:
            {
                shared_ptr<Node> A = inode->inputs[0];
                shared_ptr<Node> B = inode->inputs[1];
                // C = A - B
                auto dC = bwd_node;
                dC->label = "d" + inode->label;

                // dA = dC
                shared_ptr<Node> dC_dA = skip(dC);
                dC_dA->label = dC->label + " / d(" + A->label + ")";
                shared_ptr<Node> dA = reduce_to_shape(dC_dA, A->h_meta->shape(), A->h_meta->ndim);
                shared_ptr<InternalNode> bwd_a = static_pointer_cast<InternalNode>(dA);
                bwd_a->label = A->label + ".grad";
                set_grad_and_travers(A, bwd_a, remaining);

                // dB = -dC  (negate via mul by -1... or: 0 - dC)
                shared_ptr<Node> neg_dC = neg(dC);
                neg_dC->label = dC->label + " / d(" + B->label + ")";
                shared_ptr<Node> dB = reduce_to_shape(neg_dC, B->h_meta->shape(), B->h_meta->ndim);
                shared_ptr<InternalNode> bwd_b = static_pointer_cast<InternalNode>(dB);
                bwd_b->label = B->label + ".grad";
                set_grad_and_travers(B, bwd_b, remaining);
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
                set_grad_and_travers(A, bwd_a, remaining);


                // dC/dB = A * dC
                auto A_bwd = copy_to_internal_node(A, false, OpType::BwdHelper);
                auto dC_dB = mul(A_bwd, dC);
                dC_dB->label = dC->label + " / d("+ B->label + ")";

                // Check if shapes match, otherwise dC = (dC/dB).sum(dim=0, keepdims=false)
                shared_ptr<Node> dB = reduce_to_shape(dC_dB, B->h_meta->shape(), B->h_meta->ndim);
                shared_ptr<InternalNode> bwd_b = static_pointer_cast<InternalNode>(dB);
                bwd_b->label = B->label + ".grad";
                set_grad_and_travers(B, bwd_b, remaining);
            }
            break;

        case OpType::Div:
            {
                shared_ptr<Node> A = inode->inputs[0];
                shared_ptr<Node> B = inode->inputs[1];
                // C = A / B
                auto dC = bwd_node;
                dC->label = "d" + inode->label;

                // dA = dC / B
                auto B_copy = copy_to_internal_node(B, false, OpType::BwdHelper);
                auto dC_dA = div(dC, B_copy);
                dC_dA->label = dC->label + " / d(" + A->label + ")";
                shared_ptr<Node> dA = reduce_to_shape(dC_dA, A->h_meta->shape(), A->h_meta->ndim);
                shared_ptr<InternalNode> bwd_a = static_pointer_cast<InternalNode>(dA);
                bwd_a->label = A->label + ".grad";
                set_grad_and_travers(A, bwd_a, remaining);

                // dB = -dC * A / (B * B)
                auto B_copy_2 = copy_to_internal_node(B, false, OpType::BwdHelper);
                auto A_copy = copy_to_internal_node(A, false, OpType::BwdHelper);
                auto B_sq = sqrt(B_copy_2);                // B * B
                auto dC_A = mul(dC, A_copy);                      // dC * A
                auto ratio = div(dC_A, B_sq);                     // dC * A / (B * B)
                auto neg_ratio = neg(ratio);                 // negate
                neg_ratio->label = dC->label + " / d(" + B->label + ")";
                shared_ptr<Node> dB = reduce_to_shape(neg_ratio, B->h_meta->shape(), B->h_meta->ndim);
                shared_ptr<InternalNode> bwd_b = static_pointer_cast<InternalNode>(dB);
                bwd_b->label = B->label + ".grad";
                set_grad_and_travers(B, bwd_b, remaining);
            }
            break;

        case OpType::Pow:
            {
                shared_ptr<Node> A = inode->inputs[0];
                shared_ptr<Node> B = inode->inputs[1];
                auto dC = bwd_node;
                dC->label = "d" + inode->label;

                // 3 inputs: [upstream_grad, mantissa (A), exponent (B)]
                auto A_copy = copy_to_internal_node(A, false, OpType::BwdHelper);
                auto B_copy = copy_to_internal_node(B, false, OpType::BwdHelper);
                shared_ptr<Node> inputs[] = {dC, A_copy, B_copy};
                auto bwd_a = make_internal_node(OpType::BwdPow, inputs, 3,
                                                A->h_meta->shape(), A->h_meta->ndim);
                bwd_a->label = A->label + ".grad";
                shared_ptr<Node> dA = reduce_to_shape(bwd_a, A->h_meta->shape(), A->h_meta->ndim);
                shared_ptr<InternalNode> bwd_a_reduced = static_pointer_cast<InternalNode>(dA);
                bwd_a_reduced->label = A->label + ".grad";
                set_grad_and_travers(A, bwd_a_reduced, remaining);
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
                set_grad_and_travers(A, bwd_a, remaining);


                // dC/dB = A_T @ dC
                shared_ptr<Node> A_T = copy_to_internal_node(A, true, OpType::BwdHelper);
                shared_ptr<Node> dC_dB = matmul(A_T, dC);
                A_T->label = A_T->label + ".T";
                dC_dB->label = dC->label + " / d("+ B->label + ")";

                // Check if shapes match, otherwise dC = (dC/dB).sum(dim=0, keepdims=false)
                shared_ptr<Node> dB = reduce_to_shape(dC_dB, B->h_meta->shape(), B->h_meta->ndim);
                shared_ptr<InternalNode> bwd_b = static_pointer_cast<InternalNode>(dB);
                bwd_b->label = B->label + ".grad";
                set_grad_and_travers(B, bwd_b, remaining);
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
                set_grad_and_travers(A, bwd_a, remaining);
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
                set_grad_and_travers(A, bwd_a, remaining);
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
                set_grad_and_travers(A, bwd_a, remaining);
            }
            break;

        case OpType::ReLU:
            {
                shared_ptr<Node> A = inode->inputs[0];
                // C = relu(A)
                // dC/dA = bwd_relu(dC) (where dC = bwd_node)
                auto dC = bwd_node;
                dC->label = "d" + inode->label;

                auto A_copy = copy_to_internal_node(A, false, OpType::BwdHelper);
                shared_ptr<Node> inputs[] = {dC, A_copy};
                auto bwd_a = make_internal_node(OpType::BwdReLU, inputs, 2, 
                                                A->h_meta->shape(), A->h_meta->ndim);
                bwd_a->label = A->label + ".grad";

                set_grad_and_travers(A, bwd_a, remaining);
            }
            break;
        case OpType::Log:
            {
                shared_ptr<Node> A = inode->inputs[0];
                // C = log(A)
                // dC/dA = dC / A (where dC = bwd_node)
                auto dC = bwd_node;
                dC->label = "d" + inode->label;
                
                auto A_copy = copy_to_internal_node(A, false, OpType::BwdHelper);
                auto dA = div(dC, A_copy);
                shared_ptr<InternalNode> bwd_a = static_pointer_cast<InternalNode>(dA);
                bwd_a->label = A->label + ".grad";

                set_grad_and_travers(A, bwd_a, remaining);
            }
            break;
        case OpType::Exp:
            {
                auto A = inode->inputs[0];
                // C = exp(A)
                // dC/dA = dC * exp(A) = dC * C
                auto dC = bwd_node;
                dC->label = "d" + inode->label;
                
                auto C_copy = copy_to_internal_node(std::static_pointer_cast<Node>(inode), false, OpType::BwdHelper);                auto dA = mul(dC, C_copy);
                shared_ptr<InternalNode> bwd_a = static_pointer_cast<InternalNode>(dA);
                bwd_a->label = A->label + ".grad";

                set_grad_and_travers(A, bwd_a, remaining);
            }
            break;

        case OpType::Neg:
            {
                shared_ptr<Node> A = inode->inputs[0];
                auto dC = bwd_node;
                dC->label = "d" + inode->label;

                auto dA = neg(dC);
                shared_ptr<InternalNode> bwd_a = static_pointer_cast<InternalNode>(dA);
                bwd_a->label = A->label + ".grad";
                set_grad_and_travers(A, bwd_a, remaining);
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
    first_grad_node->inputs[0] = nullptr;
    node->grad_buffer = first_grad_node; // new node is the grad of node
    node->grad_node = first_grad_node; // new node is the grad of node


    unordered_map<Node*, int> use_counts;
    unordered_set<Node*> visited;
    count_uses(node, use_counts, visited);

    // for (auto [node, uses] : use_counts)
    // {
    //     cout<<"Node: "<<node->label<<endl;
    //     cout<<"Used: "<<uses<<endl;
    // }

    build_backward(node, first_grad_node, use_counts);
    node->grad_buffer = nullptr;
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
