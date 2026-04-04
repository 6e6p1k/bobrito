#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include "backward.h"
#include <vector>

#include "node.h"

using namespace std;

void build_backward(shared_ptr<Node> fwd_node, shared_ptr<InternalNode> bwd_node, int param_count) {
    
    if (fwd_node->kind == NodeKind::Source) {
        SourceNode* snode = static_cast<SourceNode*>(fwd_node.get());
        if (!snode->h_requiers_grad) return;
        param_count--;
        // accumulate grad into snode's grad buffer
        return;
    }

    shared_ptr<InternalNode> inode = dynamic_pointer_cast<InternalNode>(fwd_node);
    if (param_count < 0) return;
    
    switch (inode->op) {
        case OpType::Add:
            {
                auto bwd_a = gcopy_to_internal_node(inode->inputs[0]);
                bwd_node->outputs[0] = bwd_a; // connect via outputs
                bwd_a->num_inputs = 1;
                bwd_a->inputs[0] = bwd_node; // connect via inputs
                bwd_a->op = OpType::Skip;
                bwd_a->param_count = param_count;
                build_backward(inode->inputs[0], bwd_a, param_count);
                
                auto bwd_b = gcopy_to_internal_node(inode->inputs[1]);
                bwd_node->outputs[1] = bwd_b; // connect via outputs
                bwd_b->num_inputs = 1;
                bwd_b->inputs[0] = bwd_node; // connect via inputs
                bwd_b->op = OpType::Skip;
                bwd_b->param_count = param_count;
                build_backward(inode->inputs[1], bwd_b, param_count);
            }
            break;
            
        case OpType::Mul:
            {
                // A = inode->inputs[0]
                // B = innode->inputs[1]
                // C = inode    (dC = gradient of C, which is in bwd_node)
                // C = A * B
                auto dC = bwd_node;

                // dC/dA = dC * B
                // CHECK if shapes match, otherwise out = (dC/dA).sum(dim=0, keepdims=false)
                auto bwd_a = gcopy_to_internal_node(inode->inputs[0]);
                auto B = copy_to_internal_node(inode->inputs[1], false, false, OpType::BwdHelper);
                auto dC_dA = mul(dC, B);
                if (dC_dA->h_meta->ndim != bwd_a->h_meta->ndim) bwd_a->op = OpType::Sum;
                else bwd_a->op = OpType::Skip;
                bwd_node->outputs[0] = bwd_a;
                bwd_a->num_inputs = 1;
                bwd_a->inputs[0] = dC_dA;
                bwd_a->param_count = param_count;
                build_backward(inode->inputs[0], bwd_a, param_count);

                // dC/dB = A * dC
                // CHECK if shapes match, otherwise dC = (dC/dB).sum(dim=0, keepdims=false)
                auto bwd_b = gcopy_to_internal_node(inode->inputs[1]);
                auto A = copy_to_internal_node(inode->inputs[0], false, false, OpType::BwdHelper);
                auto dC_dB = mul(A, dC);
                if (dC_dB->h_meta->ndim != bwd_b->h_meta->ndim) bwd_b->op = OpType::Sum; // assuming that sum sees the out shape and like ah i get it
                else bwd_b->op = OpType::Skip;
                bwd_node->outputs[1] = bwd_b;
                bwd_b->num_inputs = 1;
                bwd_b->inputs[0] = dC_dB; // needs transpose
                bwd_b->param_count = param_count;
                build_backward(inode->inputs[1], bwd_b, param_count);
            }
            break;

        case OpType::MatMul:
            {
                // A = inode->inputs[0]
                // B = innode->inputs[1]
                // C = inode    (dC = gradient of C, which is in bwd_node)
                // C = A @ B
                shared_ptr<Node> dC = bwd_node;
                dC->label = "dC";

                // dC/dA = dC @ B_T
                shared_ptr<Node> B_T = copy_to_internal_node(inode->inputs[1], false, true, OpType::BwdHelper);
                shared_ptr<Node> dC_dA = matmul(dC, B_T);
                shared_ptr<BwdNode> bwd_a;
                B_T->label = "B_T";
                dC_dA->label = "dC_dA";

                // CHECK if shapes match, otherwise out = (dC/dA).sum(dim=0, keepdims=false)
                if (dC_dA->h_meta->ndim == inode->inputs[0]->h_meta->ndim) {
                    bwd_a = internal_to_bwd(static_pointer_cast<InternalNode>(dC_dA));
                } else {
                    shared_ptr<Node> node = sum(dC_dA, 0, false);
                    bwd_a = internal_to_bwd(static_pointer_cast<InternalNode>(node));
                }
                inode->inputs[0]->d_grad = bwd_a->d_data;
                cout<<"inode->inputs[0]->label="<<inode->inputs[0]->label<<endl;
                bwd_a->label = inode->inputs[0]->label + "_grad";
                build_backward(inode->inputs[0], bwd_a, param_count);


                // dC/dB = A_T @ dC
                shared_ptr<Node> A_T = copy_to_internal_node(inode->inputs[0], false, true, OpType::BwdHelper);
                shared_ptr<Node> dC_dB = matmul(A_T, dC);
                shared_ptr<BwdNode> bwd_b;
                A_T->label = "A_T";
                dC_dB->label = "dC_dB";

                // CHECK if shapes match, otherwise dC = (dC/dB).sum(dim=0, keepdims=false)
                if (dC_dB->h_meta->ndim == inode->inputs[1]->h_meta->ndim) {
                    bwd_b = internal_to_bwd(static_pointer_cast<InternalNode>(dC_dB));    
                } else {
                    shared_ptr<Node> node = sum(dC_dB, 0, false);
                    bwd_b = internal_to_bwd(static_pointer_cast<InternalNode>(node));
                }
                inode->inputs[1]->d_grad = bwd_b->d_data;
                cout<<"inode->inputs[1]->label="<<inode->inputs[1]->label<<endl;
                bwd_b->label = inode->inputs[1]->label + "_grad";
                build_backward(inode->inputs[1], bwd_b, param_count);
            }
            break;

        case OpType::Sum:
            {
                shared_ptr<Node> node = expand_last_dim(bwd_node, inode->inputs[0]->h_meta->shape(), inode->inputs[0]->h_meta->ndim);
                shared_ptr<InternalNode> bwd_a = static_pointer_cast<InternalNode>(node);
                build_backward(inode->inputs[0], bwd_a, param_count);
            }
            break;

        case OpType::Reshape:
            {
                cout<<"HUUUUUY!"<<endl;
                shared_ptr<Node> node = reshape(bwd_node, inode->inputs[0]->h_meta->shape(), inode->inputs[0]->h_meta->ndim);
                shared_ptr<InternalNode> bwd_a = static_pointer_cast<InternalNode>(node);
                build_backward(inode->inputs[0], bwd_a, param_count);
            }
            break;
        case OpType::Contiguous:
            {
                cout<<"HUUUUUY!"<<endl;
                shared_ptr<Node> node = contiguous(bwd_node);
                shared_ptr<InternalNode> bwd_a = static_pointer_cast<InternalNode>(node);
                // cout<<"bwd_a->inputs[0]->label in build_backward contiguous: "<<bwd_a->outputs[0]->label<<endl;
                build_backward(inode->inputs[0], bwd_a, param_count);
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

    shared_ptr<BwdNode> bwd_node = copy_to_bwd_node(cfg);
    cout<<"bwd_node->label="<<bwd_node->label<<endl;
    cout<<"bwd_node->op="<<bwd_node->op<<endl;
    build_backward(node, bwd_node, bwd_node->param_count);
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
