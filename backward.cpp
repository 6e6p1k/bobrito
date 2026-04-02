#include <iostream>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include "backward.h"
#include <vector>

#include "node.h"


void build_backward(shared_ptr<Node> fwd_node, shared_ptr<InternalNode> bwd_node, int param_count) {
    
    if (fwd_node->kind == NodeKind::Source) {
        SourceNode* snode = static_cast<SourceNode*>(fwd_node.get());
        if (!snode->h_requiers_grad) return;
        param_count--;
        // accumulate grad into snode's grad buffer
        return;
    }

    shared_ptr<InternalNode> inode = dynamic_pointer_cast<InternalNode>(fwd_node);
    param_count--;
    if (param_count < 0) return;
    
    switch (inode->op) {
        case OpType::Add:
            {
                auto bwd_a = copy_to_internal_node(inode->inputs[0]);
                bwd_node->outputs[0] = bwd_a; // connect via outputs
                bwd_a->num_inputs = 1;
                bwd_a->inputs[0] = bwd_node; // connect via inputs
                bwd_a->op = OpType::Skip;
                bwd_a->param_count = param_count;
                build_backward(inode->inputs[0], bwd_a, param_count);
                
                auto bwd_b = copy_to_internal_node(inode->inputs[1]);
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
                auto bwd_a = copy_to_internal_node(inode->inputs[0]);
                bwd_a->op = OpType::Mul;
                bwd_node->outputs[0] = bwd_a;
                bwd_a->num_inputs = 2;
                bwd_a->inputs[0] = bwd_node;           // upstream gradient
                bwd_a->inputs[1] = copy_to_internal_node(inode->inputs[1]);   // the OTHER input (b)
                bwd_a->param_count = param_count;
                build_backward(inode->inputs[0], bwd_a, param_count);

                auto bwd_b = copy_to_internal_node(inode->inputs[1]);
                bwd_b->op = OpType::Mul;
                bwd_node->outputs[1] = bwd_b;
                bwd_b->num_inputs = 2;
                bwd_b->inputs[0] = bwd_node;
                bwd_b->inputs[1] = copy_to_internal_node(inode->inputs[0]);   // the OTHER input (b)
                
                // cout<<"bwd_b->inputs: "<<bwd_b->inputs[0]->label<<" and "<<bwd_b->inputs[1]->label<<endl;
                bwd_b->param_count = param_count;
                build_backward(inode->inputs[1], bwd_b, param_count);
            }
            break;

        case OpType::MatMul:
            {
                auto bwd_a = copy_to_internal_node(inode->inputs[0]);
                bwd_a->op = OpType::MatMul;
                bwd_node->outputs[0] = bwd_a;
                bwd_a->inputs[0] = bwd_node; // gradient
                bwd_a->inputs[1] = fwd_node; // needs transpose
                bwd_a->param_count = param_count;
                build_backward(inode->inputs[0], bwd_a, param_count);

                auto bwd_b = copy_to_internal_node(inode->inputs[1]);
                bwd_b->op = OpType::MatMul;
                bwd_node->outputs[1] = bwd_b;
                bwd_b->inputs[0] = fwd_node; // needs transpose
                bwd_b->inputs[1] = fwd_node; // gradient
                bwd_b->param_count = param_count;
                build_backward(inode->inputs[1], bwd_b, param_count);
            }
            break;

        case OpType::Sum:
            break;
            
        default:
            break;
    }
}

void _backward(shared_ptr<Node> node)
{
    // post_order_dfs(i4.get(), visited, order);
    if (node->kind == NodeKind::Source) 
    {
        throw logic_error("Cannot use .backward() on this tensor.");
    }
    shared_ptr<InternalNode> bwd_node = copy_to_internal_node(node);
    node->outputs[0] = bwd_node;
    bwd_node->num_inputs = 1;
    bwd_node->inputs[0] = node; 
    bwd_node->op = OpType::Empty;
    bwd_node->label = "BACKWARD GRAPH START";
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
