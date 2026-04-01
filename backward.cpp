#include <iostream>
#include <unordered_set>

#include "node.h"


void build_backward(shared_ptr<Node> fwd_node, shared_ptr<InternalNode> bwd_node, int use_count) {
    
    if (fwd_node->kind == NodeKind::Source) {
        SourceNode* snode = static_cast<SourceNode*>(fwd_node.get());
        if (!snode->h_requiers_grad) return;
        use_count--;
        // accumulate grad into snode's grad buffer
        return;
    }

    InternalNode* inode = static_cast<InternalNode*>(fwd_node.get());
    use_count--;
    if (use_count < 0) return;
    
    switch (inode->op) {
        case OpType::Add:
            {
                auto bwd_a = copy_to_internal_node(inode->inputs[0]);
                bwd_node->outputs[0] = bwd_a;
                bwd_a->op = OpType::Skip;
                bwd_a->inputs[0] = bwd_node;
                bwd_a->use_count = use_count;
                build_backward(inode->inputs[0], bwd_a, use_count);
                
                auto bwd_b = copy_to_internal_node(inode->inputs[1]);
                bwd_node->outputs[1] = bwd_b;
                bwd_b->op = OpType::Skip;
                bwd_b->inputs[0] = bwd_node;
                bwd_b->use_count = use_count;
                build_backward(inode->inputs[1], bwd_b, use_count);
            }
            break;
            
        case OpType::Mul:
            {
                auto bwd_a = copy_to_internal_node(inode->inputs[0]);
                bwd_node->outputs[0] = bwd_a;
                bwd_a->op = OpType::Mul;
                bwd_a->inputs[0] = bwd_node;
                bwd_a->inputs[1] = fwd_node;
                bwd_a->use_count = use_count;
                build_backward(inode->inputs[0], bwd_a, use_count);

                auto bwd_b = copy_to_internal_node(inode->inputs[1]);
                bwd_node->outputs[1] = bwd_b;
                bwd_b->op = OpType::Mul;
                bwd_b->inputs[0] = bwd_node;
                bwd_b->inputs[1] = fwd_node;
                bwd_b->use_count = use_count;
                build_backward(inode->inputs[1], bwd_b, use_count);
            }
            break;

        case OpType::MatMul:
            {
                auto bwd_a = copy_to_internal_node(inode->inputs[0]);
                bwd_node->outputs[0] = bwd_a;
                bwd_a->op = OpType::MatMul;
                bwd_a->inputs[0] = bwd_node; // gradient
                bwd_a->inputs[1] = fwd_node; // needs transpose
                bwd_a->use_count = use_count;
                build_backward(inode->inputs[0], bwd_a, use_count);

                auto bwd_b = copy_to_internal_node(inode->inputs[1]);
                bwd_node->outputs[1] = bwd_b;
                bwd_b->op = OpType::MatMul;
                bwd_b->inputs[0] = fwd_node; // needs transpose
                bwd_b->inputs[1] = fwd_node; // gradient
                bwd_b->use_count = use_count;
                build_backward(inode->inputs[1], bwd_b, use_count);
            }
            break;

        case OpType::Sum:
            break;
            
        default:
            break;
    }
}

// <Topological sort>
void build_topo(Node* node, unordered_set<Node*>& visited, vector<InternalNode*>& inodes, vector<SourceNode*>& snodes)
{
    if (visited.count(node)) return;
    visited.insert(node);

    if (node->kind == NodeKind::Internal) 
    {
        InternalNode* inode = static_cast<InternalNode*>(node);
        for (int i = 0; i < inode->num_inputs; i++)
            build_topo(inode->inputs[i].get(), visited, inodes, snodes);
        inodes.push_back(inode);
    }
    else {
        snodes.push_back(static_cast<SourceNode*>(node));
    }
    
}
// </Topological sort>


void backward_pass(shared_ptr<Node> node)
{
    unordered_set<Node*> visited;
    vector<InternalNode*> inodes;
    vector<SourceNode*> snodes;
    build_topo(node.get(), visited, inodes, snodes);

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
