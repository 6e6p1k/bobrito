#include <iostream>
#include <memory>
#include <unordered_set>
#include "allocator.cuh"
#include "node.h"
#include "backward.h"
#include <bits/stdc++.h>
#include "utils.h"
#include "optim.h"

using namespace std;

__global__ void nulify(float* data, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = 0.0f;
}

struct Graph 
{
    shared_ptr<InternalNode> root;

    vector<Node*> forward_order;
    vector<InternalNode*> backward_order;
    size_t storage_size;
    void* start_ptr;

    Optim optimizer;

    virtual void get_root_node();
    
    void build_graph()
    {
        _backward(root);
    }

    void training_step()
    {

    }

    private:
    
    void compute_orders()
    {
        unordered_set<Node*> tmp_1; 
        unordered_set<InternalNode*> tmp_2;

        build_topo(root.get(), tmp_1, forward_order);
        build_topo_grad(root.get(), tmp_2, backward_order);
        reverse(forward_order.begin(), forward_order.end());
    }

    void compute_memory_required()
    {
        for (Node* node : forward_order)
        {
            if (node->kind == NodeKind::Internal)
                if (filter_op(static_cast<InternalNode*>(node)->op)) continue;
            storage_size += sizeof(float) * utils::numel(node->h_meta->ndim, node->h_meta->shape());
        }

        for (InternalNode* inode : backward_order)
        {
            if (filter_op(inode->op)) continue;
            storage_size += sizeof(float) * utils::numel(inode->h_meta->ndim, inode->h_meta->shape());
        }
    }

    void allocate_memory_required()
    { cudaMalloc(&start_ptr, storage_size); }

    void free_memory()
    { cudaFree(start_ptr); }

    void assign_memory()
    { 
        void* curr_pos = start_ptr;
        for (Node* node : forward_order)
        {            
            if (node->kind == NodeKind::Internal)
                if (filter_op(static_cast<InternalNode*>(node)->op)) continue;

            Block* block;
            int size = utils::numel(node->h_meta->ndim, node->h_meta->shape());
            block->size = sizeof(float) * size;
            block->d_ptr = (float*)curr_pos;

            node->d_data = make_shared<ContentData>();
            node->d_data->_size = size;
            node->d_data->b_ptr = block;

            curr_pos = (void*)( (char*)curr_pos + block->size );
        }

        for (InternalNode* inode : backward_order)
        {
            if (filter_op(inode->op)) continue;
            Block* block;
            int size = utils::numel(inode->h_meta->ndim, inode->h_meta->shape());
            block->size = sizeof(float) * size;
            block->d_ptr = (float*)curr_pos;

            inode->d_data = make_shared<ContentData>();
            inode->d_data->_size = size;
            inode->d_data->b_ptr = block;

            curr_pos = (void*)( (char*)curr_pos + block->size );
        }
    }

    bool filter_op(OpType op)
    {
        switch (op)
        {
            case OpType::Add:           return true;
            case OpType::Sub:           return true;
            case OpType::Mul:           return true;
            case OpType::Div:           return true;
            case OpType::Pow:           return true;


            case OpType::MatMul:        return true;
            case OpType::Contiguous:    return true;

            case OpType::SumLD:         return true;
            case OpType::MaxLD:         return true;
            case OpType::MinLD:         return true;
            case OpType::ExpandLD:      return true;

            case OpType::BwdStart:      return true;
            case OpType::BwdReLU:       return true;
            case OpType::BwdPow:        return true;

            case OpType::ReLU:          return true;
            case OpType::Log:           return true;
            case OpType::Neg:           return true;
            case OpType::Exp:           return true;
            case OpType::Sqrt:          return true;

            default:                    return false;
        }

    }

    void zero_params() 
    { 
        int size = storage_size / sizeof(float);
        int gridDim = (size + 255) / 256;
        int blockDim = 256;
        nulify<<<gridDim, blockDim>>>((float*)start_ptr, size);
    }
};