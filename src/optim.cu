#include "allocator.cuh"
#include "evaluate.h"
#include "kernels.cuh"
#include "optim.h"

using namespace std;

void Graph::sgd_step(float lr) {
    for (auto node : require_grad_nodes) {
        if (!node->owned_grad || !node->owned_grad->is_data_allocated())
            throw logic_error("sgd_step: missing grad for node '" +
                                   node->label + "'");
        int n = node->get_size();
        // cout << "ALARM LAUNCHING SGD KERNEL \n";
        // cout << "CASE: b_2\n";
        // if (node->label == "b_2") {
        //     cout << "PRIOR TO KERNEL:\n";
        //     cout << print_block(node->d_data->b_ptr);
        // }
        // cout << "GRAD DATA:";
        // cout << print_block(node->owned_grad->d_data->b_ptr);
        kernels::sgd<<<(n + 255) / 256, 256>>>(
            node->get_data(), node->owned_grad->get_data(), lr, n);    
        // if (node->label == "b_2") {
        //     cout << "AFTER KERNEL:\n";
        //     cout << print_block(node->d_data->b_ptr);
        // }
    }
    cudaDeviceSynchronize();
}

void SGD::optimize(float* d_weight, float* d_grad, int size) {
    kernels::sgd<<<(size + 255) / 256, 256>>>(
        d_weight, d_grad, LR, size);    
    cudaDeviceSynchronize();
}

shared_ptr<SGD> make_sgd(float learning_rate) {
    shared_ptr<SGD> optim = make_shared<SGD>();
    optim->LR = learning_rate;
    return optim;
}



void _apply_sgd_to_param(shared_ptr<Node> param, shared_ptr<Node> grad, float lr) {
    int n = param->get_size();
    kernels::sgd<<<(n + 255) / 256, 256>>>(
        param->get_data(), grad->get_data(), lr, n);
}

void sgd(vector<shared_ptr<Node>> params, float lr) {
    for (size_t i = 0; i < params.size(); i++) {
        int n = params[i]->get_size();
        if (!params[i]->is_data_allocated() || !params[i]->owned_grad || !params[i]->owned_grad->is_data_allocated())
            throw logic_error("sgd: missing data or grad for parameter " + to_string(i));
        kernels::sgd<<<(n + 255) / 256, 256>>>(
            params[i]->get_data(), params[i]->owned_grad->get_data(), lr, n);
    }
}