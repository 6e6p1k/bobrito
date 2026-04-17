#include <memory>
#include "allocator.cuh"
#include "evaluate.h"
#include "kernels.cuh"
#include "optim.h"

#define GRID_SIZE(n) ((n + 255) / 256)
#define BLOCK_SIZE 256

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

void Optimizer::zero_grad() {
    for (auto param : params_) {
        bool is_param = param && param->requires_grad && param->h_meta;
        if (!is_param) continue;
        auto grad = param->owned_grad;
        if (!grad || !grad->is_data_allocated()) continue;
        param->owned_grad->d_data.reset();
    }
}

// ============================================================
//                         SGD
// ============================================================
SGD::SGD(std::vector<std::shared_ptr<Node>> params, float lr)
    : Optimizer(std::move(params)), lr_(lr) {}

void SGD::step() {
    for (auto node : params_) {
        if (!node->owned_grad || !node->owned_grad->is_data_allocated())
            throw logic_error("sgd_step: missing grad for node '" +
                                   node->label + "'");
        int n = node->get_size();
        kernels::sgd<<<(n + 255) / 256, 256>>>(
            node->get_data(), node->owned_grad->get_data(), lr_, n);    
    }
    cudaDeviceSynchronize();
}

// ============================================================
//                         AdamW
// ============================================================
AdamW::AdamW(std::vector<std::shared_ptr<Node>> params,
             float lr, float beta1, float beta2, float eps, float weight_decay)
    : Optimizer(std::move(params)),
      lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), wd_(weight_decay)
{
    slots_.reserve(params_.size());
    for (auto& weight : params_) {
        slots_.emplace_back(weight);
    }
}

AdamW::AdamW(std::shared_ptr<Graph> g,
             float lr, float beta1, float beta2, float eps, float weight_decay)
    : AdamW(g->parameters(), lr, beta1, beta2, eps, weight_decay)
{}

void AdamW::step() {
    t_++; // update step
    lr_t = lr_ * sqrtf(1 - powf(beta2_, t_)) / (1 - powf(beta1_, t_));
    for (auto slot : slots_) {
        // cout << "Launching kernel\n";
        // cout << print_block(slot.m.b_ptr);
        // cout << print_block(slot.v.b_ptr);
        int n = slot.param->get_size();
        // cout << "n = " << n << endl;
        kernels::adamw<<<GRID_SIZE(n), BLOCK_SIZE>>>(
            slot.param->get_data(),
            slot.param->owned_grad->get_data(),
            slot.m.data(),
            slot.v.data(),
            lr_, lr_t, beta1_, beta2_, eps_, wd_,
            n
        );
    }
}   

void AdamW::zero_grad() {
    for (auto& p : params_) {
        if (p->owned_grad && p->owned_grad->is_data_allocated()) {
            cudaMemsetAsync(p->owned_grad->get_data(), 0, p->owned_grad->get_size() * sizeof(float));
        }
    }
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