#include "evaluate.h"
#include "kernels.cuh"

void Graph::sgd_step(float lr) {
    for (auto node : require_grad_nodes) {
        if (!node->owned_grad || !node->owned_grad->is_data_allocated())
            throw std::logic_error("sgd_step: missing grad for node '" +
                                   node->label + "'");
        int n = node->get_size();
        kernels::sgd<<<(n + 255) / 256, 256>>>(
            node->get_data(), node->owned_grad->get_data(), lr, n);
    }
}
