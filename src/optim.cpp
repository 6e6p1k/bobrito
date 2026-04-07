#include "optim.h"
#include "node.h"
#include <stdexcept>
#include "ops.cuh"

void Optim::apply() 
{ return; }

void Optim::zero_grad()
{
    for (int i = 0; i < params.size(); ++i) {
        params[i]->grad_node.reset();
        params[i]->owned_grad = nullptr;
    }
}

SGD::SGD(std::vector<std::shared_ptr<SourceNode>> parameters, float learning_rate)
{
    this->params = parameters;
    this->state.lr = learning_rate;
}

void SGD::apply()
{
    for (int i = 0; i < params.size(); i++)
    {
        if (params[i]->kind != NodeKind::Source)
            throw std::logic_error("Parameter " + params[i]->label + "'s grad cannot be computed.");
        if (params[i]->h_requiers_grad == false)
            throw std::logic_error("Parameter " + params[i]->label + " requires_grad is set to false.");
        
        auto grad = params[i]->grad_node.lock();
        if (grad == nullptr || grad->is_data_allocated() == false)
            throw std::logic_error("No grad found for parameter " + params[i]->label + ".");

        cu_ops::sgd(params[i].get(), state.lr);
    }
}
