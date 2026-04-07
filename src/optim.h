#pragma once
#include <iostream>
#include <vector>
#include <memory>

#include "node.h"

struct OptimState
{
    float lr;
};

struct Optim 
{
    std::vector<std::shared_ptr<SourceNode>> params;
    OptimState state;

    virtual void apply();
    void zero_grad();
    ~Optim() = default;
};

struct SGD : Optim
{
    SGD(std::vector<std::shared_ptr<SourceNode>> parameters, float learning_rate);
    void apply() override;
};
