#pragma once

#include <iostream>
#include <memory>

#include "nodes.h"

// -----------------------------------------------------------------
//                         MODULE
// -----------------------------------------------------------------
struct Module {
    virtual std::shared_ptr<Node> forward(std::shared_ptr<Node> x) = 0;
    virtual std::vector<std::shared_ptr<Node>> parameters() = 0;
    virtual ~Module() = default;
};

// ============================================================
//                         LINEAR
// ============================================================
struct LinearLayer : Module {
    std::shared_ptr<Node> W;
    std::shared_ptr<Node> b;

    LinearLayer(int in_dim, int out_dim);
    std::shared_ptr<Node> forward(std::shared_ptr<Node> x) override;
    std::vector<std::shared_ptr<Node>> parameters() override;
};

struct LinearReluLayer : Module {
    std::shared_ptr<Node> W;
    std::shared_ptr<Node> b;

    LinearReluLayer(int in_dim, int out_dim);
    std::shared_ptr<Node> forward(std::shared_ptr<Node> x) override;
    std::vector<std::shared_ptr<Node>> parameters() override;
};

// ============================================================
//                       SEQUENTIAL
// ============================================================
struct Sequential : Module {
    std::vector<std::shared_ptr<Module>> layers;

    void add_layer(std::shared_ptr<Module> layer);
    std::shared_ptr<Node> forward(std::shared_ptr<Node> x) override;
    std::vector<std::shared_ptr<Node>> parameters() override;
};
