#pragma once
#include <memory>
#include <random>
#include <vector>

#include "nodes.h"
#include "ops.h"

// -----------------------------------------------------------------
//                        OPTIMSTATE
// -----------------------------------------------------------------

// new idea: nodes with requires_grad = true carry their own optim state
// u can have several different optimizers within a single model
// each optim state has an initiation function and 

struct OptimPrimitive {
    virtual void optimize(float* d_weight, float* d_grad, int size);
    const std::string name();
};

struct SGD : OptimPrimitive {
    float LR = 1e-3;
    void optimize(float* d_weight, float* d_grad, int size) override;
    const std::string name() { return "SGD"; }
};

struct Adam : OptimPrimitive {
    float LR = 1e-3;
    std::shared_ptr<Node> betas[2];
    float esp;
    void optimize(std::vector<Node*> weights) override;
    const std::string name() { return "SGD"; }
};

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

    LinearLayer(int in_dim, int out_dim) {
        // Kaiming normal init — generate on CPU, copy to GPU as leaf
        float std = sqrtf(2.0f / in_dim);
        int w_size = in_dim * out_dim;
        std::vector<float> w_data(w_size);

        static std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, std);
        for (int i = 0; i < w_size; i++)
            w_data[i] = dist(rng);

        W = make_leaf_node({in_dim, out_dim}, w_data);
        W->requires_grad = true;
        W->set_label("W(" + std::to_string(in_dim) + "x" + std::to_string(out_dim) + ")");

        // Bias = zeros
        std::vector<float> b_data(out_dim, 0.0f);
        b = make_leaf_node({1, out_dim}, b_data);
        b->requires_grad = true;
        b->set_label("b(" + std::to_string(out_dim) + ")");
    }

    std::shared_ptr<Node> forward(std::shared_ptr<Node> x) override {
        return add(matmul(x, W), b);
    }

    std::vector<std::shared_ptr<Node>> parameters() override {
        return {W, b};
    }
};

// ============================================================
//                       SEQUENTIAL
// ============================================================
struct Sequential : Module {
    std::vector<std::shared_ptr<Module>> layers;

    void add_layer(std::shared_ptr<Module> layer) {
        layers.push_back(layer);
    }

    std::shared_ptr<Node> forward(std::shared_ptr<Node> x) override {
        for (auto& layer : layers)
            x = layer->forward(x);
        return x;
    }

    std::vector<std::shared_ptr<Node>> parameters() override {
        std::vector<std::shared_ptr<Node>> params;
        for (auto& layer : layers) {
            auto p = layer->parameters();
            params.insert(params.end(), p.begin(), p.end());
        }
        return params;
    }
};
