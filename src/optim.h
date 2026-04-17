#pragma once
#include <memory>
#include <random>
#include <vector>

#include "allocator.cuh"
#include "nodes.h"
#include "ops.h"
#include "evaluate.h"

// -----------------------------------------------------------------
//                        OPTIM
// -----------------------------------------------------------------
class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step() = 0;
    virtual void zero_grad();
    virtual const char* name() const = 0;

    // optional but nice to have:
    // virtual void save_state(...) = 0;
    // virtual void load_state(...) = 0;

protected:
    std::vector<std::shared_ptr<Node>> params_;
    Optimizer(std::vector<std::shared_ptr<Node>> params)
        : params_(std::move(params)) {}
};

// ============================================================
//                         SGD
// ============================================================
class SGD : public Optimizer {
    float lr_;
    // no extra state
public:
    SGD(std::vector<std::shared_ptr<Node>> params, float lr);
    void step() override;
    const char* name() const override { return "SGD"; }
};

// ============================================================
//                         Adam
// ============================================================
struct AdamSlot {
    std::shared_ptr<Node> param;
    ContentData m;
    ContentData v;

    explicit AdamSlot(std::shared_ptr<Node> p)
        : param(std::move(p))
    {
        m.allocate(param->get_size());
        v.allocate(param->get_size());
    }
};

class AdamW : public Optimizer {
    float lr_, lr_t, beta1_, beta2_, eps_, wd_;
    int t_ = 0;
    std::vector<AdamSlot> slots_;

    // parallel to params_
    std::vector<std::shared_ptr<Node>> m_;   // first moment, one per param, same shape
    std::vector<std::shared_ptr<Node>> v_;   // second moment
public:
    AdamW(std::vector<std::shared_ptr<Node>> params,
         float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f, float weight_decay = 1e-2f);
    AdamW(std::shared_ptr<Graph> g,
         float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f, float weight_decay = 1e-2f);
    void step() override;
    void zero_grad() override;
    const char* name() const override { return "Adam"; }
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
        float std = sqrtf(6.0f / (in_dim + out_dim));
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

struct LinearReluLayer : Module {
    std::shared_ptr<Node> W;
    std::shared_ptr<Node> b;

    LinearReluLayer(int in_dim, int out_dim) {
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
        return relu(add(matmul(x, W), b));
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
