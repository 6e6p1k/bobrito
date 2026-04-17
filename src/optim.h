#pragma once
#include <memory>
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
