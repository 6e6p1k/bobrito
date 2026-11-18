#pragma once
#include <cuda_runtime_api.h>
#include <memory>
#include <stdexcept>
#include <vector>
#include <string>

#include "allocator.cuh"
#include "nodes.h"
#include "ops.h"
#include "evaluate.h"
#include "stream.h"

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
        cudaError_t err_m = cudaMemsetAsync(m.data(), 0, m.bytes(), bobr::current_stream());
        cudaError_t err_v = cudaMemsetAsync(v.data(), 0, v.bytes(), bobr::current_stream());
        if (err_m != cudaSuccess) throw std::runtime_error(
            std::string("Failed to zero AdamSlot.m: ") + cudaGetErrorString(err_m)
        );
        if (err_v != cudaSuccess) throw std::runtime_error(
            std::string("Failed to zero AdamSlot.v: ") + cudaGetErrorString(err_v)
        );
    }
};

class AdamW : public Optimizer {
    float lr_, beta1_, beta2_, eps_, wd_;
    float* d_t = nullptr; // lives on GPU to use cuda graph
    // lr stays on CPU for now, should only be transferred to GPU when implementing scheduling
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
    ~AdamW() override;
};
