#include <exception>
#include <iostream>
#include <vector>

#include "train_step.h"

std::vector<void*> TrainStep::current_ptrs() const {
    std::vector<void*> ptrs;
    ptrs.reserve(g_->fwd_order.size() + g_->bwd_order.size());
    for (Node* n : g_->fwd_order)
        ptrs.push_back(n->is_data_allocated() ? n-> get_data() : nullptr);
    for (Node* n : g_->bwd_order)
        ptrs.push_back(n->is_data_allocated() ? n-> get_data() : nullptr);
    return ptrs;
}

TrainStep::TrainStep(std::shared_ptr<Graph> g, std::shared_ptr<Optimizer> opt, bool graph) {
    g_ = g;
    opt_ = opt;
    use_graph_ = graph;
}

void TrainStep::operator()() {
    if (!use_graph_) { // graph is unusable
        g_->fwd_bwd_pass();
        opt_->step();
        return;
    }
    if (captured_ && current_ptrs() != fingerprint_) { // buffers were reallocated
        captured_ = false;
        calls_ = 0; // next call is a warmup again
    }
    if (calls_++ == 0) { // first call: eager warmup
        g_->fwd_bwd_pass();
        opt_->step();
        return;
    }
    if (!captured_ || current_ptrs() != fingerprint_) {
        try {
            graph_.capture([&]{ g_->fwd_bwd_pass(); opt_->step(); });
        } catch (const std::exception& e) {
            std::cerr << "capture failed, running eagerly: " << e.what() << std::endl;
            use_graph_ = false;
            g_->fwd_bwd_pass();
            opt_->step();
            return;
        }
        fingerprint_ = current_ptrs();
        captured_ = true;
    }
    graph_.replay();
}
