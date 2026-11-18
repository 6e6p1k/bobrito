#pragma once
#include <memory>
#include <vector>


#include "optim.h"
#include "evaluate.h"
#include "stream.h"

class TrainStep {
    std::shared_ptr<Graph> g_;
    std::shared_ptr<Optimizer> opt_;
    CapturedStep graph_;
    std::vector<void*> fingerprint_;
    int calls_ = 0;
    bool use_graph_;
    bool captured_ = false;

    std::vector<void*> current_ptrs() const;
    
    public:
    TrainStep(std::shared_ptr<Graph> g, std::shared_ptr<Optimizer> opt, bool graph = true);
    void operator()();
};