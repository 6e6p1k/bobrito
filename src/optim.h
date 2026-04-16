#pragma ones
#include <memory>
#include <vector>

#include "nodes.h"

struct OptimState {
    virtual void update(Node* weight) = 0;
};

struct SGD : OptimState {
    void update(Node *weight) override;
};

struct Layer {
    std::vector<std::shared_ptr<Node>> weights;
    OptimState optimizer;
    std::shared_ptr<Node> root;

};