#include <functional>
#include <vector>
#include "nodes.h"

struct FusionRule {
    // pattern: sequence of categories this rule matches, in producer -> consumer order
    // e.g. {EwiseBinary, EwiseUnary} matches "relu(a + b)"
    std::vector<PrimCategory> pattern;

    // name for debugging
    std::string name;

    // optional extra check (shape compat, specific op types, etc.)
    // gets the matched nodes in order and returns whether the rule really applies
    std::function<bool(const std::vector<std::shared_ptr<Node>>&)> check;

    // builds the fused replacement node given the matched nodes
    std::function<std::shared_ptr<Node>(const std::vector<std::shared_ptr<Node>>&)> build;
};

inline std::vector<FusionRule> fusion_rules;  // global registry