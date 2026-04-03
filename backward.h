#pragma once
#include <vector>
#include <unordered_set>

#include "node.h"


void build_backward(std::shared_ptr<Node> fwd_node, std::shared_ptr<InternalNode> bwd_node, int param_count);
void _backward(std::shared_ptr<Node> node);
void backward_pass(std::shared_ptr<Node> node);
