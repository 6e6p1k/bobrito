#pragma once
#include <vector>
#include <unordered_set>
#include <unordered_map>

#include "node.h"

void count_uses(std::shared_ptr<Node> node, std::unordered_map<Node*, int>& use_counts, std::unordered_set<Node*>& visited);
void set_grad_and_travers(std::shared_ptr<Node> fwd_node, std::shared_ptr<InternalNode> new_grad);
void build_backward(std::shared_ptr<Node> fwd_node, std::shared_ptr<InternalNode> bwd_node, std::unordered_map<Node*, int>& remaining);
void _backward(std::shared_ptr<Node> node);
