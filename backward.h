#pragma once
#include <vector>
#include <unordered_set>

#include "node.h"


void build_backward(shared_ptr<Node> fwd_node, shared_ptr<InternalNode> bwd_node, int param_count);
void _backward(shared_ptr<Node> node);
void backward_pass(shared_ptr<Node> node);
