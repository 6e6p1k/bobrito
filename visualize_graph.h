#pragma once
#include <unordered_set>
#include "node.h"


std::shared_ptr<SourceNode> make_source_node(int ndim, int* shape, bool requires_grad);
void post_order_dfs(Node* node, unordered_set<Node*>& visited, vector<Node*>& order);