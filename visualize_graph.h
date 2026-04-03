#pragma once
#include <unordered_set>
#include "node.h"
#include <vector>

std::shared_ptr<SourceNode> make_source_node(int ndim, int* shape, bool requires_grad);
void post_order_dfs(Node* node, std::unordered_set<Node*>& visited, std::vector<Node*>& order);



void print_all_nodes_labels(std::vector<Node*> nodes);
void print_all_nodes_grads(std::vector<Node*> nodes);
std::shared_ptr<SourceNode> make_source_node(int ndim, int* shape, bool requires_grad);
namespace qq
{
std::shared_ptr<SourceNode> fill(int ndim, int* shape, float val, bool requires_grad);
}
void _find_roots(Node* node, std::unordered_set<Node*>& visited, std::vector<Node*>& roots);
std::vector<Node*> get_roots(std::shared_ptr<InternalNode> node);
