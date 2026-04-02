#pragma once
#include <unordered_set>
#include "node.h"
#include <vector>

std::shared_ptr<SourceNode> make_source_node(int ndim, int* shape, bool requires_grad);
void post_order_dfs(Node* node, unordered_set<Node*>& visited, vector<Node*>& order);



void print_all_nodes_labels(vector<Node*> nodes);
void print_all_nodes_grads(vector<Node*> nodes);
std::shared_ptr<SourceNode> make_source_node(int ndim, int* shape, bool requires_grad);
namespace qq
{
std::shared_ptr<SourceNode> fill(int ndim, int* shape, float val, bool requires_grad);
}
void _find_roots(Node* node, unordered_set<Node*>& visited, vector<Node*>& roots);
vector<Node*> get_roots(std::shared_ptr<InternalNode> node);
