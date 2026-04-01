#pragma once
#include <unordered_set>

#include "node.h"


void build_backward(shared_ptr<Node> fwd_node, shared_ptr<InternalNode> bwd_node, int use_count);

// <Topological sort>
void build_topo(Node* node, unordered_set<Node*>& visited, vector<InternalNode*>& inodes, vector<SourceNode*>& snodes);
// </Topological sort>

void backward_pass(shared_ptr<Node> node);
