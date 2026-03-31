#pragma once
#include "grad.h"
#include <memory>

void visualize(const std::shared_ptr<Node>& root);
void export_dot(const std::shared_ptr<Node>& root, const char* filename);
void backward_viz(const std::shared_ptr<Node>& root);
