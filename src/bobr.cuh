#pragma once

#include <ostream>
#include <vector>
#include "tensor.cuh"
#include "optim.h"


namespace bobr 
{
Tensor full(std::vector<int> shape, float val, bool requires_grad);
Tensor full_like(Tensor t, float val);

Tensor ones(std::vector<int> shape, bool requires_grad);
Tensor zeros(std::vector<int> shape, bool requires_grad);
Tensor ones_like(Tensor t);
Tensor zeros_like(Tensor t);

std::string get_allocation_stats();
void reset_memory_pool();
void clear_graph(Tensor t);
void draw_graph(Tensor t, const char* png_path);
}