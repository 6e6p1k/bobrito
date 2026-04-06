#pragma once

#include <ostream>
#include <vector>
#include "tensor.cuh"

namespace bobr 
{
Tensor full(std::vector<int> shape, float val, bool requires_grad);
Tensor full_like(Tensor t, float val);

Tensor ones(std::vector<int> shape, bool requires_grad = false);
Tensor zeros(std::vector<int> shape, bool requires_grad = false);
Tensor ones_like(Tensor t);
Tensor zeros_like(Tensor t);
}