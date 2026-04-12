#pragma once
#include <iostream>
#include <memory>
#include <vector>

#include "nodes.h"

void CUDA_CHECK();

// unary
std::shared_ptr<Node> relu(std::shared_ptr<Node> a);
std::shared_ptr<Node> bwd_relu(std::shared_ptr<Node> grad, std::shared_ptr<Node> fwd_node);
std::shared_ptr<Node> log(std::shared_ptr<Node> a);
std::shared_ptr<Node> exp(std::shared_ptr<Node> a);
std::shared_ptr<Node> neg(std::shared_ptr<Node> a);
std::shared_ptr<Node> sqrt(std::shared_ptr<Node> a);

// elemwise
std::shared_ptr<Node> add(std::shared_ptr<Node> a, std::shared_ptr<Node> b);
std::shared_ptr<Node> sub(std::shared_ptr<Node> a, std::shared_ptr<Node> b);
std::shared_ptr<Node> mul(std::shared_ptr<Node> a, std::shared_ptr<Node> b);
std::shared_ptr<Node> div(std::shared_ptr<Node> a, std::shared_ptr<Node> b);
std::shared_ptr<Node> pow(std::shared_ptr<Node> a, float exponent);
std::shared_ptr<Node> matmul(std::shared_ptr<Node> a, std::shared_ptr<Node> b); // kinda
std::shared_ptr<Node> bwd_div(std::shared_ptr<Node> fwd, std::shared_ptr<Node> bwd);
std::shared_ptr<Node> bwd_pow(std::shared_ptr<Node> fwd, std::shared_ptr<Node> bwd);

// reductions
std::shared_ptr<Node> sum_ld(std::shared_ptr<Node> a);
std::shared_ptr<Node> min_ld(std::shared_ptr<Node> a);
std::shared_ptr<Node> max_ld(std::shared_ptr<Node> a);
std::shared_ptr<Node> expand_ld(std::shared_ptr<Node> a, int* out_shape, int out_ndim); // not really a reduction
std::shared_ptr<Node> sum(std::shared_ptr<Node> a, int dim, bool keepdims);
std::shared_ptr<Node> min(std::shared_ptr<Node> a, int dim, bool keepdims);
std::shared_ptr<Node> max(std::shared_ptr<Node> a, int dim, bool keepdims);

// tools 
std::shared_ptr<Node> contiguous(std::shared_ptr<Node> a);
std::shared_ptr<Node> reshape(std::shared_ptr<Node> a, int* shape, int ndim);
std::shared_ptr<Node> reduce_to_shape(std::shared_ptr<Node> a, int* out_shape, int out_ndim);
std::shared_ptr<Node> reduce_to_shape(std::shared_ptr<Node> a, std::vector<int> out_shape);

// interface 
std::shared_ptr<Node> make_leaf_node(std::vector<int> shape, std::vector<float> h_data);
std::shared_ptr<Node> full(int* shape, int ndim, float fill_val);
std::shared_ptr<Node> full(std::vector<int> shape, float fill_val);
std::shared_ptr<Node> arange(std::vector<int> shape, int start, int step = 1);
std::shared_ptr<Node> arange(int start, int end, int step = 1);
std::shared_ptr<Node> uniform(int* shape, int ndim, float limit = 1);
std::shared_ptr<Node> uniform(std::vector<int> shape, float limit);
std::shared_ptr<Node> xavier_uniform(int* shape, int ndim);
std::shared_ptr<Node> xavier_uniform(std::vector<int> shape);
std::shared_ptr<Node> normal(int* shape, int ndim, float mean, float std);
std::shared_ptr<Node> normal(std::vector<int> shape, float mean, float std);
std::shared_ptr<Node> xavier_normal(int* shape, int ndim);
std::shared_ptr<Node> xavier_normal(std::vector<int> shape);
std::shared_ptr<Node> kaiming_normal(int* shape, int ndim);
std::shared_ptr<Node> kaiming_normal(std::vector<int> shape);

// layers 
std::shared_ptr<Node> linear(std::shared_ptr<Node> a, int out_dim);

// loss
std::shared_ptr<Node> cross_entropy(std::shared_ptr<Node> logits, std::shared_ptr<Node> targets);







