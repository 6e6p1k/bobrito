#pragma once
#include <vector>
#include <memory>
#include "node.h"

class Tensor {
public:
    std::shared_ptr<Node> node;

    Tensor(std::vector<int> shape, bool requires_grad = false);

    Tensor(std::shared_ptr<Node> n);

    // OPS
    void fill(float val);
    Tensor matmul(const Tensor& other);
    Tensor operator+(const Tensor& other);
    Tensor operator-(const Tensor& other);
    Tensor operator*(const Tensor& other);
    Tensor operator/(const Tensor& other);
    Tensor relu();
    Tensor sum(int dim);

    // CALCULATION
    void evaluate(void);
    void geval();
    void build_backward(void);

    // PROPERTIES
    std::string get_label();
    void set_label(std::string label);
    bool get_requires_grad();
    void set_requires_grad(bool requires_grad);
    void sync(void);
    void zero_grad(void);

    // UTILS
    std::string shape_to_string() const;

};