#include <ostream>
#include <stdexcept>
#include <vector>
#include <memory>
#include "node.h"
#include "backward.h"
#include "node_helpers.h"
#include "ops.cuh"
#include "utils.h"
#include "tensor.cuh"


// class Tensor {
// public:
//     std::shared_ptr<Node> node;

    Tensor::Tensor(std::vector<int> shape, bool requires_grad) {
        int ndim = shape.size();
        auto snode = std::make_shared<SourceNode>();
        snode->kind = NodeKind::Source;
        snode->h_meta = make_shape_data(ndim, shape.data());
        snode->h_requiers_grad = requires_grad;
        if (requires_grad) snode->param_count = 1;
        snode->d_data = std::make_shared<ContentData>();
        int size = utils::numel(ndim, shape.data());
        snode->d_data->allocate(size);
        node = snode;
    }

    Tensor::Tensor(std::shared_ptr<Node> n) : node(n) {}

    void Tensor::fill(float val) {
        cu_ops::fill(node.get(), val);
    }

    Tensor Tensor::matmul(const Tensor& other) { return Tensor(::matmul(node, other.node)); }

    // elemwise
    Tensor Tensor::operator+(const Tensor& other) { return Tensor(::add(node, other.node)); }
    Tensor Tensor::operator-(const Tensor& other) { return Tensor(::sub(node, other.node)); }
    Tensor Tensor::operator*(const Tensor& other) { return Tensor(::mul(node, other.node)); }
    Tensor Tensor::operator/(const Tensor& other) { return Tensor(::div(node, other.node)); }
    Tensor Tensor::pow(const float exponent)     
    { 
        auto n2 = make_filled_source_node(node->h_meta->shape(), node->h_meta->ndim, exponent, false);
        return Tensor(::pow(node, n2)); 
    }

    // reductions
    Tensor Tensor::sum(int dim, bool keepdims) { return Tensor(::sum(node, dim, keepdims)); }
    Tensor Tensor::max(int dim, bool keepdims) { return Tensor(::max(node, dim, keepdims)); }
    Tensor Tensor::min(int dim, bool keepdims) { return Tensor(::min(node, dim, keepdims)); }

    // unary ops
    Tensor Tensor::relu() { return Tensor(::relu(node)); }
    Tensor Tensor::log() { return Tensor(::log(node)); }
    Tensor Tensor::sqrt() { return Tensor(::sqrt(node)); }
    Tensor Tensor::neg() { return Tensor(::neg(node)); }
    Tensor Tensor::exp() { return Tensor(::exp(node)); }

    void Tensor::evaluate(void) {
        eval(node);
    }

    void Tensor::geval() {
        _geval(node);
    }

    void Tensor::build_backward(void) {
        _backward(node);
    }

    std::string Tensor::get_label() {
        return node->label;
    }

    void Tensor::set_label(std::string label) {
        node->label = label;
    }

    bool Tensor::get_requires_grad() 
    {
        if (node->kind == NodeKind::Source)
        {
            std::shared_ptr<SourceNode> snode = std::static_pointer_cast<SourceNode>(node);
            return snode->h_requiers_grad;
        }
        return false;
    }

    void Tensor::set_requires_grad(bool requires_grad)
    {
        if (node->kind == NodeKind::Source)
        {
            std::shared_ptr<SourceNode> snode = std::static_pointer_cast<SourceNode>(node);
            snode->h_requiers_grad = requires_grad;
        }
        else throw std::logic_error("You can't assign requires_grad to an intermediate result.");
    }

    void Tensor::sync(void) {
        cudaDeviceSynchronize();
    }

    std::string Tensor::shape_to_string(void) const
    {
        int ndim = node->h_meta->ndim;
        int* shape = node->h_meta->shape();
        std::string s = "[";
        for (int i = 0; i < ndim; i++) {
            if (i) s += ", ";
            s += std::to_string(shape[i]);
        }
        s += "]";
        return s;
    }
// private:


// }; END OF CLASS TENSOR
