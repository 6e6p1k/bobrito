#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "node.h"
#include "backward.h"
#include "ops.cuh"
#include "utils.h"

namespace py = pybind11;

class Tensor {
public:
    std::shared_ptr<Node> node;

    Tensor(std::vector<int> shape, bool requires_grad = false) {
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

    Tensor(std::shared_ptr<Node> n) : node(n) {}

    void fill(float val) {
        cu_ops::fill(node.get(), val);
    }

    Tensor matmul(const Tensor& other) {
        return Tensor(::matmul(node, other.node));
    }

    Tensor operator+(const Tensor& other) {
        return Tensor(::add(node, other.node));
    }

    Tensor operator*(const Tensor& other) {
        return Tensor(::mul(node, other.node));
    }

    Tensor relu() {
        return Tensor(::relu(node));
    }

    Tensor sum(int dim) {
        return Tensor(::sum(node, dim));
    }

    void evaluate(void) {
        eval(node);
    }

    void geval() {
        _geval(node);
    }

    void build_backward(void) {
        _backward(node);
    }

    py::array_t<float> data() {
        if (!node->is_data_allocated())
            throw std::runtime_error("No data available");

        int ndim = node->h_meta->ndim;
        int* shape = node->h_meta->shape();
        int size = node->h_meta->size;
        float* h = node->d_data->d_to_h();

        std::vector<ssize_t> py_shape(shape, shape + ndim);
        std::vector<ssize_t> py_strides(ndim);
        ssize_t stride = sizeof(float);
        for (int i = ndim - 1; i >= 0; i--) {
            py_strides[i] = stride;
            stride *= shape[i];
        }

        auto capsule = py::capsule(h, [](void* p) { free(p); });
        return py::array_t<float>(py_shape, py_strides, h, capsule);
    }

    py::array_t<float> grad() {
        if (!node->grad_node || !node->grad_node->is_data_allocated())
            throw std::runtime_error("No grad available");

        int ndim = node->h_meta->ndim;
        int* shape = node->h_meta->shape();
        float* h = node->grad_node->d_data->d_to_h();

        std::vector<ssize_t> py_shape(shape, shape + ndim);
        std::vector<ssize_t> py_strides(ndim);
        ssize_t stride = sizeof(float);
        for (int i = ndim - 1; i >= 0; i--) {
            py_strides[i] = stride;
            stride *= shape[i];
        }

        auto capsule = py::capsule(h, [](void* p) { free(p); });
        return py::array_t<float>(py_shape, py_strides, h, capsule);
    }

    std::string get_label() {
        return node->label;
    }

    void set_label(std::string label) {
        node->label = label;
    }

    void sync() {
        cudaDeviceSynchronize();
    }

    void from_numpy(py::array_t<float> arr) {
        auto buf = arr.request();
        float* ptr = static_cast<float*>(buf.ptr);
        int size = 1;
        for (auto s : buf.shape) size *= s;
        node->d_data->h_to_d(ptr, size);
    }

    void zero_grad(void) {
        node->grad_node = nullptr;
    }
};


PYBIND11_MODULE(bobr, m) {
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<std::vector<int>, bool>(),
             py::arg("shape"), py::arg("requires_grad") = false)
        .def("fill", &Tensor::fill)
        .def("matmul", &Tensor::matmul)
        .def("relu", &Tensor::relu)
        .def("sum", &Tensor::sum)
        .def("eval", &Tensor::evaluate)
        .def("geval", &Tensor::geval)
        .def("build_backward", &Tensor::build_backward)
        .def("sync", &Tensor::sync)
        .def("from_numpy", &Tensor::from_numpy)
        .def("zero_grad", &Tensor::zero_grad)
        .def("__add__", &Tensor::operator+)
        .def("__mul__", &Tensor::operator*)
        .def("__matmul__", &Tensor::matmul)
        .def("__repr__", [](const Tensor& t) {
            int ndim = t.node->h_meta->ndim;
            int* shape = t.node->h_meta->shape();
            std::string s = "Tensor(";
            s += "label='" + t.node->label + "', ";
            s += "shape=[";
            for (int i = 0; i < ndim; i++) {
                if (i) s += ", ";
                s += std::to_string(shape[i]);
            }
            s += "])";
            return s;
        })
        .def_property_readonly("data", &Tensor::data)
        .def_property_readonly("grad", &Tensor::grad)
        .def_property("label", &Tensor::get_label, &Tensor::set_label);
}

// nvcc -shared -std=c++17 -Xcompiler -fPIC     $(python3 -m pybind11 --includes)     -I..     bindings.cu ../ops.cu ../allocator.cu ../kernels.cu ../utils.cpp ../node.cpp ../backward.cpp ../graph_viz.cpp ../visualize_graph.cpp     -o bobr$(python3-config --extension-suffix)     -lcublas