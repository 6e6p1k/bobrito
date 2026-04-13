#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "../src/nodes.h"
#include "../src/ops.h"
#include "../src/evaluate.h"

namespace py = pybind11;
using namespace std;

// Helper: Convert Node data to numpy array
py::array_t<float> node_to_pyarray(shared_ptr<Node> node) {
    if (!node->is_data_allocated()) {
        eval(node);
    }
    
    int ndim = node->get_ndim();
    const int* shape_ptr = node->get_shape();
    int size = node->get_size();
    
    float* h = node->d_data->d_to_h();
    
    std::vector<ssize_t> py_shape(shape_ptr, shape_ptr + ndim);
    std::vector<ssize_t> py_strides(ndim);
    ssize_t stride = sizeof(float);
    for (int i = ndim - 1; i >= 0; i--) {
        py_strides[i] = stride;
        stride *= py_shape[i];
    }
    
    auto capsule = py::capsule(h, [](void* p) { free(p); });
    return py::array_t<float>(py_shape, py_strides, h, capsule);
}

PYBIND11_MODULE(bobr, m) {
    m.doc() = "BOBR Framework - CUDA autograd library";
    
    // Node wrapper
    py::class_<Node, shared_ptr<Node>>(m, "Node")
        .def("shape", [](const shared_ptr<Node>& n) {
            vector<int> shape(n->get_shape(), n->get_shape() + n->get_ndim());
            return shape;
        })
        .def("ndim", &Node::get_ndim)
        .def("size", &Node::get_size)
        .def("eval", [](shared_ptr<Node> n) { eval(n); })
        .def("data", [](shared_ptr<Node> n) { return node_to_pyarray(n); })
        .def("set_label", &Node::set_label)
        .def("label", [](const shared_ptr<Node>& n) { return n->label; })
        .def("__repr__", [](const shared_ptr<Node>& n) {
            return "Node(" + n->label + ", shape=" + to_string(n->get_ndim()) + "D)";
        });
    
    // Graph wrapper
    py::class_<Graph, shared_ptr<Graph>>(m, "Graph")
        .def(py::init<shared_ptr<Node>, bool>(), py::arg("loss"), py::arg("disposable") = true)
        .def("forward", [](shared_ptr<Graph>& g) { g->eval(); })
        .def("backward", &Graph::backward)
        .def("step", &Graph::sgd_step, py::arg("learning_rate"))
        .def("zero_grad", [](shared_ptr<Graph>& g) { g->detach_grads_from_bwd_graph(); });
    
    // Creation ops
    m.def("make_node", &make_leaf_node);
    m.def("full", [](vector<int> shape, float value) { return full(shape, value); });
    m.def("arange", [](int start, int end, int step = 1) { return arange(start, end, step); },
          py::arg("start"), py::arg("end"), py::arg("step") = 1);
    m.def("uniform", [](vector<int> shape, float limit = 1.0f) { return uniform(shape, limit); }, 
          py::arg("shape"), py::arg("limit") = 1.0f);
    m.def("xavier_normal", [](vector<int> shape) { return xavier_normal(shape); });
    m.def("kaiming_normal", [](vector<int> shape) { return kaiming_normal(shape); });
    m.def("normal", [](vector<int> shape, float mean = 0.0f, float std = 1.0f) { return normal(shape, mean, std); },
          py::arg("shape"), py::arg("mean") = 0.0f, py::arg("std") = 1.0f);
    
    // Unary ops
    m.def("relu", [](shared_ptr<Node> a) { return relu(a); });
    m.def("exp", [](shared_ptr<Node> a) { return exp(a); });
    m.def("log", [](shared_ptr<Node> a) { return log(a); });
    m.def("neg", [](shared_ptr<Node> a) { return neg(a); });
    m.def("sqrt", [](shared_ptr<Node> a) { return sqrt(a); });
    
    // Binary ops
    m.def("add", [](shared_ptr<Node> a, shared_ptr<Node> b) { return add(a, b); });
    m.def("sub", [](shared_ptr<Node> a, shared_ptr<Node> b) { return sub(a, b); });
    m.def("mul", [](shared_ptr<Node> a, shared_ptr<Node> b) { return mul(a, b); });
    m.def("div", [](shared_ptr<Node> a, shared_ptr<Node> b) { return div(a, b); });
    m.def("matmul", [](shared_ptr<Node> a, shared_ptr<Node> b) { return matmul(a, b); });
    
    // Reduction ops
    m.def("sum_ld", &sum_ld);
    m.def("min_ld", &min_ld);
    m.def("max_ld", &max_ld);
    m.def("sum", [](shared_ptr<Node> a, int dim, bool keepdims = false) { return sum(a, dim, keepdims); },
          py::arg("a"), py::arg("dim"), py::arg("keepdims") = false);
    m.def("min", [](shared_ptr<Node> a, int dim, bool keepdims = false) { return min(a, dim, keepdims); },
          py::arg("a"), py::arg("dim"), py::arg("keepdims") = false);
    m.def("max", [](shared_ptr<Node> a, int dim, bool keepdims = false) { return max(a, dim, keepdims); },
          py::arg("a"), py::arg("dim"), py::arg("keepdims") = false);
    
    // Shape ops
    m.def("reshape", [](shared_ptr<Node> a, vector<int> shape) { 
        return reshape(a, shape.data(), shape.size()); 
    });
    m.def("contiguous", &contiguous);
    
    // Layer ops
    m.def("linear", &linear);
    
    // Loss ops
    m.def("cross_entropy", &cross_entropy);
    
    // Training helpers
    m.def("eval", [](shared_ptr<Node> node, bool export_graph = false) { eval(node, export_graph); },
          py::arg("node"), py::arg("export_graph") = false);
    m.def("make_model", &make_model);
}

