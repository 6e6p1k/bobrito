#include <future>
#include <memory>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "../src/nodes.h"
#include "../src/ops.h"
#include "../src/evaluate.h"
#include "../src/utils.h"
#include "pybind11/detail/common.h"
#include "pybind11/pytypes.h"

namespace py = pybind11;
using namespace std;

py::array_t<float> data_to_pyarray(float* h_data, int* shape, int ndim) {
    std::vector<ssize_t> py_shape(shape, shape + ndim);
    std::vector<ssize_t> py_strides(ndim);
    ssize_t stride = sizeof(float);
    for (int i = ndim - 1; i >= 0; i--) {
        py_strides[i] = stride;
        stride *= shape[i];
    }
    auto capsule = py::capsule(h_data, [](void* p) { std::free(p); });
    return py::array_t<float>(py_shape, py_strides, h_data, capsule);
}

py::array_t<int> meta_to_pyarray(int* h_data, int size) {
    py::array_t<int> py_meta(size, h_data);
    return py_meta;
}

void steal_data(shared_ptr<Node>& thief, py::object victim) {
    std::vector<int> shape;
    float* data_ptr = nullptr;
    bool on_gpu = false;

    if (py::hasattr(victim, "__cuda_array_interface__")) {
        py::dict iface = victim.attr("__cuda_array_interface__").cast<py::dict>();
        py::tuple shape_tuple = iface["shape"].cast<py::tuple>();
        for (auto s : shape_tuple) shape.push_back(s.cast<int>());
        py::tuple data_tuple = iface["data"].cast<py::tuple>();
        data_ptr = reinterpret_cast<float*>(data_tuple[0].cast<uintptr_t>());
        on_gpu = true;
    }
    // Then buffer protocol (numpy, array.array)
    else if (py::isinstance<py::buffer>(victim)) {
        py::buffer_info info = py::buffer(victim).request();
        for (auto s : info.shape) shape.push_back(s);
        data_ptr = static_cast<float*>(info.ptr);
    }
    else {
        throw std::runtime_error("Expected numpy array, torch tensor, or cuda array");
    }

    // check shape
    int size = utils::numel(shape.size(), shape.data());
    bool ndim_match = shape.size() == thief->get_ndim();
    bool size_match = size == thief->get_size();
    if (!ndim_match || !size_match) 
        throw logic_error("Metadata doesn't match.");

    // check thief's ContentData
    if (!thief->is_data()) {
        thief->d_data = make_shared<ContentData>();
        thief->d_data->allocate(size);
    } else if (!thief->is_data_allocated()) {
        thief->d_data->allocate(size);
    } 

    if (on_gpu) {
        thief->d_data->d_to_d(data_ptr, size);
    } else {
        thief->d_data->h_to_d(data_ptr, size);
    }
}

void steal_data(shared_ptr<Node>& thief, shared_ptr<Node> victim) {
    thief->d_data = victim->d_data;
    // thief's d_data will be freed automatically
    // as its ref_count is dropped to 0
}

PYBIND11_MODULE(bobr, m) {
    m.doc() = "BOBR Framework - CUDA autograd library";
    
    // Node wrapper
    py::class_<Node, shared_ptr<Node>>(m, "Node")
        .def(py::init([](py::object obj, bool requires_grad) 
        {
            // Extract shape and data from whatever we got
            std::vector<int> shape;
            float* data_ptr = nullptr;
            int size = 1;
            bool on_gpu = false;

            // Check __cuda_array_interface__ first (PyTorch CUDA, CuPy)
            if (py::hasattr(obj, "__cuda_array_interface__")) {
                py::dict iface = obj.attr("__cuda_array_interface__").cast<py::dict>();
                py::tuple shape_tuple = iface["shape"].cast<py::tuple>();
                for (auto s : shape_tuple) shape.push_back(s.cast<int>());
                py::tuple data_tuple = iface["data"].cast<py::tuple>();
                data_ptr = reinterpret_cast<float*>(data_tuple[0].cast<uintptr_t>());
                on_gpu = true;
            }
            // Then buffer protocol (numpy, array.array)
            else if (py::isinstance<py::buffer>(obj)) {
                py::buffer_info info = py::buffer(obj).request();
                for (auto s : info.shape) shape.push_back(s);
                data_ptr = static_cast<float*>(info.ptr);
            }
            else {
                throw std::runtime_error("Expected numpy array, torch tensor, or cuda array");
            }

            shared_ptr<Node> node = make_shared<Node>();
            node->requires_grad = requires_grad;
            node->h_meta = make_shape_data(shape.size(), shape.data());
            node->prim = &prim_leaf;
            node->d_data = make_shared<ContentData>();
            node->d_data->allocate(node->get_size());

            if (on_gpu) {
                for (int s : shape) size *= s;
                cudaMemcpy(node->d_data->data(), data_ptr,
                        size * sizeof(float), cudaMemcpyDeviceToDevice);
            } else {
                for (int s : shape) size *= s;
                node->d_data->h_to_d(data_ptr, size);
            }

            return node;
        }), py::arg("data"), py::arg("requires_grad") = false)

        .def_property_readonly("data", [](const shared_ptr<Node>& n) {
            if (!n->is_data_allocated()) throw logic_error("Data not allocated!");
            return data_to_pyarray(n->d_data->d_to_h(), n->get_shape(), n->get_ndim());
        })

        .def_property_readonly("grad", [](const shared_ptr<Node>& n) {
            if (!n->requires_grad) throw logic_error("This is not a leaf node!");
            if (!n->owned_grad || !n->owned_grad->is_data_allocated())
                throw logic_error("Grad is not allocated!");
            return data_to_pyarray(n->owned_grad->d_data->d_to_h(), n->get_shape(), n->get_ndim());
        })

        .def_property_readonly("shape", [](const shared_ptr<Node>& n) {
            return meta_to_pyarray(n->get_shape(), n->get_ndim());
        })
        .def_property_readonly("strides", [](const shared_ptr<Node>& n) {
            return meta_to_pyarray(n->get_strides(), n->get_ndim());
        })
        .def_property_readonly("size", &Node::get_size)
        .def_property_readonly("ndim", &Node::get_ndim)
        .def_property_readonly("requires_grad", [](const shared_ptr<Node>& n) { return n->requires_grad; })

        .def_property("label", [](const shared_ptr<Node>& n) { return n->label; },
                             [](shared_ptr<Node>& n, const string& new_label) { n->set_label(new_label); })

        // debug properties
        .def_property_readonly("is_bwd", [](const shared_ptr<Node>& n) { return n->is_bwd; })
        .def_property_readonly("is_label", &Node::is_label)
        .def_property_readonly("is_meta", &Node::is_meta)
        .def_property_readonly("is_data", &Node::is_data)
        .def_property_readonly("is_data_allocated", &Node::is_data_allocated)
        .def_property_readonly("is_grad_buffer", &Node::is_grad_buffer)
        .def_property_readonly("is_grad_node", &Node::is_grad_node)
        .def_property_readonly("is_owned_grad", [](const shared_ptr<Node>& n) { return n->owned_grad != nullptr && n->owned_grad->is_data_allocated(); })
        .def_property_readonly("data_ptr", [](const shared_ptr<Node>& n) { return n->get_data(); })

        .def("eval", [](shared_ptr<Node>& n) { eval(n); })
        .def("steal_data", py::overload_cast<shared_ptr<Node>&, shared_ptr<Node>>(&steal_data))
        .def("steal_data", py::overload_cast<shared_ptr<Node>&, py::object>(&steal_data))
        .def("set_label", &Node::set_label)
        .def("get_owned_grad_ptr", [](shared_ptr<Node>& n) { return n->owned_grad->d_data->data(); })
        .def("__repr__", [](const shared_ptr<Node>& n) {
            return make_node_string(n);
        })

        // Unary ops
        .def("relu", [](const shared_ptr<Node>& n) { return relu(n); })
        .def("exp", [](const shared_ptr<Node>& n) { return exp(n); })
        .def("log", [](const shared_ptr<Node>& n) { return log(n); })
        .def("__neg__", [](const shared_ptr<Node>& n) { return neg(n); })
        .def("sqrt", [](const shared_ptr<Node>& n) { return sqrt(n); })

        // Binary ops
        .def("__add__", [](const shared_ptr<Node>& n, shared_ptr<Node> b) { return add(n, b); })
        .def("__sub__", [](const shared_ptr<Node>& n, shared_ptr<Node> b) { return sub(n, b); })
        .def("__mul__", [](const shared_ptr<Node>& n, shared_ptr<Node> b) { return mul(n, b); })
        .def("__div__", [](const shared_ptr<Node>& n, shared_ptr<Node> b) { return div(n, b); })
        .def("__matmul__", [](const shared_ptr<Node>& n, shared_ptr<Node> b) { return matmul(n, b); })

        // Reduction ops
        .def("sum", [](const shared_ptr<Node>& n, int dim, bool keepdims = false) { return sum(n, dim, keepdims); },
            py::arg("dim") = 52, py::arg("keepdims") = false)
        .def("min", [](const shared_ptr<Node>& n, int dim, bool keepdims = false) { return min(n, dim, keepdims); },
            py::arg("dim") = 52, py::arg("keepdims") = false)
        .def("max", [](const shared_ptr<Node>& n, int dim, bool keepdims = false) { return max(n, dim, keepdims); },
            py::arg("dim") = 52, py::arg("keepdims") = false)

        // Shape ops
        .def("reshape", [](const shared_ptr<Node>& n, vector<int> shape) { 
            return reshape(n, shape.data(), shape.size()); })
        .def("contiguous", &contiguous);

    
    // Graph wrapper
    py::class_<Graph, shared_ptr<Graph>>(m, "Graph")
        .def(py::init<shared_ptr<Node>, bool>(), py::arg("loss"), py::arg("disposable") = true)
        .def_property_readonly("weights", [](shared_ptr<Graph>& g) { 
            py::list py_lst(g->require_grad_nodes.size());
            for (auto node : g->require_grad_nodes) {
                py_lst.append(make_node_string(node));
            }
            return py_lst;
        })
        .def("eval", &Graph::eval)
        .def("build_backward", &Graph::backward)
        .def("sgd_step", &Graph::sgd_step, py::arg("learning_rate"))
        .def("detach_grads", [](shared_ptr<Graph>& g) { g->detach_grads_from_bwd_graph(); });
    
    

    // Creation ops


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
          py::arg("a"), py::arg("dim") = 52, py::arg("keepdims") = false);
    m.def("min", [](shared_ptr<Node> a, int dim, bool keepdims = false) { return min(a, dim, keepdims); },
          py::arg("a"), py::arg("dim") = 52, py::arg("keepdims") = false);
    m.def("max", [](shared_ptr<Node> a, int dim, bool keepdims = false) { return max(a, dim, keepdims); },
          py::arg("a"), py::arg("dim") = 52, py::arg("keepdims") = false);
    
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
    m.def("eval", [](shared_ptr<Node> node, bool export_graph) { eval(node, export_graph); },
          py::arg("node"), py::arg("export_graph") = false);
    m.def("debug_eval", [](shared_ptr<Node> node, bool export_graph, bool disposable) { debug_eval(node, export_graph, disposable); },
          py::arg("node"), py::arg("export_graph") = false, py::arg("disposable") = true);
    m.def("make_model", &make_model);

    m.def("sync", []() { CUDA_CHECK(); });
    m.def("mempool_stats", []() { cout << MemoryPool::call().stats() << endl; });
    m.def("mem_snapshot", [](){
        auto s = snapshot();
        print_snapshot("VRAM SUMMARY", s);
    });
    m.def("see_allocated_blocks", []() {
        print_allocated_blocks();
    });
    m.def("export_graph", [](shared_ptr<Node> node) { node->export_graph(); });
}

