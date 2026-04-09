#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../src/tensor.cuh"
#include "../src/bobr.cuh"
#include "node.h"
#include "pybind11/cast.h"

namespace py = pybind11;


    // py::array_t<float> data() {
    //     if (!node->is_data_allocated())
    //         throw std::runtime_error("No data available");

    //     int ndim = node->h_meta->ndim;
    //     intstea* shape = node->h_meta->shape();
    //     int size = node->h_meta->size;
    //     float* h = node->d_data->d_to_h();

    //     std::vector<ssize_t> py_shape(shape, shape + ndim);
    //     std::vector<ssize_t> py_strides(ndim);
    //     ssize_t stride = sizeof(float);
    //     for (int i = ndim - 1; i >= 0; i--) {
    //         py_strides[i] = stride;
    //         stride *= shape[i];
    //     }

    //     auto capsule = py::capsule(h, [](void* p) { free(p); });
    //     return py::array_t<float>(py_shape, py_strides, h, capsule);
    // }

    // py::array_t<float> grad() {
    //     if (!node->grad_node || !node->grad_node->is_data_allocated())
    //         throw std::runtime_error("No grad available");

    //     int ndim = node->h_meta->ndim;
    //     int* shape = node->h_meta->shape();
    //     float* h = node->grad_node->d_data->d_to_h();

    //     std::vector<ssize_t> py_shape(shape, shape + ndim);
    //     std::vector<ssize_t> py_strides(ndim);
    //     ssize_t stride = sizeof(float);
    //     for (int i = ndim - 1; i >= 0; i--) {
    //         py_strides[i] = stride;
    //         stride *= shape[i];
    //     }

    //     auto capsule = py::capsule(h, [](void* p) { free(p); });
    //     return py::array_t<float>(py_shape, py_strides, h, capsule);
    // }

py::array_t<float> data_to_pyarray(Tensor t)
{
    if (!t.node->is_data_allocated())
        t.evaluate();

    int ndim = t.node->h_meta->ndim;
    int* shape = t.node->h_meta->shape();
    int size = t.node->h_meta->size;
    float* h = t.node->d_data->d_to_h();

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

PYBIND11_MODULE(bobr, m) {
    py::class_<Tensor>(m, "Tensor")
        // CONSTRUCTORS
        .def(py::init<std::vector<int>, bool>(),
             py::arg("shape"), py::arg("requires_grad") = false)

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

            Tensor t(shape, requires_grad);

            if (on_gpu) {
                for (int s : shape) size *= s;
                cudaMemcpy(t.node->d_data->data(), data_ptr,
                        size * sizeof(float), cudaMemcpyDeviceToDevice);
            } else {
                for (int s : shape) size *= s;
                t.node->d_data->h_to_d(data_ptr, size);
            }

            return t;
        }), py::arg("data"), py::arg("requires_grad") = false)

        // OPS
        .def("fill", &Tensor::fill)
        .def("matmul", &Tensor::matmul)
        .def("relu", &Tensor::relu)
        .def("log", &Tensor::log)
        .def("exp", &Tensor::exp)
        .def("sqrt", &Tensor::sqrt)
        .def("sum", &Tensor::sum, py::arg("dim") = -1, py::arg("keepdims") = false)
        .def("max", &Tensor::max, py::arg("dim") = -1, py::arg("keepdims") = false)


        // EVAL ETC
        .def("eval", &Tensor::evaluate)
        .def("geval", &Tensor::geval)
        .def("build_backward", &Tensor::build_backward)
        .def("sync", &Tensor::sync)

        // __OPS__
        .def("__add__", &Tensor::operator+)
        .def("__sub__", &Tensor::operator-)
        .def("__mul__", &Tensor::operator*)
        .def("__truediv__", &Tensor::operator/)
        .def("__pow__", &Tensor::pow, py::arg("exponent"))
        .def("__matmul__", &Tensor::matmul)
        .def("__neg__", &Tensor::neg)

        // Tensor.data
        .def_property_readonly("data", [](const Tensor &t) {
            return data_to_pyarray(t);
        })

        // Tensor.grad
        .def_property_readonly("grad", [](const Tensor &t) {
            if (t.node->grad_node.expired())
                throw std::runtime_error("No grad available");
            return data_to_pyarray(t);
        })

        .def_property_readonly("requires_grad", [](const Tensor &t) {
            if (t.node->kind == NodeKind::Source)
            {
                auto snode = std::static_pointer_cast<SourceNode>(t.node);
                return snode->h_requiers_grad;
            }
            return false;
        })

        .def_property_readonly("shape", [](const Tensor &t) {
            return t.shape_to_string();
        })
        
        .def_property("label", &Tensor::get_label, &Tensor::set_label)

        // REPRESENTATION
        .def("__repr__", [](const Tensor& t) {
            
            py::array_t<float> data = data_to_pyarray(t);
            std::string shape = t.shape_to_string();
            std::ostringstream oss;

            oss << "Tensor(shape = " << shape << ", data = \n" << py::str(data) << ")";
            return oss.str();
        });

    // SGD
    py::class_<SGD>(m, "SGD")
        .def(py::init([](std::vector<Tensor> tensors, float lr) {
            std::vector<std::shared_ptr<SourceNode>> params;
            for (auto& t : tensors) {
                auto snode = std::dynamic_pointer_cast<SourceNode>(t.node);
                if (!snode) throw std::runtime_error("Parameter must be a source tensor");
                params.push_back(snode);
            }
            return SGD(params, lr);
        }), py::arg("params"), py::arg("lr") = 0.01)
        .def("step", &SGD::apply)
        .def("zero_grad", &SGD::zero_grad);

    // FREE FUNCTIONS
    m.def("full", &bobr::full, py::arg("shape"), py::arg("val"), py::arg("requires_grad") = false);
    m.def("ones", &bobr::ones, py::arg("shape"), py::arg("requires_grad") = false);
    m.def("zeros", &bobr::zeros, py::arg("shape"), py::arg("requires_grad") = false);

    m.def("full_like", &bobr::full_like, py::arg("tensor"), py::arg("value"));
    m.def("ones_like", &bobr::ones_like, py::arg("tensor"));
    m.def("zeros_like", &bobr::zeros_like, py::arg("tensor"));

    m.def("memdata", &bobr::get_allocation_stats);
    m.def("memreset", &bobr::reset_memory_pool, "Lowk clears cache.");
    m.def("clear_grads", &bobr::clear_graph);
    m.def("draw_graph", &bobr::draw_graph, py::arg("tensor"), py::arg("file name.pdf") = "full_graph.pdf");
}
