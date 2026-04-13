#include "nodes.h"

#include <memory>
#include <sstream>
#include <stdexcept>

#include "shape_data.h"
#include "graph_viz.h"

using namespace std;

float* Node::get_data() { return d_data->data(); }

bool Node::is_data() { return d_data != nullptr; }

bool Node::is_label() { return label != ""; }

bool Node::is_data_allocated() {
    if (!is_data()) return false;
    return d_data->is_allocated();
}

bool Node::is_meta() { return h_meta != nullptr; }

int Node::get_size() {
    if (!is_meta()) throw logic_error("h_meta is nullptr!");
    if (h_meta->size == -1) throw logic_error("size wasn't defined!");
    return h_meta->size;
}

int Node::get_ndim() {
    if (!is_meta()) throw logic_error("h_meta is nullptr!");
    if (h_meta->ndim == -1) throw logic_error("ndim wasn't defined!");
    return h_meta->ndim;
}

int* Node::get_shape() {
    if (!is_meta()) throw logic_error("h_meta is nullptr!");
    if (h_meta->shape() == nullptr) throw logic_error("shape wasn't defined!");
    return h_meta->shape();
}

int* Node::get_strides() {
    if (!is_meta()) throw logic_error("h_meta is nullptr!");
    if (h_meta->strides() == nullptr)
        throw logic_error("strides weren't defined!");
    return h_meta->strides();
}

void Node::set_label(string new_label) { label = new_label; }

void Node::export_graph(string filename) {
    string dot_filename = filename + ".dot";
    export_dot_full(this, dot_filename.c_str());
    string command = "dot -Tpdf " + dot_filename + " -o " + filename + ".pdf";
    int status = system(command.c_str());
    if (status != 0) throw logic_error("Graph export failed.");
    else cout << "Graph was successfully exported to file '" << filename << ".pdf'\n";
}

std::shared_ptr<Node> make_node(Primitive* prim,
                                std::vector<std::shared_ptr<Node>> inputs,
                                int* shape, int ndim) {
    shared_ptr<Node> node = make_shared<Node>();
    node->prim = prim;
    node->h_meta = make_shape_data(ndim, shape);
    node->d_data = make_shared<ContentData>();

    bool is_bwd = false;
    for (auto input : inputs) {
        node->inputs.push_back(input);
        is_bwd += input->is_bwd;
    }
    node->is_bwd = is_bwd;
    if (is_bwd) {
        for (auto input : inputs)
            input->outputs.push_back(node);
    }
    return node;
}

std::shared_ptr<Node> copy_node_no_prim_before_eval(shared_ptr<Node> node) {
    shared_ptr<Node> out_node = make_shared<Node>();
    out_node->h_meta = node->h_meta;
    out_node->d_data = node->d_data;
    out_node->is_bwd = node->is_bwd;
    out_node->set_label(node->label + ".copy()");
    return out_node;
}

std::shared_ptr<Node> copy_node_no_prim_with_link(shared_ptr<Node> node) {
    auto out_node = copy_node_no_prim_before_eval(node);
    out_node->inputs = node->inputs;
    out_node->outputs = node->outputs;
    return out_node;
}

std::shared_ptr<ScalarNode> make_scalar_node(float value, string label) {
    shared_ptr<ScalarNode> scalar = make_shared<ScalarNode>();
    scalar->value = value;
    scalar->label = label;
    return scalar;
}

// ============================================================
//                         PRINTING
// ============================================================

void tensorToStringRecursive(std::ostringstream& oss, float* data, int* shape,
                             int ndim, int dim, int offset) {
    // Calculate the stride (how many floats to skip for one step in this
    // dimension)
    int sliceSize = 1;
    for (int i = dim + 1; i < ndim; ++i) {
        sliceSize *= shape[i];
    }

    oss << "[";
    for (int i = 0; i < shape[dim]; ++i) {
        if (dim == ndim - 1) {
            // Base Case: We're at the deepest level (the floats)
            oss << data[offset + i];
            if (i < shape[dim] - 1) oss << ", ";
        } else {
            // Recursive Step: Handle the next dimension down
            tensorToStringRecursive(oss, data, shape, ndim, dim + 1,
                                    offset + (i * sliceSize));

            // Formatting: Add commas and newlines
            if (i < shape[dim] - 1) {
                oss << "\n";
                // Indentation: matching the number of open brackets
                for (int space = 0; space <= dim; ++space) oss << " ";
            }
        }
    }
    oss << "]";
}

std::string make_node_string(std::shared_ptr<Node> node) {
    std::ostringstream oss;
    oss << "Node(";

    // label
    if (node->is_label()) oss << "label='" + node->label + "', ";

    // shape
    oss << "shape=[";
    for (int i = 0; i < node->get_ndim(); ++i) {
        oss << node->get_shape()[i];
        if (i != node->get_ndim() - 1) oss << ", ";
    }
    oss << "]";

    // data
    if (node->is_data_allocated()) {
        float* h_data = node->d_data->d_to_h();
        oss << ", data:\n";
        tensorToStringRecursive(oss, h_data, node->get_shape(),
                                node->get_ndim(), 0, 0);
        free(h_data);
    }

    // grad
    if (node->owned_grad && node->owned_grad->is_data_allocated()) {
        float* h_data = node->owned_grad->d_data->d_to_h();
        oss << ",\ngrad:\n";
        tensorToStringRecursive(oss, h_data, node->owned_grad->get_shape(),
                                node->owned_grad->get_ndim(), 0, 0);
        free(h_data);
    }
    oss << ")";
    return oss.str();
}

std::string make_node_string(Node* node) {
    std::ostringstream oss;
    oss << "Node(";

    // label
    if (node->is_label()) oss << "label='" + node->label + "', ";

    // shape
    oss << "shape=[";
    for (int i = 0; i < node->get_ndim(); ++i) {
        oss << node->get_shape()[i];
        if (i != node->get_ndim() - 1) oss << ", ";
    }
    oss << "]";

    // data
    if (node->is_data_allocated()) {
        float* h_data = node->d_data->d_to_h();
        oss << ", data:\n";
        tensorToStringRecursive(oss, h_data, node->get_shape(),
                                node->get_ndim(), 0, 0);
        free(h_data);
    }

    // grad
    if (node->owned_grad && node->owned_grad->is_data_allocated()) {
        float* h_data = node->owned_grad->d_data->d_to_h();
        oss << ",\ngrad:\n";
        tensorToStringRecursive(oss, h_data, node->owned_grad->get_shape(),
                                node->owned_grad->get_ndim(), 0, 0);
        free(h_data);
    }
    oss << ")";
    return oss.str();
}


void print_node(std::shared_ptr<Node> node) {
    std::string node_str = make_node_string(node);
    std::cout << node_str << std::endl;
}
