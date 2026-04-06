#pragma once
#include <ostream>
#include "node.h"

struct InodeConfig {
    // node to copy from
    std::shared_ptr<Node> src = nullptr;

    // general info
    OpType op = OpType::null;
    std::string label = "";

    // ShapeData
    int* shape = nullptr;
    int* strides = nullptr;
    int ndim = -1;

    // linking
    std::shared_ptr<Node>* inputs = nullptr; 
    int num_inputs = -1;

    int param_count = -1;

    // to transform data existing node
    std::shared_ptr<Node> dst = nullptr;
    ContentPtr src_d_data = nullptr;
    ContentPtr src_grad_node = nullptr;
    bool copy_linking = false;
    bool copy_data = false;
    bool copy_grad = false;

    bool is_op() const {
        return op != OpType::null;
    }

    bool is_linking() const {
        if (num_inputs == -1 || inputs == nullptr) return false;
        if (num_inputs < 1 && num_inputs > 2) throw std::logic_error("Unacceptable number of inputs."); // cap at 2 for now
        return true;
    }

    // Checks if shape data is provided
    bool is_meta() const {
        return shape != nullptr && ndim >= 0; // ndim 0 is valid for scalars
    }

    bool is_label() const {
        return label != "";
    }

    bool is_param_count() const {
        return param_count != -1;
    }

    bool data_transfer_required() const {
        return src_d_data != nullptr || src_grad_node != nullptr;
    }

    bool data_copying_required() const {
        return src != nullptr;
    }
};
std::shared_ptr<SourceNode> make_filled_source_node(int* shape, int ndim, float value, bool requires_grad = false, std::string label = "");
std::shared_ptr<InternalNode> make_internal_node(OpType op, std::shared_ptr<Node> inputs[], int num_inputs, int* shape, int ndim, int* strides = nullptr, int param_count = 0);
std::shared_ptr<InternalNode> make_internal_node(InodeConfig cfg);
std::shared_ptr<InternalNode> copy_to_internal_node(std::shared_ptr<Node> example);
std::shared_ptr<InternalNode> copy_to_internal_node(std::shared_ptr<Node> example, bool transposed, OpType op);
std::shared_ptr<InternalNode> copy_to_internal_node(InodeConfig cfg);
void print_node(Node* node);
