#include <memory>
#include <random>


#include "allocator.cuh"
#include "nodes.h"
#include "module.h"
#include "ops.h"
#include "kernels.cuh"
#include "stream.h"

// void init_weight(std::string initiation_type, float* d_data, int in_dim, int out_dim, int size) {
//     if (initiation_type == "kaiming_normal") {
//         _normal(d_data, 0, sqrtf(2.0f / in_dim), size);
//     } else if (initiation_type == "xavier_normal") {
//         _normal(d_data, 0, sqrtf(6.0f / (in_dim + out_dim)), size);
//     } else if (initiation_type == "xavier_normal") {

//     }
// }

// ============================================================
//                         LINEAR
// ============================================================
LinearLayer::LinearLayer(int in_dim, int out_dim) {
    auto W_data = std::make_shared<ContentData>();
    int W_size = in_dim * out_dim;
    W_data->allocate(W_size);
    _uniform(W_data->data(), sqrtf(6.0f / (in_dim + out_dim)), W_size);
    W = make_leaf_node({in_dim, out_dim}, W_data);
    W->requires_grad = true;
    W->set_label("W(" + std::to_string(in_dim) + "x" + std::to_string(out_dim) + ")");

    auto b_data = std::make_shared<ContentData>();
    int b_size = out_dim;
    b_data->allocate(b_size);
    kernels::fill<<<GRID_SIZE(b_size), BLOCK_SIZE, 0, bobr::current_stream()>>>(b_data->data(), 0.0f, b_size);
    b = make_leaf_node({1, out_dim}, b_data);
    b->requires_grad = true;
    b->set_label("b(" + std::to_string(out_dim) + ")");
}

std::shared_ptr<Node> LinearLayer::forward(std::shared_ptr<Node> x) {
    return add(matmul(x, W), b);
}

std::vector<std::shared_ptr<Node>> LinearLayer::parameters() {
    return {W, b};
}

LinearReluLayer::LinearReluLayer(int in_dim, int out_dim) {
    auto W_data = std::make_shared<ContentData>();
    int W_size = in_dim * out_dim;
    W_data->allocate(W_size);
    _normal(W_data->data(), 0, sqrtf(2.0 / in_dim), W_size);
    W = make_leaf_node({in_dim, out_dim}, W_data);
    W->requires_grad = true;
    W->set_label("W(" + std::to_string(in_dim) + "x" + std::to_string(out_dim) + ")");

    auto b_data = std::make_shared<ContentData>();
    int b_size = out_dim;
    b_data->allocate(b_size);
    kernels::fill<<<GRID_SIZE(b_size), BLOCK_SIZE, 0, bobr::current_stream()>>>(b_data->data(), 0.0f, b_size);
    b = make_leaf_node({1, out_dim}, b_data);
    b->requires_grad = true;
    b->set_label("b(" + std::to_string(out_dim) + ")");
}

std::shared_ptr<Node> LinearReluLayer::forward(std::shared_ptr<Node> x) {
    return relu(add(matmul(x, W), b));
}

std::vector<std::shared_ptr<Node>> LinearReluLayer::parameters() {
    return {W, b};
}

// ============================================================
//                       SEQUENTIAL
// ============================================================
void Sequential::add_layer(std::shared_ptr<Module> layer) {
    layers.push_back(layer);
}

std::shared_ptr<Node> Sequential::forward(std::shared_ptr<Node> x) {
    for (auto& layer : layers)
        x = layer->forward(x);
    return x;
}

std::vector<std::shared_ptr<Node>> Sequential::parameters() {
    std::vector<std::shared_ptr<Node>> params;
    for (auto& layer : layers) {
        auto p = layer->parameters();
        params.insert(params.end(), p.begin(), p.end());
    }
    return params;
}

