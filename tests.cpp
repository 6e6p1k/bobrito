#include <cassert>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include "utils.h"
#include "grad.h"
#include "graph_viz.h"

using namespace std;

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    void name(); \
    struct name##_reg { name##_reg() { \
        cout << "  " #name "... "; \
        try { name(); tests_passed++; cout << "PASS" << endl; } \
        catch (const exception& e) { tests_failed++; cout << "FAIL: " << e.what() << endl; } \
        catch (...) { tests_failed++; cout << "FAIL (unknown exception)" << endl; } \
    } } name##_inst; \
    void name()

// ============ utils::compute_strides ============

TEST(strides_1d) {
    int shape[] = {5};
    int strides[1];
    utils::compute_strides(strides, 1, shape);
    assert(strides[0] == 1);
}

TEST(strides_2d) {
    int shape[] = {3, 4};
    int strides[2];
    utils::compute_strides(strides, 2, shape);
    assert(strides[0] == 4);  // stride along dim 0: jump 4 elements
    assert(strides[1] == 1);  // stride along dim 1: contiguous
}

TEST(strides_3d) {
    int shape[] = {2, 3, 4};
    int strides[3];
    utils::compute_strides(strides, 3, shape);
    assert(strides[0] == 12); // 3*4
    assert(strides[1] == 4);  // 4
    assert(strides[2] == 1);
}

TEST(strides_4d) {
    int shape[] = {2, 3, 4, 5};
    int strides[4];
    utils::compute_strides(strides, 4, shape);
    assert(strides[0] == 60); // 3*4*5
    assert(strides[1] == 20); // 4*5
    assert(strides[2] == 5);  // 5
    assert(strides[3] == 1);
}

// ============ utils::numel ============

TEST(numel_1d) {
    int shape[] = {5};
    assert(utils::numel(1, shape) == 5);
}

TEST(numel_2d) {
    int shape[] = {3, 4};
    assert(utils::numel(2, shape) == 12);
}

TEST(numel_3d) {
    int shape[] = {2, 3, 4};
    assert(utils::numel(3, shape) == 24);
}

TEST(numel_scalar) {
    int shape[] = {1};
    assert(utils::numel(1, shape) == 1);
}

// ============ ShapeData ============

TEST(shape_data_init) {
    ShapeData* sd = ShapeData::init(3);
    assert(sd != nullptr);
    assert(sd->ndim == 3);
    free(sd);
}

TEST(shape_data_shape_strides_access) {
    ShapeData* sd = ShapeData::init(2);
    int* s = sd->shape();
    int* st = sd->strides();
    s[0] = 3; s[1] = 4;
    st[0] = 4; st[1] = 1;
    assert(sd->shape()[0] == 3);
    assert(sd->shape()[1] == 4);
    assert(sd->strides()[0] == 4);
    assert(sd->strides()[1] == 1);
    free(sd);
}

TEST(make_shape_data_auto_strides) {
    int shape[] = {2, 3, 4};
    ShapeData* sd = make_shape_data(3, shape);
    assert(sd->ndim == 3);
    assert(sd->shape()[0] == 2);
    assert(sd->shape()[1] == 3);
    assert(sd->shape()[2] == 4);
    assert(sd->strides()[0] == 12);
    assert(sd->strides()[1] == 4);
    assert(sd->strides()[2] == 1);
    free(sd);
}

TEST(make_shape_data_custom_strides) {
    int shape[] = {3, 4};
    int strides[] = {1, 3}; // column-major
    ShapeData* sd = make_shape_data(2, shape, strides);
    assert(sd->shape()[0] == 3);
    assert(sd->shape()[1] == 4);
    assert(sd->strides()[0] == 1);
    assert(sd->strides()[1] == 3);
    free(sd);
}

// ============ check_shape_return_out_shape::elemwise ============

TEST(elemwise_same_shape) {
    int a[] = {2, 3};
    int b[] = {2, 3};
    auto [shape, ndim] = check_shape_return_out_shape::elemwise(a, b, 2, 2);
    assert(ndim == 2);
    assert(shape[0] == 2);
    assert(shape[1] == 3);
    delete[] shape;
}

TEST(elemwise_broadcast_scalar) {
    int a[] = {3, 4};
    int b[] = {1};
    auto [shape, ndim] = check_shape_return_out_shape::elemwise(a, b, 2, 1);
    assert(ndim == 2);
    assert(shape[0] == 3);
    assert(shape[1] == 4);
    delete[] shape;
}

TEST(elemwise_broadcast_row_col) {
    int a[] = {3, 1};
    int b[] = {1, 4};
    auto [shape, ndim] = check_shape_return_out_shape::elemwise(a, b, 2, 2);
    assert(ndim == 2);
    assert(shape[0] == 3);
    assert(shape[1] == 4);
    delete[] shape;
}

TEST(elemwise_broadcast_different_ndim) {
    int a[] = {2, 3, 4};
    int b[] = {4};
    auto [shape, ndim] = check_shape_return_out_shape::elemwise(a, b, 3, 1);
    assert(ndim == 3);
    assert(shape[0] == 2);
    assert(shape[1] == 3);
    assert(shape[2] == 4);
    delete[] shape;
}

TEST(elemwise_broadcast_complex) {
    int a[] = {1, 3, 1};
    int b[] = {2, 1, 4};
    auto [shape, ndim] = check_shape_return_out_shape::elemwise(a, b, 3, 3);
    assert(ndim == 3);
    assert(shape[0] == 2);
    assert(shape[1] == 3);
    assert(shape[2] == 4);
    delete[] shape;
}

TEST(elemwise_incompatible_throws) {
    int a[] = {2, 3};
    int b[] = {2, 4};
    bool threw = false;
    try {
        auto [shape, ndim] = check_shape_return_out_shape::elemwise(a, b, 2, 2);
        delete[] shape;
    } catch (const logic_error&) {
        threw = true;
    }
    assert(threw);
}

// ============ check_shape_return_out_shape::reduction ============

TEST(reduction_dim0) {
    int shape[] = {2, 3, 4};
    auto [res, ndim] = check_shape_return_out_shape::reduction(shape, 24, 3, 0, false);
    assert(ndim == 2);
    assert(res[0] == 3);
    assert(res[1] == 4);
    delete[] res;
}

TEST(reduction_dim1) {
    int shape[] = {2, 3, 4};
    auto [res, ndim] = check_shape_return_out_shape::reduction(shape, 24, 3, 1, false);
    assert(ndim == 2);
    assert(res[0] == 2);
    assert(res[1] == 4);
    delete[] res;
}

TEST(reduction_dim2) {
    int shape[] = {2, 3, 4};
    auto [res, ndim] = check_shape_return_out_shape::reduction(shape, 24, 3, 2, false);
    assert(ndim == 2);
    assert(res[0] == 2);
    assert(res[1] == 3);
    delete[] res;
}

TEST(reduction_keepdims) {
    int shape[] = {2, 3, 4};
    auto [res, ndim] = check_shape_return_out_shape::reduction(shape, 24, 3, 1, true);
    assert(ndim == 3);
    assert(res[0] == 2);
    assert(res[1] == 1);
    assert(res[2] == 4);
    delete[] res;
}

TEST(reduction_negative_dim) {
    int shape[] = {2, 3, 4};
    auto [res, ndim] = check_shape_return_out_shape::reduction(shape, 24, 3, -1, false);
    assert(ndim == 2);
    assert(res[0] == 2);
    assert(res[1] == 3);
    delete[] res;
}

// ============ Node-level elemwise (via make_shape_data, no GPU) ============

TEST(elemwise_node_same_shape) {
    int s1[] = {2, 3};
    int s2[] = {2, 3};
    Node a, b;
    a.kind = NodeKind::Source; b.kind = NodeKind::Source;
    a.h_meta = make_shape_data(2, s1);
    b.h_meta = make_shape_data(2, s2);

    auto [shape, ndim] = check_shape_return_out_shape::elemwise(&a, &b);
    assert(ndim == 2);
    assert(shape[0] == 2);
    assert(shape[1] == 3);
    delete[] shape;
    free(a.h_meta);
    free(b.h_meta);
}

TEST(elemwise_node_broadcast) {
    int s1[] = {2, 3, 4};
    int s2[] = {1, 4};
    Node a, b;
    a.kind = NodeKind::Source; b.kind = NodeKind::Source;
    a.h_meta = make_shape_data(3, s1);
    b.h_meta = make_shape_data(2, s2);

    auto [shape, ndim] = check_shape_return_out_shape::elemwise(&a, &b);
    assert(ndim == 3);
    assert(shape[0] == 2);
    assert(shape[1] == 3);
    assert(shape[2] == 4);
    delete[] shape;
    free(a.h_meta);
    free(b.h_meta);
}

// ============ Node-level reduction ============

TEST(reduction_node_dim1) {
    int s[] = {2, 3, 4};
    Node a;
    a.kind = NodeKind::Source;
    a.h_meta = make_shape_data(3, s);

    auto [shape, ndim] = check_shape_return_out_shape::reduction(&a, 1, false);
    assert(ndim == 2);
    assert(shape[0] == 2);
    assert(shape[1] == 4);
    delete[] shape;
    free(a.h_meta);
}

TEST(reduction_node_keepdims) {
    int s[] = {2, 3, 4};
    Node a;
    a.kind = NodeKind::Source;
    a.h_meta = make_shape_data(3, s);

    auto [shape, ndim] = check_shape_return_out_shape::reduction(&a, 0, true);
    assert(ndim == 3);
    assert(shape[0] == 1);
    assert(shape[1] == 3);
    assert(shape[2] == 4);
    delete[] shape;
    free(a.h_meta);
}

// ============ make_internal_node ============

TEST(make_internal_node_basic) {
    int s1[] = {2, 3};
    int s2[] = {2, 3};
    Node a, b;
    a.kind = NodeKind::Source; b.kind = NodeKind::Source;
    a.h_meta = make_shape_data(2, s1);
    b.h_meta = make_shape_data(2, s2);

    int out_shape[] = {2, 3};
    Node* inputs[] = {&a, &b};
    Node* result = make_internal_node(OpType::Add, inputs, 2, out_shape, 2);

    InternalNode* inode = static_cast<InternalNode*>(result);
    assert(inode->op == OpType::Add);
    assert(inode->num_inputs == 2);
    assert(inode->inputs[0] == &a);
    assert(inode->inputs[1] == &b);
    assert(inode->h_meta->ndim == 2);
    assert(inode->h_meta->shape()[0] == 2);
    assert(inode->h_meta->shape()[1] == 3);

    free(inode->h_meta);
    delete inode;
    free(a.h_meta);
    free(b.h_meta);
}

// ============ Graph visualization demo ============

void viz_demo() {
    // Build a small computation graph:
    //   result = sum( add(x, mul(y, z)), dim=1 )

    // Leaf nodes (manually, no GPU)
    int sx[] = {2, 3, 4};
    int sy[] = {1, 4};
    int sz[] = {4};

    SourceNode x_node, y_node, z_node;
    x_node.kind = NodeKind::Source;
    x_node.h_meta = make_shape_data(3, sx);
    x_node.h_requiers_grad = true;

    y_node.kind = NodeKind::Source;
    y_node.h_meta = make_shape_data(2, sy);
    y_node.h_requiers_grad = false;

    z_node.kind = NodeKind::Source;
    z_node.h_meta = make_shape_data(1, sz);
    z_node.h_requiers_grad = true;

    // mul(y, z) -> shape [1, 4]
    int mul_shape[] = {1, 4};
    Node* mul_inputs[] = {&y_node, &z_node};
    Node* mul_node = make_internal_node(OpType::Mul, mul_inputs, 2, mul_shape, 2);
    // Node* mul_node = mul(&y_node, &z_node);

    // add(x, mul_node) -> shape [2, 3, 4]  (broadcast)
    int add_shape[] = {2, 3, 4};
    Node* add_inputs[] = {&x_node, mul_node};
    Node* add_node = make_internal_node(OpType::Add, add_inputs, 2, add_shape, 3);
    // Node* add_node = add(&x_node, mul_node);

    // sum(add_node, dim=1) -> shape [2, 4]
    int sum_shape[] = {2, 4};
    Node* sum_inputs[] = {add_node};
    Node* sum_node = make_internal_node(OpType::Sum, sum_inputs, 1, sum_shape, 2);
    // Node* sum_node = sum(add_node, 1);

    visualize(sum_node);
    export_dot(sum_node, "graph.dot");
    backward_viz(sum_node);
}

int main() {
    cout << "\nResults: " << tests_passed << " passed, " << tests_failed << " failed" << endl;

    if (tests_failed == 0) {
        viz_demo();
    }

    return tests_failed > 0 ? 1 : 0;
}
