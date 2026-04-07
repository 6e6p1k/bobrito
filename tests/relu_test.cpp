#include "graph_viz.h"
#include "node.h"
#include "backward.h"
#include "allocator.cuh"
#include "ops.cuh"
#include "utils.h"
#include "visualize_graph.h"
#include <iostream>
#include <cmath>
#include <cassert>

using namespace std;

static int passed = 0;
static int failed = 0;

void check_grad(const char* name, Node* node, float expected, int expected_size) {
    auto grad = node->grad_node.lock();
    if (!grad) {
        cout << "  FAIL " << name << ": grad_node is null" << endl;
        failed++;
        return;
    }
    if (!grad->d_data || !grad->d_data->is_allocated()) {
        cout << "  FAIL " << name << ": grad not allocated" << endl;
        failed++;
        return;
    }

    float* h_grad = grad->d_data->d_to_h();
    int size = grad->d_data->size();

    if (size != expected_size) {
        cout << "  FAIL " << name << ": expected size " << expected_size << " got " << size << endl;
        free(h_grad);
        failed++;
        return;
    }

    for (int i = 0; i < size; i++) {
        if (fabs(h_grad[i] - expected) > 1e-4) {
            cout << "  FAIL " << name << ": element " << i << " expected " << expected << " got " << h_grad[i] << endl;
            free(h_grad);
            failed++;
            return;
        }
    }
    cout << "  PASS " << name << endl;
    free(h_grad);
    passed++;
}

void check_forward(const char* name, shared_ptr<Node> node, float expected) {
    if (!node->d_data || !node->d_data->is_allocated()) {
        cout << "  FAIL " << name << ": data not allocated" << endl;
        failed++;
        return;
    }
    float* h_data = node->d_data->d_to_h();
    int size = node->d_data->size();
    for (int i = 0; i < size; i++) {
        if (fabs(h_data[i] - expected) > 1e-4) {
            cout << "  FAIL " << name << ": element " << i << " expected " << expected << " got " << h_data[i] << endl;
            free(h_data);
            failed++;
            return;
        }
    }
    cout << "  PASS " << name << endl;
    free(h_data);
    passed++;
}

// ============ Test: ReLU forward and backward ============
// A [3,4] = {-2, 3} alternating? Nah, fill is uniform.
// Need to test with mixed positive/negative values.
// 
// A [2,3] with manual data: [-1, 2, -3, 4, -5, 6]
// relu(A) = [0, 2, 0, 4, 0, 6]
// L = sum(relu(A), 0) -> [3] = [4, -3, 12]... 
//
// Easier: use two sources and subtract to get mixed signs.
// A [3,4] = 3, B [3,4] = 5
// C = A - B... you don't have sub.
//
// Simplest: A [3,4] = 3, all positive
// relu(A) = 3, L = sum(relu(A), 0) = [9,9,9,9]
// dA = 1 (all positive, mask is all 1s)
//
// But that doesn't test the zero-masking. Need negative values.
// Use mul: A [3,4] = -2, B [3,4] = 1
// C = A * B = -2, relu(C) = 0
// L = sum(relu(C), 0) = 0
// drelu = 1, but mask is all 0 (input was negative)
// dC = 0, dA = 0, dB = 0

void test_relu_all_positive() {
    cout << "test_relu_all_positive:" << endl;
    int sa[] = {3, 4}; auto A = qq::fill(2, sa, 3, true); A->label = "A";

    auto B = relu(A);
    auto L = sum(B, 0);
    _backward(L);
    eval(L);
    export_dot_full(L.get(), "relu_test_graph.dot");
    _geval(L);

    check_forward("relu(A) all positive", B, 3.0f);
    check_forward("L = sum(relu(A),0)", L, 9.0f);
    check_grad("dA (all positive, mask=1)", A.get(), 1.0f, 12);
}

void test_relu_all_negative() {
    cout << "test_relu_all_negative:" << endl;
    int sa[] = {3, 4}; auto A = qq::fill(2, sa, -3, true); A->label = "A";

    auto B = relu(A);
    auto L = sum(B, 0);
    _backward(L);
    eval(L);
    _geval(L);

    check_forward("relu(A) all negative", B, 0.0f);
    check_forward("L = sum(relu(A),0)", L, 0.0f);
    check_grad("dA (all negative, mask=0)", A.get(), 0.0f, 12);
}

void test_relu_in_chain() {
    cout << "test_relu_in_chain:" << endl;
    // x [4,3] = 1, W [3,5] = -0.5, b [1,5] = 2
    // h = x @ W + b = [4,5], each = 3*1*(-0.5) + 2 = -1.5 + 2 = 0.5
    // r = relu(h) = 0.5 (all positive)
    // L = sum(r, 0) -> [5] = 2
    // dr = 1, mask all 1s so dh = 1
    // db = sum(dh, dim=0) = 4
    // dW = x^T @ dh = [3,4] @ [4,5], each = 4*1*1 = 4
    // dx = dh @ W^T = [4,5] @ [5,3], each = 5*1*(-0.5) = -2.5
    int sx[] = {4, 3}; auto x = qq::fill(2, sx, 1, true); x->label = "x";
    int sw[] = {3, 5}; auto W = qq::fill(2, sw, -0.5, true); W->label = "W";
    int sb[] = {1, 5}; auto b = qq::fill(2, sb, 2, true); b->label = "b";

    auto h = add(matmul(x, W), b);
    auto r = relu(h);
    auto L = sum(r, 0);
    _backward(L);
    eval(L);
    _geval(L);

    check_forward("h = x@W+b", h, 0.5f);
    check_forward("relu(h)", r, 0.5f);
    check_forward("L", L, 2.0f);
    check_grad("dx", x.get(), -2.5f, 12);
    check_grad("dW", W.get(), 4.0f, 15);
    check_grad("db", b.get(), 4.0f, 5);
}

void test_relu_kills_grad() {
    cout << "test_relu_kills_grad:" << endl;
    // x [4,3] = 1, W [3,5] = -0.5, b [1,5] = 0
    // h = x @ W + b = [4,5], each = -1.5
    // r = relu(h) = 0 (all killed)
    // L = sum(r, 0) = 0
    // Everything below relu gets zero grad
    int sx[] = {4, 3}; auto x = qq::fill(2, sx, 1, true); x->label = "x";
    int sw[] = {3, 5}; auto W = qq::fill(2, sw, -0.5, true); W->label = "W";
    int sb[] = {1, 5}; auto b = qq::fill(2, sb, 0, true); b->label = "b";

    auto h = add(matmul(x, W), b);
    auto r = relu(h);
    auto L = sum(r, 0);
    _backward(L);
    eval(L);
    _geval(L);

    check_forward("h = x@W+b", h, -1.5f);
    check_forward("relu(h)", r, 0.0f);
    check_forward("L", L, 0.0f);
    check_grad("dx (killed by relu)", x.get(), 0.0f, 12);
    check_grad("dW (killed by relu)", W.get(), 0.0f, 15);
    check_grad("db (killed by relu)", b.get(), 0.0f, 5);
}


int main() {
    test_relu_all_positive();
    test_relu_all_negative();
    test_relu_in_chain();
    test_relu_kills_grad();

    cout << "\nResults: " << passed << " passed, " << failed << " failed" << endl;
    return failed > 0 ? 1 : 0;
}

