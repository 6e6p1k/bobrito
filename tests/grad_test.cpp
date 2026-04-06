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
    if (!node->grad_node) {
        cout << "  FAIL " << name << ": grad_node is null" << endl;
        failed++;
        return;
    }
    if (!node->grad_node->d_data || !node->grad_node->d_data->is_allocated()) {
        cout << "  FAIL " << name << ": grad not allocated" << endl;
        failed++;
        return;
    }

    float* h_grad = node->grad_node->d_data->d_to_h();
    int size = node->grad_node->d_data->size();

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

// ============ Test 1: Simple Add ============
// C = A + B, D = sum(C, 0)
// A [3,4] = 3.0, B [3,4] = 2.0
// C [3,4] = 5.0, D [4] = 15.0
// dD = 1, dC = expand(1) = 1, dA = 1, dB = 1
void test_add_simple() {
    cout << "test_add_simple:" << endl;
    int sa[] = {3, 4}; auto A = qq::fill(2, sa, 3, true); A->label = "A";
    int sb[] = {3, 4}; auto B = qq::fill(2, sb, 2, true); B->label = "B";

    auto C = add(A, B);
    auto D = sum(C, 0);
    _backward(D);
    eval(D);
    _geval(D);

    check_forward("C = A+B", C, 5.0f);
    check_forward("D = sum(C,0)", D, 15.0f);
    check_grad("dA", A.get(), 1.0f, 12);
    check_grad("dB", B.get(), 1.0f, 12);
}

// ============ Test 2: Add with broadcast ============
// A [3,4] = 3.0, B [1,4] = 2.0
// C [3,4] = 5.0, D = sum(C, 0) -> [4] = 15.0
// dA [3,4] = 1.0, dB [1,4] = 3.0 (summed over dim 0)
void test_add_broadcast() {
    cout << "test_add_broadcast:" << endl;
    int sa[] = {3, 4}; auto A = qq::fill(2, sa, 3, true); A->label = "A";
    int sb[] = {1, 4}; auto B = qq::fill(2, sb, 2, true); B->label = "B";

    auto C = add(A, B);
    auto D = sum(C, 0);
    _backward(D);
    eval(D);
    _geval(D);

    check_forward("C = A+B", C, 5.0f);
    check_forward("D = sum(C,0)", D, 15.0f);
    check_grad("dA", A.get(), 1.0f, 12);
    check_grad("dB", B.get(), 3.0f, 4);
}

// ============ Test 3: Simple Mul ============
// A [3,4] = 3.0, B [3,4] = 2.0
// C = A * B = 6.0, D = sum(C, 0) -> [4] = 18.0
// dC = 1, dA = 1 * B = 2, dB = A * 1 = 3
void test_mul_simple() {
    cout << "test_mul_simple:" << endl;
    int sa[] = {3, 4}; auto A = qq::fill(2, sa, 3, true); A->label = "A";
    int sb[] = {3, 4}; auto B = qq::fill(2, sb, 2, true); B->label = "B";

    auto C = mul(A, B);
    auto D = sum(C, 0);
    _backward(D);
    eval(D);
    _geval(D);

    check_forward("C = A*B", C, 6.0f);
    check_forward("D = sum(C,0)", D, 18.0f);
    check_grad("dA", A.get(), 2.0f, 12);
    check_grad("dB", B.get(), 3.0f, 12);
}

// ============ Test 4: Mul with broadcast ============
// A [3,4] = 3.0, B [1,4] = 2.0
// C = A * B = 6.0, D = sum(C, 0) -> [4] = 18.0
// dC = 1.0
// dA [3,4] = dC * B = 1 * 2 = 2.0
// dB [1,4] = sum(A * dC, dim=0) = sum(3, dim=0) = 9.0
void test_mul_broadcast() {
    cout << "test_mul_broadcast:" << endl;
    int sa[] = {3, 4}; auto A = qq::fill(2, sa, 3, true); A->label = "A";
    int sb[] = {1, 4}; auto B = qq::fill(2, sb, 2, true); B->label = "B";

    auto C = mul(A, B);
    auto D = sum(C, 0);
    _backward(D);
    eval(D);
    _geval(D);

    check_forward("C = A*B", C, 6.0f);
    check_forward("D = sum(C,0)", D, 18.0f);
    check_grad("dA", A.get(), 2.0f, 12);
    check_grad("dB", B.get(), 9.0f, 4);
}

// ============ Test 5: Simple MatMul ============
// A [3,4] = 3.0, B [4,3] = 1.0
// C = A @ B = [3,3], each element = 4*3*1 = 12
// D = sum(C, 0) -> [3] = 36
// dC = 1
// dA = dC @ B^T = [3,3] @ [3,4], each = 3*1 = 3
// dB = A^T @ dC = [4,3] @ [3,3], each = 3*3 = 9
void test_matmul_simple() {
    cout << "test_matmul_simple:" << endl;
    int sa[] = {3, 4}; auto A = qq::fill(2, sa, 3, true); A->label = "A";
    int sb[] = {4, 3}; auto B = qq::fill(2, sb, 1, true); B->label = "B";

    auto C = matmul(A, B);
    auto D = sum(C, 0);
    _backward(D);
    eval(D);
    _geval(D);

    check_forward("C = A@B", C, 12.0f);
    check_forward("D = sum(C,0)", D, 36.0f);
    check_grad("dA", A.get(), 3.0f, 12);
    check_grad("dB", B.get(), 9.0f, 12);
}

// ============ Test 6: Batched MatMul ============
// A [2,3,4] = 3.0, B [4,3] = 1.0
// C = A @ B = [2,3,3], each = 12
// D = sum(C, 0) -> reduces batch dim
// dA [2,3,4] = dC @ B^T, each = 3
// dB [4,3] = sum(A^T @ dC, dim=0), each = 2*3*3 = 18
void test_matmul_batched() {
    cout << "test_matmul_batched:" << endl;
    int sa[] = {2, 3, 4}; auto A = qq::fill(3, sa, 3, true); A->label = "A";
    int sb[] = {4, 3};    auto B = qq::fill(2, sb, 1, true); B->label = "B";

    auto C = matmul(A, B);
    auto D = sum(C, 0);
    _backward(D);
    eval(D);
    _geval(D);

    check_forward("C = A@B", C, 12.0f);
    check_grad("dA", A.get(), 3.0f, 24);
    check_grad("dB", B.get(), 18.0f, 12);
}

// ============ Test 7: Chain: MatMul + Add + Mul ============
// A [3,4] = 2.0, B [4,3] = 1.0, bias [1,3] = 0.5, scale [3,3] = 3.0
// C = A @ B = [3,3], each = 8
// D = C + bias = [3,3], each = 8.5
// E = D * scale = [3,3], each = 25.5
// F = sum(E, 0) -> [3] = 76.5
// dE = 1
// dD = dE * scale = 3.0
// dC = dD = 3.0
// dbias = sum(dD, dim=0) = 9.0
// dA = dC @ B^T = [3,3] @ [3,4], each = 3*3*1 = 9
// dB = A^T @ dC = [4,3] @ [3,3], each = 3*3*2 = 18
void test_chain() {
    cout << "test_chain:" << endl;
    int sa[] = {3, 4};   auto A     = qq::fill(2, sa, 2.0, true);   A->label = "A";
    int sb[] = {4, 3};   auto B     = qq::fill(2, sb, 1.0, true);   B->label = "B";
    int sc[] = {1, 3};   auto bias  = qq::fill(2, sc, 0.5, true);   bias->label = "bias";
    int sd[] = {3, 3};   auto scale = qq::fill(2, sd, 3.0, true);   scale->label = "scale";

    auto C = matmul(A, B);     // [3,3] = 8
    auto D = add(C, bias);     // [3,3] = 8.5
    auto E = mul(D, scale);    // [3,3] = 25.5
    auto F = sum(E, 0);        // [3] = 76.5
    _backward(F);
    eval(F);
    _geval(F);

    check_forward("C = A@B", C, 8.0f);
    check_forward("D = C+bias", D, 8.5f);
    check_forward("E = D*scale", E, 25.5f);
    check_forward("F = sum(E,0)", F, 76.5f);
    check_grad("dA", A.get(), 9.0f, 12);
    check_grad("dB", B.get(), 18.0f, 12);
    check_grad("dbias", bias.get(), 9.0f, 3);
    check_grad("dscale", scale.get(), 8.5f, 9);
}

// ============ Test 8: Double matmul chain ============
// A [3,4] = 2, B [4,5] = 1, C [5,3] = 3
// D = A @ B = [3,5], each = 4*2*1 = 8
// E = D @ C = [3,3], each = 5*8*3 = 120
// F = sum(E, 0) -> [3] = 360
// dE = 1
// dD = dE @ C^T = [3,3] @ [3,5], each = 3*3 = 9
// dA = dD @ B^T = [3,5] @ [5,4], each = 5*9*1 = 45
// dB = A^T @ dD = [4,3] @ [3,5], each = 3*2*9 = 54
// dC = D^T @ dE = [5,3] @ [3,3], each = 3*8*1 = 24
void test_double_matmul() {
    cout << "test_double_matmul:" << endl;
    int sa[] = {3, 4}; auto A = qq::fill(2, sa, 2, true); A->label = "A";
    int sb[] = {4, 5}; auto B = qq::fill(2, sb, 1, true); B->label = "B";
    int sc[] = {5, 3}; auto C = qq::fill(2, sc, 3, true); C->label = "C";

    auto D = matmul(A, B);     // [3,5] = 8
    auto E = matmul(D, C);     // [3,3] = 120
    auto F = sum(E, 0);        // [3] = 360
    _backward(F);
    eval(F);
    _geval(F);

    check_forward("D = A@B", D, 8.0f);
    check_forward("E = D@C", E, 120.0f);
    check_forward("F = sum(E,0)", F, 360.0f);
    check_grad("dA", A.get(), 45.0f, 12);
    check_grad("dB", B.get(), 54.0f, 20);
    check_grad("dC", C.get(), 24.0f, 15);
}

// ============ Test 9: Matmul + bias + mul (MLP-like) ============
// x [4,3] = 1, W [3,5] = 2, b [1,5] = 0.5, s [1,5] = 3
// h = x @ W = [4,5], each = 3*1*2 = 6
// h2 = h + b = [4,5], each = 6.5
// h3 = h2 * s = [4,5], each = 19.5
// L = sum(h3, 0) -> [5] = 78
// dh3 = 1
// dh2 = dh3 * s = 3
// ds = dh3 * h2 = 6.5, but s is [1,5] so ds = sum(6.5, dim=0) = 4*6.5 = 26
// dh = dh2 = 3 (no broadcast needed)
// db = sum(dh2, dim=0) = 4*3 = 12
// dW = x^T @ dh = [3,4] @ [4,5], each = 4*1*3 = 12
// dx = dh @ W^T = [4,5] @ [5,3], each = 5*3*2 = 30
void test_mlp_layer() {
    cout << "test_mlp_layer:" << endl;
    int sx[] = {4, 3}; auto x = qq::fill(2, sx, 1, true); x->label = "x";
    int sw[] = {3, 5}; auto W = qq::fill(2, sw, 2, true); W->label = "W";
    int sb[] = {1, 5}; auto b = qq::fill(2, sb, 0.5, true); b->label = "b";
    int ss[] = {1, 5}; auto s = qq::fill(2, ss, 3, true); s->label = "s";

    auto h = matmul(x, W);
    auto h2 = add(h, b);
    auto h3 = mul(h2, s);
    auto L = sum(h3, 0);
    _backward(L);
    eval(L);
    _geval(L);

    check_forward("h = x@W", h, 6.0f);
    check_forward("h2 = h+b", h2, 6.5f);
    check_forward("h3 = h2*s", h3, 19.5f);
    check_forward("L = sum(h3,0)", L, 78.0f);
    check_grad("dx", x.get(), 30.0f, 12);
    check_grad("dW", W.get(), 12.0f, 15);
    check_grad("db", b.get(), 12.0f, 5);
    check_grad("ds", s.get(), 26.0f, 5);
}

// ============ Test 10: Same node used twice: C = A * A ============
// A [3,4] = 3
// C = A * A = 9, D = sum(C, 0) -> [4] = 27
// dC = 1
// dA = dC * A + A * dC = 2 * A = 6
// BUT: this only works if grad accumulates. If your framework
// only sets grad_node once, dA = 3 (only one path).
// This test catches that.
void test_square() {
    cout << "test_square:" << endl;
    int sa[] = {3, 4}; auto A = qq::fill(2, sa, 3, true); A->label = "A";

    auto C = mul(A, A);
    auto D = sum(C, 0);
    _backward(D);
    eval(D);
    _geval(D);

    check_forward("C = A*A", C, 9.0f);
    check_forward("D = sum(C,0)", D, 27.0f);
    // correct grad is 6.0 if accumulation works, 3.0 if only one path
    check_grad("dA (should be 2*A=6)", A.get(), 6.0f, 12);
}

// ============ Test 11: Shared node in add: C = A + A ============
// A [3,4] = 5
// C = A + A = 10, D = sum(C, 0) -> [4] = 30
// dA = 1 + 1 = 2
void test_add_self() {
    cout << "test_add_self:" << endl;
    int sa[] = {3, 4}; auto A = qq::fill(2, sa, 5, true); A->label = "A";

    auto C = add(A, A);
    auto D = sum(C, 0);
    _backward(D);
    eval(D);
    _geval(D);
    export_dot_full(D.get(), "grad_test_graph.dot");


    check_forward("C = A+A", C, 10.0f);
    check_forward("D = sum(C,0)", D, 30.0f);
    check_grad("dA (should be 2)", A.get(), 2.0f, 12);
}

// ============ Test 12: Two-layer MLP ============
// x [4,3] = 1, W1 [3,5] = 0.5, b1 [1,5] = 0, W2 [5,2] = 1, b2 [1,2] = 0
// h1 = x @ W1 + b1 = [4,5], each = 3*0.5 = 1.5
// h2 = h1 @ W2 + b2 = [4,2], each = 5*1.5*1 = 7.5
// L = sum(h2, 0) -> [2] = 30
// dh2 = 1
// dh1 = dh2 @ W2^T = [4,2] @ [2,5], each = 2*1 = 2
// dW2 = h1^T @ dh2 = [5,4] @ [4,2], each = 4*1.5*1 = 6
// db2 = sum(dh2, dim=0) = [1,2], each = 4
// dW1 = x^T @ dh1 = [3,4] @ [4,5], each = 4*1*2 = 8
// db1 = sum(dh1, dim=0) = [1,5], each = 4*2 = 8
// dx = dh1 @ W1^T = [4,5] @ [5,3], each = 5*2*0.5 = 5
void test_two_layer_mlp() {
    cout << "test_two_layer_mlp:" << endl;
    int sx[]  = {4, 3}; auto x  = qq::fill(2, sx, 1, true);   x->label = "x";
    int sw1[] = {3, 5}; auto W1 = qq::fill(2, sw1, 0.5, true); W1->label = "W1";
    int sb1[] = {1, 5}; auto b1 = qq::fill(2, sb1, 0, true);   b1->label = "b1";
    int sw2[] = {5, 2}; auto W2 = qq::fill(2, sw2, 1, true);   W2->label = "W2";
    int sb2[] = {1, 2}; auto b2 = qq::fill(2, sb2, 0, true);   b2->label = "b2";

    auto h1 = add(matmul(x, W1), b1);
    auto h2 = add(matmul(h1, W2), b2);
    auto L = sum(h2, 0);
    _backward(L);
    eval(L);
    _geval(L);

    check_forward("h1", h1, 1.5f);
    check_forward("h2", h2, 7.5f);
    check_forward("L", L, 30.0f);
    check_grad("dx", x.get(), 5.0f, 12);
    check_grad("dW1", W1.get(), 8.0f, 15);
    check_grad("db1", b1.get(), 8.0f, 5);
    check_grad("dW2", W2.get(), 6.0f, 10);
    check_grad("db2", b2.get(), 4.0f, 2);
}

int main() {
    test_add_simple();
    test_add_broadcast();
    test_mul_simple();
    test_mul_broadcast();
    test_matmul_simple();
    test_matmul_batched();    
    test_chain();

    test_double_matmul();
    test_mlp_layer();
    test_square();
    test_square();
    test_add_self();
    test_two_layer_mlp();

    cout << "\nResults: " << passed << " passed, " << failed << " failed" << endl;
    return failed > 0 ? 1 : 0;
}

