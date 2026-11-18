// test_div_grad.cpp
// Reproduces the "phantom bug": div_grad fails when mul_grad runs before it.
// Run with:  make test_div_grad && ./build/test_div_grad
//
// Uses hardcoded data so results are deterministic across runs.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../../src/allocator.cuh"
#include "../../src/evaluate.h"
#include "../../src/nodes.h"
#include "../../src/ops.h"
#include "../../src/graph_viz.h"

using namespace std;

// ============================================================
//                       HELPERS
// ============================================================
static int g_run = 0, g_passed = 0;

#define RUN(fn) do {                                                \
    g_run++;                                                        \
    try { fn(); g_passed++; cout << "[PASS] " #fn "\n"; }          \
    catch (const exception& e) {                                    \
        cout << "[FAIL] " #fn ": " << e.what() << "\n";            \
    }                                                               \
} while (0)

static void check(bool cond, const string& msg) {
    if (!cond) throw logic_error(msg);
}

static void print_array(const string& label, float* data, int size) {
    cout << label << " = [";
    for (int i = 0; i < size; i++) {
        if (i > 0) cout << ", ";
        printf("%.6f", data[i]);
    }
    cout << "]\n";
}

// Check if two host arrays are close (like np.allclose)
static void check_close(const string& name, float* actual, float* expected,
                         int size, float rtol = 1e-3f, float atol = 1e-4f) {
    float max_diff = 0;
    int max_idx = -1;
    for (int i = 0; i < size; i++) {
        float diff = fabsf(actual[i] - expected[i]);
        float tol = atol + rtol * fabsf(expected[i]);
        if (diff > tol && diff > max_diff) {
            max_diff = diff;
            max_idx = i;
        }
    }
    if (max_idx >= 0) {
        printf("  [%s] MISMATCH at [%d]: got %.6f, expected %.6f (diff=%.2e)\n",
               name.c_str(), max_idx, actual[max_idx], expected[max_idx], max_diff);
        print_array("  actual  ", actual, size);
        print_array("  expected", expected, size);
        throw logic_error(name + ": gradient mismatch");
    }
}

// Create a leaf node from host data
static shared_ptr<Node> make_leaf(vector<int> shape, vector<float> data,
                                   const string& label, bool requires_grad) {
    auto node = make_leaf_node(shape, data);
    node->set_label(label);
    node->requires_grad = requires_grad;
    return node;
}

// Eval forward+backward (disposable), sync, return grad as host array.
// Caller must free() the returned pointers.
static float* get_grad(shared_ptr<Node> leaf) {
    check(leaf->requires_grad, "not a grad leaf");
    check(leaf->owned_grad != nullptr, "owned_grad is null for " + leaf->label);
    check(leaf->owned_grad->is_data_allocated(),
          "owned_grad not allocated for " + leaf->label);
    return leaf->owned_grad->d_data->d_to_h();
}

static float* get_data(shared_ptr<Node> node) {
    check(node->is_data_allocated(), "data not allocated");
    return node->d_data->d_to_h();
}

// ============================================================
//   Hardcoded data (shape [3, 4])
//   Equivalent to np.random.seed(42); np.random.randn(3,4)
//   for mul, and np.random.seed(42); np.random.uniform(0.5, 2.0, (3,4))
//   for div.
// ============================================================

// mul test data: random normal-ish values
static vector<float> mul_a_data = {
     0.4967f, -0.1383f,  0.6477f,  1.5230f,
    -0.2342f, -0.2341f,  1.5792f,  0.7674f,
    -0.4695f,  0.5426f, -0.4634f, -0.4657f
};
static vector<float> mul_b_data = {
     0.2420f, -1.9133f, -1.7249f, -0.5623f,
    -1.0128f,  0.3142f, -0.9080f, -1.4123f,
     1.4656f, -0.2258f,  0.0675f, -1.4247f
};

// div test data: uniform [0.5, 2.0]
static vector<float> div_a_data = {
    0.8010f, 1.4028f, 0.8006f, 1.2982f,
    0.5458f, 0.5583f, 1.1889f, 0.7104f,
    0.9617f, 1.0706f, 1.0935f, 0.5563f
};
static vector<float> div_b_data = {
    0.6416f, 0.9855f, 1.4714f, 1.8077f,
    0.7472f, 0.7903f, 1.3698f, 1.0111f,
    0.9990f, 0.9988f, 1.0071f, 1.0025f
};

// ============================================================
//   GROUND TRUTH: computed analytically
// ============================================================

// mul: grad_a = b, grad_b = a
// (these are just the other input's values)

// div: grad_a[i] = 1/b[i],  grad_b[i] = -a[i]/(b[i]^2)
static vector<float> compute_div_grad_a(const vector<float>& a, const vector<float>& b) {
    vector<float> g(a.size());
    for (size_t i = 0; i < a.size(); i++) g[i] = 1.0f / b[i];
    return g;
}
static vector<float> compute_div_grad_b(const vector<float>& a, const vector<float>& b) {
    vector<float> g(a.size());
    for (size_t i = 0; i < a.size(); i++) g[i] = -a[i] / (b[i] * b[i]);
    return g;
}

// ============================================================
//   TESTS
// ============================================================

void test_mul_grad() {
    auto a = make_leaf({3, 4}, mul_a_data, "mul_A", true);
    auto b = make_leaf({3, 4}, mul_b_data, "mul_B", true);

    auto prod = mul(a, b);
    auto loss = sum(prod, 0, false);
    loss->set_label("mul_loss");

    debug_eval(loss, false, true);
    // auto g = Graph(loss, true);
    // g.backward();
    // g.eval();
    // eval(loss);
    cudaDeviceSynchronize();
    // loss->export_graph("after_eval");   
    // export_dot_full(loss.get(), "after_eval.dot");

    cout<<"\nAfter mul grad eval:\n";
    cout<<"  A grad ptr: " << a->owned_grad->get_data() << ", B grad ptr: " << b->owned_grad->get_data() << endl;

    cout<<endl;
    // print first 12 elements of grad A for sanity check
    cout<<" INSIDE " << a->owned_grad->d_data->data() <<endl;
    float tmpA[12];
    cudaMemcpy(tmpA, a->owned_grad->d_data->data(), 48, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 12; i++) printf("%.6f ", tmpA[i]);
    cout<<endl;
    cout<<" EXPECTED:\n";
    for (int i = 0; i < 12; i++) printf("%.6f ", mul_b_data[i]);
    cout<<endl<<endl;

    // print first 12 elements of grad B for sanity check
    float tmpB[12];
    cudaMemcpy(tmpB, b->owned_grad->d_data->data(), 48, cudaMemcpyDeviceToHost);
    cout<<" INSIDE " << b->owned_grad->d_data->data() <<endl;
    for (int i = 0; i < 12; i++) printf("%.6f ", tmpB[i]);
    cout<<endl;
    cout<<" EXPECTED:\n";
    for (int i = 0; i < 12; i++) printf("%.6f ", mul_a_data[i]);
    cout<<endl<<endl;

    float* ga = get_grad(a);
    float* gb = get_grad(b);

    // ground truth: grad_a = b, grad_b = a
    check_close("mul grad_a", ga, mul_b_data.data(), 12);
    check_close("mul grad_b", gb, mul_a_data.data(), 12);

    free(ga);
    free(gb);

    // prod->grad_buffer.reset();
    // loss->grad_buffer.reset();
    // loss.reset();
    // prod.reset();
    // a->owned_grad.reset();
    // b->owned_grad.reset();
    // a->grad_buffer.reset();
    // b->grad_buffer.reset();
    // a.reset();
    // b.reset();

    // print memory state after test
    print_snapshot("after mul_grad", snapshot());
}

void test_div_grad() {
    print_snapshot("BEFORE DIV_GRAD", snapshot());
    auto a = make_leaf({3, 4}, div_a_data, "div_A", true);
    auto b = make_leaf({3, 4}, div_b_data, "div_B", true);

    // print data before eval
    {
        float* ha = get_data(a);
        float* hb = get_data(b);
        print_array("div A data before eval", ha, 12);
        print_array("div B data before eval", hb, 12);
        free(ha);
        free(hb);
    }

    auto quotient = div(a, b);
    auto loss = sum(quotient, 0, false);
    loss->set_label("div_loss");

    debug_eval(loss, false, true);
    // eval(loss);
    cudaDeviceSynchronize();
    // loss->export_graph();

    // print data after eval (check for corruption)
    {
        float* ha = get_data(a);
        float* hb = get_data(b);
        print_array("div A data after eval ", ha, 12);
        print_array("div B data after eval ", hb, 12);
        free(ha);
        free(hb);
    }

    float* ga = get_grad(a);
    float* gb = get_grad(b);

    // ground truth
    auto expected_ga = compute_div_grad_a(div_a_data, div_b_data);
    auto expected_gb = compute_div_grad_b(div_a_data, div_b_data);

    print_array("div grad_a (bobr)    ", ga, 12);
    print_array("div grad_a (expected)", expected_ga.data(), 12);
    print_array("div grad_b (bobr)    ", gb, 12);
    print_array("div grad_b (expected)", expected_gb.data(), 12);

    check_close("div grad_a", ga, expected_ga.data(), 12);
    check_close("div grad_b", gb, expected_gb.data(), 12);

    print_snapshot("after div_grad", snapshot());
}

void test_div_grad_only() {
    // Same as test_div_grad but runs in isolation
    // to compare with the mul+div sequence
    test_div_grad();
}

// ============================================================
//                          MAIN
// ============================================================
int main() {
    cudaError_t err;
    cout << "=== Test 1: div_grad ALONE ===\n";
    // Run div in isolation first
    {
        auto a = make_leaf({3, 4}, div_a_data, "div_A", true);
        auto b = make_leaf({3, 4}, div_b_data, "div_B", true);
        auto quotient = div(a, b);
        auto loss = sum(quotient, 0, false);
        eval(loss);
        cudaDeviceSynchronize();

        float* ga = get_grad(a);
        float* gb = get_grad(b);
        auto expected_ga = compute_div_grad_a(div_a_data, div_b_data);
        auto expected_gb = compute_div_grad_b(div_a_data, div_b_data);

        print_array("  grad_a (bobr)    ", ga, 12);
        print_array("  grad_a (expected)", expected_ga.data(), 12);
        print_array("  grad_b (bobr)    ", gb, 12);
        print_array("  grad_b (expected)", expected_gb.data(), 12);

        try {
            check_close("div_alone grad_a", ga, expected_ga.data(), 12);
            check_close("div_alone grad_b", gb, expected_gb.data(), 12);
            cout << "[PASS] div_grad alone\n";
        } catch (const exception& e) {
            cout << "[FAIL] div_grad alone: " << e.what() << "\n";
        }
        free(ga);
        free(gb);
    }
    print_snapshot("after isolated div", snapshot());

    cout << "\n=== Test 2: mul_grad then div_grad ===\n";
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cerr << "Error occurred during mul_grad test: " << cudaGetErrorString(err) << "\n";
    } else cout << "CUDA synchronized successfully.\n";
    RUN(test_mul_grad);

    CUDA_CHECK();
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cerr << "Error occurred during mul_grad test: " << cudaGetErrorString(err) << "\n";
    } else cout << "CUDA synchronized successfully.\n";

    RUN(test_div_grad);

    cout<<"\n----------------------------------------------\n\n";

    cout << "\n" << g_passed << "/" << g_run << " passed\n";
    return g_passed == g_run ? 0 : 1;
}