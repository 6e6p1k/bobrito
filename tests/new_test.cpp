#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "../src/nodes.h"
#include "../src/ops.h"
#include "../src/evaluate.h"

using namespace std;

// ============================================================
//                       EVAL HELPER
// ============================================================
// void build_topo(Node* node, unordered_set<Node*>& visited, vector<Node*>& order) {
//     if (visited.count(node)) return;
//     visited.insert(node);
//     for (auto& inp : node->inputs)
//         build_topo(inp.get(), visited, order);
//     order.push_back(node);
// }

// void eval(shared_ptr<Node> root) {
//     unordered_set<Node*> visited;
//     vector<Node*> order;
//     build_topo(root.get(), visited, order);
//     for (Node* node : order) {
//         if (!node->is_meta()) continue;
//         int size = node->get_size();
//         if (!node->is_data()) node->d_data = make_shared<ContentData>();
//         if (!node->is_data_allocated()) node->d_data->allocate(size);
//         node->prim->eval(node);
//     }
// }

// ============================================================
//                       TEST RUNNER
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

static void check_shape(shared_ptr<Node> node, vector<int> expected) {
    check(node->get_ndim() == (int)expected.size(),
          "ndim: got " + to_string(node->get_ndim()) +
          " expected " + to_string(expected.size()));
    for (int i = 0; i < (int)expected.size(); i++)
        check(node->get_shape()[i] == expected[i],
              "shape[" + to_string(i) + "]: got " +
              to_string(node->get_shape()[i]) + " expected " +
              to_string(expected[i]));
}

static void check_values(shared_ptr<Node> node, vector<float> expected, float tol = 1e-4f) {
    check(node->is_data_allocated(), "node data not allocated");
    check(node->get_size() == (int)expected.size(),
          "size: got " + to_string(node->get_size()) +
          " expected " + to_string(expected.size()));
    float* h = node->d_data->d_to_h();
    for (int i = 0; i < (int)expected.size(); i++) {
        float diff = fabsf(h[i] - expected[i]);
        if (diff > tol) {
            string err = "data[" + to_string(i) + "]: got " +
                         to_string(h[i]) + " expected " + to_string(expected[i]);
            free(h);
            throw logic_error(err);
        }
    }
    free(h);
}

static bool cuda_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

// ============================================================
//                   SHAPE TESTS  (no CUDA needed)
// ============================================================

void test_arange_shape() {
    auto n = arange({2, 3}, 0);
    check_shape(n, {2, 3});
    check(n->get_size() == 6, "size should be 6");
}

void test_arange_1d_shape() {
    // arange(0, 10, 2) → [0,2,4,6,8], shape {5}
    auto n = arange(0, 10, 2);
    check_shape(n, {5});
}

void test_full_shape() {
    auto n = full({3, 4}, 1.0f);
    check_shape(n, {3, 4});
    check(n->get_size() == 12, "size should be 12");
}

void test_relu_shape() {
    auto a = make_leaf_node({2, 3}, vector<float>(6, 1.0f));
    check_shape(relu(a), {2, 3});
}

void test_exp_shape() {
    auto a = make_leaf_node({4}, vector<float>(4, 1.0f));
    check_shape(exp(a), {4});
}

void test_log_shape() {
    auto a = make_leaf_node({4}, vector<float>(4, 1.0f));
    check_shape(log(a), {4});
}

void test_neg_shape() {
    auto a = make_leaf_node({2, 2}, vector<float>(4, 1.0f));
    check_shape(neg(a), {2, 2});
}

void test_sqrt_shape() {
    auto a = make_leaf_node({3}, vector<float>(3, 1.0f));
    check_shape(sqrt(a), {3});
}

void test_add_same_shape() {
    auto a = make_leaf_node({2, 3}, vector<float>(6, 0.0f));
    auto b = make_leaf_node({2, 3}, vector<float>(6, 0.0f));
    check_shape(add(a, b), {2, 3});
}

void test_add_broadcast_row() {
    // [2,3] + [3] → [2,3]
    auto a = make_leaf_node({2, 3}, vector<float>(6, 0.0f));
    auto b = make_leaf_node({3},    vector<float>(3, 0.0f));
    check_shape(add(a, b), {2, 3});
}

void test_add_broadcast_both_dims() {
    // [2,1] + [1,3] → [2,3]
    auto a = make_leaf_node({2, 1}, vector<float>(2, 0.0f));
    auto b = make_leaf_node({1, 3}, vector<float>(3, 0.0f));
    check_shape(add(a, b), {2, 3});
}

void test_add_broadcast_batch() {
    // [4,2,3] + [2,3] → [4,2,3]
    auto a = make_leaf_node({4, 2, 3}, vector<float>(24, 0.0f));
    auto b = make_leaf_node({2, 3},    vector<float>(6,  0.0f));
    check_shape(add(a, b), {4, 2, 3});
}

void test_matmul_shape_2d() {
    auto a = make_leaf_node({2, 3}, vector<float>(6,  0.0f));
    auto b = make_leaf_node({3, 4}, vector<float>(12, 0.0f));
    check_shape(matmul(a, b), {2, 4});
}

void test_matmul_shape_batched() {
    // [5,2,3] @ [5,3,4] → [5,2,4]
    auto a = make_leaf_node({5, 2, 3}, vector<float>(30, 0.0f));
    auto b = make_leaf_node({5, 3, 4}, vector<float>(60, 0.0f));
    check_shape(matmul(a, b), {5, 2, 4});
}

void test_matmul_shape_broadcast_batch() {
    // [5,2,3] @ [3,4] → [5,2,4]
    auto a = make_leaf_node({5, 2, 3}, vector<float>(30, 0.0f));
    auto b = make_leaf_node({3, 4},    vector<float>(12, 0.0f));
    check_shape(matmul(a, b), {5, 2, 4});
}

void test_matmul_k_mismatch_throws() {
    auto a = make_leaf_node({2, 3}, vector<float>(6,  0.0f));
    auto b = make_leaf_node({4, 5}, vector<float>(20, 0.0f));
    bool threw = false;
    try { matmul(a, b); } catch (const logic_error&) { threw = true; }
    check(threw, "expected logic_error for k-dim mismatch");
}

void test_sum_ld_shape_2d() {
    auto a = make_leaf_node({3, 4}, vector<float>(12, 0.0f));
    check_shape(sum_ld(a), {3});
}

void test_sum_ld_shape_3d() {
    auto a = make_leaf_node({2, 3, 4}, vector<float>(24, 0.0f));
    check_shape(sum_ld(a), {2, 3});
}

void test_min_ld_shape() {
    auto a = make_leaf_node({5, 6}, vector<float>(30, 0.0f));
    check_shape(min_ld(a), {5});
}

void test_max_ld_shape() {
    auto a = make_leaf_node({5, 6}, vector<float>(30, 0.0f));
    check_shape(max_ld(a), {5});
}

void test_reshape_shape() {
    auto a = make_leaf_node({6}, vector<float>(6, 0.0f));
    int s[] = {2, 3};
    check_shape(reshape(a, s, 2), {2, 3});
}

void test_reshape_numel_mismatch_throws() {
    auto a = make_leaf_node({6}, vector<float>(6, 0.0f));
    int s[] = {2, 4};
    bool threw = false;
    try { reshape(a, s, 2); } catch (const logic_error&) { threw = true; }
    check(threw, "expected logic_error for numel mismatch");
}

void test_labels() {
    auto a = make_leaf_node({2}, vector<float>(2, 0.0f));
    a->set_label("a");
    auto b = make_leaf_node({2}, vector<float>(2, 0.0f));
    b->set_label("b");
    check(add(a, b)->label  == "a+b",      "add label");
    check(mul(a, b)->label  == "a*b",      "mul label");
    check(relu(a)->label    == "a.relu()", "relu label");
    check(neg(a)->label     == "-a",       "neg label");
    check(exp(a)->label     == "a.exp()",  "exp label");
    check(log(a)->label     == "a.log()",  "log label");
}

void test_cross_entropy_shape() {
    auto logits  = make_leaf_node({4, 3}, vector<float>(12, 0.0f));
    auto targets = make_leaf_node({4, 3}, vector<float>(12, 0.0f));
    check_shape(cross_entropy(logits, targets), {4});
}

// ============================================================
//                   EVAL TESTS  (CUDA required)
// ============================================================

void test_arange_values() {
    // arange(0, 6, 1) → [0,1,2,3,4,5]
    auto n = arange(0, 6, 1);
    eval(n);
    check_values(n, {0, 1, 2, 3, 4, 5});
}

void test_arange_step_values() {
    // arange(0, 10, 2) → [0,2,4,6,8]
    auto n = arange(0, 10, 2);
    eval(n);
    check_values(n, {0, 2, 4, 6, 8});
}

void test_full_values() {
    auto n = full({2, 3}, 7.0f);
    eval(n);
    check_values(n, {7, 7, 7, 7, 7, 7});
}

void test_add_values() {
    auto a = make_leaf_node({3}, {1.0f, 2.0f, 3.0f});
    auto b = make_leaf_node({3}, {4.0f, 5.0f, 6.0f});
    auto c = add(a, b);
    eval(c);
    check_values(c, {5, 7, 9});
}

void test_mul_values() {
    auto a = make_leaf_node({3}, {1.0f, 2.0f, 3.0f});
    auto b = make_leaf_node({3}, {4.0f, 5.0f, 6.0f});
    auto c = mul(a, b);
    eval(c);
    check_values(c, {4, 10, 18});
}

void test_add_broadcast_values() {
    // [2,3] + [3]: each row of a gets b added
    auto a = make_leaf_node({2, 3}, {1, 2, 3, 4, 5, 6});
    auto b = make_leaf_node({3},    {10, 20, 30});
    auto c = add(a, b);
    eval(c);
    check_values(c, {11, 22, 33, 14, 25, 36});
}

void test_relu_values() {
    auto a = make_leaf_node({5}, {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f});
    auto r = relu(a);
    eval(r);
    check_values(r, {0, 0, 0, 1, 2});
}

void test_neg_values() {
    auto a = make_leaf_node({3}, {1.0f, -2.0f, 0.0f});
    auto r = neg(a);
    eval(r);
    check_values(r, {-1, 2, 0});
}

void test_exp_values() {
    auto a = make_leaf_node({3}, {0.0f, 1.0f, 2.0f});
    auto r = exp(a);
    eval(r);
    check_values(r, {1.0f, 2.71828f, 7.38906f}, 1e-3f);
}

void test_log_values() {
    auto a = make_leaf_node({3}, {1.0f, (float)M_E, (float)(M_E * M_E)});
    auto r = log(a);
    eval(r);
    check_values(r, {0.0f, 1.0f, 2.0f}, 1e-4f);
}

void test_sqrt_values() {
    auto a = make_leaf_node({4}, {0.0f, 1.0f, 4.0f, 9.0f});
    auto r = sqrt(a);
    eval(r);
    check_values(r, {0, 1, 2, 3});
}

void test_sum_ld_values() {
    // [[1,2,3],[4,5,6]] → [6, 15]
    auto a = make_leaf_node({2, 3}, {1, 2, 3, 4, 5, 6});
    auto s = sum_ld(a);
    eval(s);
    check_values(s, {6, 15});
}

void test_max_ld_values() {
    // [[1,5,3],[4,2,6]] → [5, 6]
    auto a = make_leaf_node({2, 3}, {1, 5, 3, 4, 2, 6});
    auto s = max_ld(a);
    eval(s);
    check_values(s, {5, 6});
}

void test_min_ld_values() {
    // [[1,5,3],[4,2,6]] → [1, 2]
    auto a = make_leaf_node({2, 3}, {1, 5, 3, 4, 2, 6});
    auto s = min_ld(a);
    eval(s);
    check_values(s, {1, 2});
}

void test_matmul_values() {
    // [[1,2],[3,4]] @ [[1,2],[3,4]] = [[7,10],[15,22]]
    auto a = make_leaf_node({2, 2}, {1, 2, 3, 4});
    auto b = make_leaf_node({2, 2}, {1, 2, 3, 4});
    auto c = matmul(a, b);
    eval(c);
    check_values(c, {7, 10, 15, 22});
}

void test_cross_entropy_uniform_logits() {
    // logits = [[0,0,0]], targets = [[1,0,0]]
    // log_softmax = -log(3) for each class → loss = log(3) / 1
    auto logits  = make_leaf_node({1, 3}, {0.0f, 0.0f, 0.0f});
    auto targets = make_leaf_node({1, 3}, {1.0f, 0.0f, 0.0f});
    auto loss = cross_entropy(logits, targets);
    eval(loss);
    check_values(loss, {std::log(3.0f)}, 1e-4f);
}

void test_cross_entropy_perfect_prediction() {
    // logits strongly favour class 0, target is class 0 → loss ≈ 0
    auto logits  = make_leaf_node({1, 3}, {10.0f, -10.0f, -10.0f});
    auto targets = make_leaf_node({1, 3}, {1.0f,   0.0f,   0.0f});
    auto loss = cross_entropy(logits, targets);
    eval(loss);
    check_values(loss, {0.0f}, 1e-2f);
}

void test_cross_entropy_batch() {
    // logits = [[1,0],[0,1]], targets = [[1,0],[0,1]]
    // each sample: neg_loss_i = log(1 + e^-1); divided by batch=2
    float expected = std::log(1.0f + std::exp(-1.0f)) / 2.0f;
    auto logits  = make_leaf_node({2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
    auto targets = make_leaf_node({2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
    auto loss = cross_entropy(logits, targets);
    eval(loss);
    check_values(loss, {expected, expected}, 1e-4f);
}

void test_chained_eval() {
    // relu([-1,0,1] + [2,2,2]) = relu([1,2,3]) = [1,2,3]
    auto a = make_leaf_node({3}, {-1.0f, 0.0f, 1.0f});
    auto b = make_leaf_node({3}, { 2.0f, 2.0f, 2.0f});
    auto r = relu(add(a, b));
    eval(r);
    check_values(r, {1, 2, 3});
}

// ============================================================
//                           MAIN
// ============================================================
int main() {
    cout << "=== Shape / metadata tests (no CUDA needed) ===\n";
    RUN(test_arange_shape);
    RUN(test_arange_1d_shape);
    RUN(test_full_shape);
    RUN(test_relu_shape);
    RUN(test_exp_shape);
    RUN(test_log_shape);
    RUN(test_neg_shape);
    RUN(test_sqrt_shape);
    RUN(test_add_same_shape);
    RUN(test_add_broadcast_row);
    RUN(test_add_broadcast_both_dims);
    RUN(test_add_broadcast_batch);
    RUN(test_matmul_shape_2d);
    RUN(test_matmul_shape_batched);
    RUN(test_matmul_shape_broadcast_batch);
    RUN(test_matmul_k_mismatch_throws);
    RUN(test_sum_ld_shape_2d);
    RUN(test_sum_ld_shape_3d);
    RUN(test_min_ld_shape);
    RUN(test_max_ld_shape);
    RUN(test_reshape_shape);
    RUN(test_reshape_numel_mismatch_throws);
    RUN(test_labels);
    RUN(test_cross_entropy_shape);

    if (cuda_available()) {
        cout << "\n=== Eval / value tests (CUDA) ===\n";
        RUN(test_arange_values);
        RUN(test_arange_step_values);
        RUN(test_full_values);
        RUN(test_add_values);
        RUN(test_mul_values);
        RUN(test_add_broadcast_values);
        RUN(test_relu_values);
        RUN(test_neg_values);
        RUN(test_exp_values);
        RUN(test_log_values);
        RUN(test_sqrt_values);
        RUN(test_sum_ld_values);
        RUN(test_max_ld_values);
        RUN(test_min_ld_values);
        RUN(test_matmul_values);
        RUN(test_chained_eval);
        RUN(test_cross_entropy_uniform_logits);
        RUN(test_cross_entropy_perfect_prediction);
        RUN(test_cross_entropy_batch);
    } else {
        cout << "\n[SKIP] No CUDA device — skipping eval tests\n";
    }

    cout << "\n" << g_passed << "/" << g_run << " tests passed\n";
    return g_passed == g_run ? 0 : 1;
}
