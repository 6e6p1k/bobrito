#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "../src/graph_viz.h"
#include "../src/evaluate.h"
#include "../src/nodes.h"
#include "../src/ops.h"

using namespace std;

// ============================================================
//                       HELPERS
// ============================================================
static int g_run = 0, g_passed = 0;

#define RUN(fn) do {                                               \
    g_run++;                                                       \
    try { fn(); g_passed++; cout << "[PASS] " #fn "\n"; }         \
    catch (const exception& e) {                                   \
        cout << "[FAIL] " #fn ": " << e.what() << "\n";           \
    }                                                              \
} while (0)

static void check(bool cond, const string& msg) {
    if (!cond) throw logic_error(msg);
}

// Build topo order and eval each node (allocating memory as needed).
// Used to explicitly eval a subgraph (e.g. a grad node).
static void run_eval(shared_ptr<Node> root) {
    unordered_set<Node*> visited;
    vector<Node*> order;
    function<void(Node*)> topo = [&](Node* node) {
        if (visited.count(node)) return;
        visited.insert(node);
        for (auto& inp : node->inputs) topo(inp.get());
        order.push_back(node);
    };
    topo(root.get());
    for (Node* node : order) {
        if (!node->is_meta()) continue;
        if (!node->is_data()) node->d_data = make_shared<ContentData>();
        if (!node->is_data_allocated()) node->d_data->allocate(node->get_size());
        node->prim->eval(node);
    }
}

static void check_values(shared_ptr<Node> node, vector<float> expected,
                         float tol = 1e-4f) {
    check(node->is_data_allocated(), "node data not allocated");
    check(node->get_size() == (int)expected.size(),
          "size: got " + to_string(node->get_size()) +
          " expected " + to_string(expected.size()));
    float* h = node->d_data->d_to_h();
    for (int i = 0; i < (int)expected.size(); i++) {
        float diff = fabsf(h[i] - expected[i]);
        if (diff > tol) {
            string err = "data[" + to_string(i) + "]: got " + to_string(h[i]) +
                         " expected " + to_string(expected[i]);
            free(h);
            throw logic_error(err);
        }
    }
    free(h);
}

// Shorthand: make a leaf node with label and optional requires_grad
static shared_ptr<Node> leaf(vector<int> s, vector<float> d, string l,
                             bool rg = false) {
    auto n = make_leaf_node(s, d);
    n->set_label(l);
    n->requires_grad = rg;
    return n;
}

// Build the backward graph without running it (no CUDA work).
static void build_bwd_graph(shared_ptr<Node> loss) {
    Graph g(loss);
    g.backward();
}

static bool cuda_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

// ============================================================
//           BACKWARD CONSTRUCTION TESTS  (no CUDA)
// ============================================================

void test_backward_sets_owned_grad_add() {
    auto a = leaf({2, 3}, vector<float>(6, 1.0f), "a", true);
    auto b = leaf({2, 3}, vector<float>(6, 2.0f), "b", true);
    auto c = add(a, b);
    build_bwd_graph(c);
    check(a->owned_grad != nullptr, "a->owned_grad should be set");
    check(b->owned_grad != nullptr, "b->owned_grad should be set");
}

void test_backward_sets_owned_grad_mul() {
    auto a = leaf({3}, {1.0f, 2.0f, 3.0f}, "a", true);
    auto b = leaf({3}, {4.0f, 5.0f, 6.0f}, "b", true);
    auto c = mul(a, b);
    build_bwd_graph(c);
    check(a->owned_grad != nullptr, "a->owned_grad should be set after mul backward");
    check(b->owned_grad != nullptr, "b->owned_grad should be set after mul backward");
}

void test_backward_deep_chain() {
    // c = neg(add(a, b)) — two levels deep
    auto a = leaf({2}, {1.0f, 2.0f}, "a", true);
    auto b = leaf({2}, {3.0f, 4.0f}, "b", true);
    auto c = neg(add(a, b));
    build_bwd_graph(c);
    check(a->owned_grad != nullptr, "a->owned_grad in neg(add) chain");
    check(b->owned_grad != nullptr, "b->owned_grad in neg(add) chain");
}

void test_backward_throws_on_leaf() {
    auto a = leaf({2}, {1.0f, 2.0f}, "a", true);  // requires_grad=true → IS a leaf
    bool threw = false;
    try { build_bwd_graph(a); } catch (const logic_error&) { threw = true; }
    check(threw, "backward() on a requires_grad leaf should throw");
}

void test_backward_noop_without_grad_leaves() {
    // nothing requires grad -> Graph::backward() has nothing to build and returns quietly
    auto a = leaf({2}, {1.0f, 2.0f}, "a", false);
    auto b = leaf({2}, {3.0f, 4.0f}, "b", false);
    auto c = add(a, b);
    build_bwd_graph(c);
    check(a->owned_grad == nullptr && b->owned_grad == nullptr,
          "no grads should be built when no leaf requires_grad");
}

void test_backward_owned_grad_shape_matches_input() {
    auto a = leaf({2, 3}, vector<float>(6, 1.0f), "a", true);
    auto b = leaf({2, 3}, vector<float>(6, 1.0f), "b", true);
    auto c = mul(a, b);
    build_bwd_graph(c);
    // grad shape must match the input's shape
    check(a->owned_grad->get_ndim() == a->get_ndim(),
          "grad ndim should match input ndim");
    for (int i = 0; i < a->get_ndim(); i++)
        check(a->owned_grad->get_shape()[i] == a->get_shape()[i],
              "grad shape[" + to_string(i) + "] should match input");
}

// ============================================================
//               GRAD VALUE TESTS  (CUDA required)
// ============================================================

// ------ ADD ------
// c = a + b,  dc/da = 1,  dc/db = 1
void test_add_grad() {
    auto a = leaf({3}, {1.0f, 2.0f, 3.0f}, "a", true);
    auto b = leaf({3}, {4.0f, 5.0f, 6.0f}, "b", true);
    auto c = add(a, b);
    eval(c);
    check_values(a->owned_grad, {1, 1, 1});
    check_values(b->owned_grad, {1, 1, 1});
}

// ------ ADD (batched / broadcast seed) ------
void test_add_grad_2d() {
    auto a = leaf({2, 3}, vector<float>(6, 1.0f), "a", true);
    auto b = leaf({2, 3}, vector<float>(6, 2.0f), "b", true);
    auto c = add(a, b);
    eval(c);
    check_values(a->owned_grad, {1, 1, 1, 1, 1, 1});
    check_values(b->owned_grad, {1, 1, 1, 1, 1, 1});
}

// ------ SUB ------
// c = a - b,  dc/da = 1 (skip),  dc/db = -1 (neg then eval separately)
void test_sub_grad_a() {
    auto a = leaf({3}, {1.0f, 2.0f, 3.0f}, "a", true);
    auto b = leaf({3}, {4.0f, 5.0f, 6.0f}, "b", true);
    auto c = sub(a, b);
    eval(c);
    // dc/da = 1 via skip — shares data with seed
    check_values(a->owned_grad, {1, 1, 1});
}

void test_sub_grad_b() {
    auto a = leaf({3}, {1.0f, 2.0f, 3.0f}, "a", true);
    auto b = leaf({3}, {4.0f, 5.0f, 6.0f}, "b", true);
    auto c = sub(a, b);
    eval(c);
    // dc/db = -1 via neg(seed)
    check_values(b->owned_grad, {-1, -1, -1});
}

// ------ MUL ------
// c = a * b,  dc/da = b,  dc/db = a
// vjp: out[0] = mul(upstream, b), out[1] = mul(upstream, a)
void test_mul_grad_a() {
    auto a = leaf({3}, {1.0f, 2.0f, 3.0f}, "a", true);
    auto b = leaf({3}, {4.0f, 5.0f, 6.0f}, "b", true);
    auto c = mul(a, b);
    eval(c);
    // dc/da = b = [4, 5, 6]
    check_values(a->owned_grad, {4, 5, 6});
}

void test_mul_grad_b() {
    auto a = leaf({3}, {1.0f, 2.0f, 3.0f}, "a", true);
    auto b = leaf({3}, {4.0f, 5.0f, 6.0f}, "b", true);
    auto c = mul(a, b);
    eval(c);
    // dc/db = a = [1, 2, 3]
    check_values(b->owned_grad, {1, 2, 3});
}

// ------ NEG ------
// c = -a,  dc/da = -1
void test_neg_grad() {
    auto a = leaf({3}, {1.0f, 2.0f, 3.0f}, "specific_a", true);
    auto c = neg(a);
    eval(c);
    check_values(a->owned_grad, {-1, -1, -1});
}

// ------ LOG ------
// c = log(a),  dc/da = 1/a
void test_log_grad() {
    auto a = leaf({3}, {1.0f, 2.0f, 4.0f}, "a", true);
    auto c = log(a);
    eval(c);
    // dc/da = 1/a = [1, 0.5, 0.25]
    check_values(a->owned_grad, {1.0f, 0.5f, 0.25f});
}

// ------ SUM_LD ------
// s = sum_ld(a) where a is [2,3]
// ds/da: each element in a row gets gradient 1 → expand_ld(ones[2]) = ones[2,3]
void test_sum_ld_grad() {
    auto a = leaf({2, 3}, {1, 2, 3, 4, 5, 6}, "a", true);
    auto s = sum_ld(a);
    eval(s);
    check_values(a->owned_grad, {1, 1, 1, 1, 1, 1});
}

// ------ CHAIN: neg(add(a, b)) ------
// loss = -(a + b),  dloss/da = -1,  dloss/db = -1
void test_chain_neg_add_grad() {
    auto a = leaf({3}, {1.0f, 0.0f, -1.0f}, "a", true);
    auto b = leaf({3}, {2.0f, 2.0f,  2.0f}, "b", true);
    auto loss = neg(add(a, b));
    eval(loss);
    check_values(a->owned_grad, {-1, -1, -1});
    check_values(b->owned_grad, {-1, -1, -1});
}

// ------ CHAIN: mul then add (shared input) ------
// loss = a*b + a,  dloss/da = b + 1,  dloss/db = a
// This tests gradient accumulation when a is used twice.
void test_shared_input_grad() {
    auto a = leaf({3}, {1.0f, 2.0f, 3.0f}, "a", true);
    auto b = leaf({3}, {4.0f, 5.0f, 6.0f}, "b", true);
    auto loss = add(mul(a, b), a);  // a*b + a
    eval(loss);
    // dloss/da = b + 1 = [5, 6, 7]
    // dloss/db = a     = [1, 2, 3]
    check_values(a->owned_grad, {5, 6, 7});
    check_values(b->owned_grad, {1, 2, 3});
}

// ------ MATMUL ------
// C = A @ B,  dL/dA = upstream @ B^T,  dL/dB = A^T @ upstream
// A = ones[2,2], B = ones[2,2], upstream = ones[2,2]
// dL/dA = ones[2,2] @ ones^T[2,2] = [[2,2],[2,2]]
// dL/dB = ones^T[2,2] @ ones[2,2] = [[2,2],[2,2]]
//
// NOTE: vjp uses copy_node_no_prim which returns the original node and then
// sets transposed=true on it — mutating the original A/B. This is a known bug
// that will corrupt the forward data. Test is written for correct math.
void test_matmul_grad() {
    auto a = leaf({2, 2}, {1, 1, 1, 1}, "a", true);
    auto b = leaf({2, 2}, {1, 1, 1, 1}, "b", true);
    auto c = matmul(a, b);
    eval(c);
    check_values(a->owned_grad, {2, 2, 2, 2});
    check_values(b->owned_grad, {2, 2, 2, 2});
}

// ------ TRANSPOSE (3D) ------
// [2, 2, 3] -> [2, 3, 2]: last two dims swapped, each batch on its own
void test_transpose_3d_values() {
    auto a = leaf({2, 2, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, "a");
    auto t = transpose(a);
    run_eval(t);
    check_values(t, {0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8, 11});
}

// ------ SUM over a non-last dim ------
// [2, 2, 3] summed over dim 0 -> [2, 3], every input gets grad 1
void test_sum_dim0_3d() {
    auto a = leaf({2, 2, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, "a", true);
    auto s = sum(a, 0, false);
    eval(s);
    check_values(s, {6, 8, 10, 12, 14, 16});
    check_values(a->owned_grad, vector<float>(12, 1.0f));
}

// ------ MAX over a middle dim, keepdims ------
// [2, 2, 2] max over dim 1 -> [2, 1, 2], grad goes only to the max elements
void test_max_dim1_keepdims_3d() {
    auto a = leaf({2, 2, 2}, {1, 5, 3, 2, 7, 0, 4, 8}, "a", true);
    auto m = max(a, 1, true);
    eval(m);
    check(m->get_ndim() == 3 && m->get_shape()[1] == 1, "keepdims should keep dim 1 as 1");
    check_values(m, {3, 5, 7, 8});
    check_values(a->owned_grad, {0, 1, 1, 0, 1, 0, 0, 1});
}

// ------ SUM over all dims, keepdims ------
// no dim given (52) + keepdims -> shape [1, 1, 1], value is the total
void test_sum_all_keepdims() {
    auto a = leaf({2, 2, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, "a", true);
    auto s = sum(a, 52, true);
    eval(s);
    check(s->get_ndim() == 3 && s->get_size() == 1, "keepdims should give shape [1, 1, 1]");
    check_values(s, {66});
    check_values(a->owned_grad, vector<float>(12, 1.0f));
}

// ------ reduction dim out of range ------
void test_reduction_dim_out_of_range() {
    auto a = leaf({2, 3}, {1, 2, 3, 4, 5, 6}, "a");
    bool threw_pos = false, threw_neg = false;
    try { sum(a, 2, false); } catch (const logic_error&) { threw_pos = true; }
    try { max(a, -3, true); } catch (const logic_error&) { threw_neg = true; }
    check(threw_pos && threw_neg, "dim outside [-ndim, ndim) should throw");
}

// ------ ContentData ------
// moved-from data reports "nothing allocated", d_to_d allocates and sets the size
void test_content_data_move_and_d_to_d() {
    ContentData a;
    a.allocate(4);
    ContentData b(std::move(a));
    check(a.b_ptr == nullptr && a._size == 0, "moved-from ContentData should be empty");
    check(b.size() == 4, "move should keep the size");

    bool threw = false;
    try { a.size(); } catch (const logic_error&) { threw = true; }
    check(threw, "size() on a moved-from ContentData should throw");

    ContentData c;
    c.allocate(2);
    c = std::move(b);  // c's old block is freed, c takes b's block
    check(c.size() == 4 && b.b_ptr == nullptr, "move assignment should take the block");

    ContentData d;
    d.d_to_d(c.data(), 4);  // not allocated yet -> d_to_d allocates
    check(d.size() == 4, "d_to_d should set the size when it allocates");
}

// ============================================================
//                          MAIN
// ============================================================
int main() {
    cout << "=== Backward construction tests (no CUDA needed) ===\n";
    RUN(test_backward_sets_owned_grad_add);
    RUN(test_backward_sets_owned_grad_mul);
    RUN(test_backward_deep_chain);
    RUN(test_backward_throws_on_leaf);
    RUN(test_backward_noop_without_grad_leaves);
    RUN(test_backward_owned_grad_shape_matches_input);

    if (cuda_available()) {
        cout << "\n=== Grad value tests (CUDA) ===\n";
        RUN(test_add_grad);
        RUN(test_add_grad_2d);
        RUN(test_sub_grad_a);
        RUN(test_sub_grad_b);
        RUN(test_mul_grad_a);
        RUN(test_mul_grad_b);
        RUN(test_neg_grad);
        RUN(test_log_grad);
        RUN(test_sum_ld_grad);
        RUN(test_chain_neg_add_grad);
        RUN(test_shared_input_grad);
        RUN(test_matmul_grad);
        RUN(test_transpose_3d_values);
        RUN(test_sum_dim0_3d);
        RUN(test_max_dim1_keepdims_3d);
        RUN(test_sum_all_keepdims);
        RUN(test_reduction_dim_out_of_range);
        RUN(test_content_data_move_and_d_to_d);
    } else {
        cout << "\n[SKIP] No CUDA device — skipping grad value tests\n";
    }

    cout << "\n" << g_passed << "/" << g_run << " tests passed\n";

    if (g_passed < g_run) {
        cout << "\nKnown bugs that may cause failures:\n";
        cout << "  - copy_node_no_prim returns original node instead of copy\n";
        cout << "    -> matmul vjp mutates transposed flag on original inputs\n";
        cout << "  - exp vjp computes x*grad instead of exp(x)*grad\n";
        cout << "  - bwd_relu only puts {fwd_node} in inputs, misses grad -> segfault on eval\n";
    }

    return g_passed == g_run ? 0 : 1;
}
    // export_dot_full(c.get(), "file.dot");
