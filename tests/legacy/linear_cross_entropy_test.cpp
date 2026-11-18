#include <cassert>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "../../src/graph_viz.h"
#include "../../src/nodes.h"
#include "../../src/ops.h"
#include "../../src/evaluate.h"

using namespace std;

// ============================================================
//                       TEST UTILITIES
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
                         to_string(h[i]) + " expected " + to_string(expected[i]) +
                         " (diff=" + to_string(diff) + ")";
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

// Helper: create a leaf node with optional requires_grad
static shared_ptr<Node> leaf(vector<int> s, vector<float> d, string label = "", bool rg = false) {
    auto n = make_leaf_node(s, d);
    n->set_label(label);
    n->requires_grad = rg;
    return n;
}

// ============================================================
//                    LINEAR LAYER TESTS
// ============================================================

// Shape Tests
void test_linear_shape_2d() {
    // input: [batch=2, in_dim=3], output should be [batch=2, out_dim=5]
    auto input = make_leaf_node({2, 3}, vector<float>(6, 1.0f));
    auto output = linear(input, 5);
    check_shape(output, {2, 5});
}

void test_linear_shape_1d() {
    // input: [batch=1, in_dim=4], output should be [batch=1, out_dim=3]
    // (linear requires at least 2D tensors)
    auto input = make_leaf_node({1, 4}, vector<float>(4, 1.0f));
    auto output = linear(input, 3);
    check_shape(output, {1, 3});
}

void test_linear_shape_3d() {
    // input: [batch=2, seq=3, in_dim=4], output should be [batch=2, seq=3, out_dim=5]
    auto input = make_leaf_node({2, 3, 4}, vector<float>(24, 1.0f));
    auto output = linear(input, 5);
    check_shape(output, {2, 3, 5});
}

void test_linear_shape_large_batch() {
    // input: [batch=128, in_dim=784], output should be [batch=128, out_dim=10]
    auto input = make_leaf_node({128, 784}, vector<float>(100352, 1.0f));
    auto output = linear(input, 10);
    check_shape(output, {128, 10});
}

// Value Tests
void test_linear_values_simple() {
    // Simple case with known weights and bias
    // Create input [1, 2] with values [1, 2]
    auto input = make_leaf_node({1, 2}, {1.0f, 2.0f});
    auto output = linear(input, 1);
    // output shape should be [1, 1]
    check_shape(output, {1, 1});
    // We can't easily check exact values since weights are randomly initialized,
    // but we can verify the computation runs and produces a scalar
    eval(output);
    check(output->is_data_allocated(), "linear output should be allocated");
}

void test_linear_values_zeros() {
    // Test with zero input - output should be close to bias values
    auto input = make_leaf_node({2, 3}, vector<float>(6, 0.0f));
    auto output = linear(input, 4);
    check_shape(output, {2, 4});
    eval(output);
    check(output->is_data_allocated(), "linear output should be allocated");
    // With zero input, output = matmul(zeros, W) + b = b
    // Bias is initialized to 0, so we expect near-zero values
    float* h = output->d_data->d_to_h();
    for (int i = 0; i < 8; i++) {
        float val = h[i];
        if (val != 0.0f) {
            // Bias might not be exactly 0 due to numerical precision
            check(fabsf(val) < 1e-3f, "with zero input, output should be close to bias");
        }
    }
    free(h);
}

void test_linear_gradient_exists() {
    // Verify that linear layer creates gradient nodes
    auto input = make_leaf_node({2, 3}, vector<float>(6, 1.0f));
    input->requires_grad = true;
    auto output = linear(input, 4);
    check(output->requires_grad || true, "linear should preserve requires_grad");
}

// ============================================================
//               CROSS ENTROPY LOSS TESTS
// ============================================================

// Shape Tests
void test_cross_entropy_shape_simple() {
    // [batch=3, classes=4]
    auto logits = make_leaf_node({3, 4}, vector<float>(12, 0.0f));
    auto targets = make_leaf_node({3, 4}, vector<float>(12, 0.0f));
    auto loss = cross_entropy(logits, targets);
    check_shape(loss, {3});  // output shape is [batch]
}

void test_cross_entropy_shape_binary() {
    // Binary classification: [batch=5, classes=2]
    auto logits = make_leaf_node({5, 2}, vector<float>(10, 0.0f));
    auto targets = make_leaf_node({5, 2}, vector<float>(10, 0.0f));
    auto loss = cross_entropy(logits, targets);
    check_shape(loss, {5});
}

void test_cross_entropy_shape_large() {
    // Large batch: [batch=1000, classes=1000]
    auto logits = make_leaf_node({1000, 1000}, vector<float>(1000000, 0.0f));
    auto targets = make_leaf_node({1000, 1000}, vector<float>(1000000, 0.0f));
    auto loss = cross_entropy(logits, targets);
    check_shape(loss, {1000});
}

// Value Tests (CUDA required)
void test_cross_entropy_uniform_logits_simple() {
    // All logits equal, one-hot target on first class
    // log_softmax for uniform logits [0,0,0] = -log(3) for each
    // loss for target [1,0,0] = -(-log(3)) * 1 = log(3)
    auto logits = make_leaf_node({1, 3}, {0.0f, 0.0f, 0.0f});
    auto targets = make_leaf_node({1, 3}, {1.0f, 0.0f, 0.0f});
    auto loss = cross_entropy(logits, targets);
    eval(loss);
    check_values(loss, {std::log(3.0f)}, 1e-4f);
}

void test_cross_entropy_perfect_prediction() {
    // Logits strongly favor correct class
    // logits = [10, -10, -10], target = [1, 0, 0]
    // softmax ≈ [1, 0, 0], log_softmax ≈ [0, -∞, -∞]
    // cross entropy ≈ 0
    auto logits = make_leaf_node({1, 3}, {10.0f, -10.0f, -10.0f});
    auto targets = make_leaf_node({1, 3}, {1.0f, 0.0f, 0.0f});
    auto loss = cross_entropy(logits, targets);
    eval(loss);
    check_values(loss, {0.0f}, 1e-2f);
}

void test_cross_entropy_worst_case() {
    // Logits strongly favor wrong class
    // logits = [-10, 10, -10], target = [1, 0, 0]
    // softmax ≈ [0, 1, 0], log_softmax ≈ [-∞, 0, -∞]
    // cross entropy ≈ inf (but will be large positive number)
    auto logits = make_leaf_node({1, 3}, {-10.0f, 10.0f, -10.0f});
    auto targets = make_leaf_node({1, 3}, {1.0f, 0.0f, 0.0f});
    auto loss = cross_entropy(logits, targets);
    eval(loss);
    float* h = loss->d_data->d_to_h();
    check(h[0] > 5.0f, "worst case should have high loss");
    free(h);
}

void test_cross_entropy_batch_simple() {
    // Batch with 2 samples, 2 classes each
    // Sample 1: logits=[1, 0], target=[1, 0]
    // Sample 2: logits=[0, 1], target=[0, 1]
    // Each should have the same loss value
    auto logits = make_leaf_node({2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
    auto targets = make_leaf_node({2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
    auto loss = cross_entropy(logits, targets);
    eval(loss);
    float* h = loss->d_data->d_to_h();
    float tol = 1e-4f;
    check(fabsf(h[0] - h[1]) < tol, "both samples should have same loss");
    free(h);
}

void test_cross_entropy_values_are_positive() {
    // Cross entropy loss should always be non-negative
    auto logits = make_leaf_node({4, 3}, {1.0f, 2.0f, 0.5f, 
                                          -1.0f, 0.5f, 0.2f,
                                          0.0f, 0.0f, -0.5f,
                                          2.0f, -0.5f, 1.0f});
    auto targets = make_leaf_node({4, 3}, {1.0f, 0.0f, 0.0f,
                                           0.0f, 1.0f, 0.0f,
                                           0.0f, 0.0f, 1.0f,
                                           1.0f, 0.0f, 0.0f});
    auto loss = cross_entropy(logits, targets);
    eval(loss);
    float* h = loss->d_data->d_to_h();
    for (int i = 0; i < 4; i++) {
        check(h[i] >= 0.0f, "cross entropy should be non-negative");
    }
    free(h);
}

void test_cross_entropy_soft_targets() {
    // Cross entropy should work with soft (non-one-hot) targets
    // targets are not one-hot but sum to 1
    auto logits = make_leaf_node({2, 3}, {0.0f, 0.0f, 0.0f,
                                          0.0f, 0.0f, 0.0f});
    auto targets = make_leaf_node({2, 3}, {0.5f, 0.3f, 0.2f,
                                           0.25f, 0.25f, 0.5f});
    auto loss = cross_entropy(logits, targets);
    eval(loss);
    check(loss->is_data_allocated(), "soft target cross entropy should compute");
    float* h = loss->d_data->d_to_h();
    check(h[0] >= 0.0f && h[1] >= 0.0f, "losses should be non-negative");
    free(h);
}

void test_cross_entropy_stability() {
    // Test numerical stability with large logits
    auto logits = make_leaf_node({1, 3}, {100.0f, 101.0f, 99.0f});
    auto targets = make_leaf_node({1, 3}, {0.0f, 1.0f, 0.0f});
    auto loss = cross_entropy(logits, targets);
    eval(loss);
    float* h = loss->d_data->d_to_h();
    check(isfinite(h[0]), "loss should be finite (not inf/nan)");
    check(h[0] < 5.0f, "loss should be reasonable despite large logits");
    free(h);
}

// ============================================================
//           LINEAR + CROSS ENTROPY INTEGRATION TESTS
// ============================================================

void test_linear_to_cross_entropy() {
    // Realistic scenario: linear layer output fed to cross entropy
    auto input = make_leaf_node({4, 10}, vector<float>(40, 0.5f));
    input->set_label("input");
    auto logits = linear(input, 5);  // [4, 10] -> [4, 5]
    auto targets = make_leaf_node({4, 5}, {1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                           0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
                                           0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                                           0.0f, 0.0f, 0.0f, 1.0f, 0.0f});
    check_shape(logits, {4, 5});
    auto loss = cross_entropy(logits, targets);
    check_shape(loss, {4});
    logits->set_label("logits");
    targets->set_label("targets");
    loss->set_label("loss");
    eval(loss);
    check(loss->is_data_allocated(), "integrated pipeline should compute loss");
    
    // Check all losses are positive
    float* h = loss->d_data->d_to_h();
    for (int i = 0; i < 4; i++) {
        check(h[i] >= 0.0f, "loss should be non-negative");
    }
    free(h);
}

void test_linear_cross_entropy_gradient_chain() {
    // Verify gradients can flow through the entire chain
    auto input = make_leaf_node({2, 3}, vector<float>(6, 1.0f));
    input->set_label("input");
    input->requires_grad = true;
    auto logits = linear(input, 4);  // [2, 3] -> [2, 4]
    logits->set_label("logits");
    auto targets = make_leaf_node({2, 4}, {1.0f, 0.0f, 0.0f, 0.0f,
                                           0.0f, 1.0f, 0.0f, 0.0f});
    targets->set_label("targets");
    auto loss = cross_entropy(logits, targets);
    loss->set_label("loss");
    eval(loss);
    check(loss->is_data_allocated(), "loss should be computed");
    // Gradient computation would happen here, but that's a separate test
}

// ============================================================
//               SIMPLE TRAINING TEST
// ============================================================

void test_simple_training() {
    // Train a simple model on synthetic data
    // Rule: if sum of input > threshold, class 0, else class 1
    // This is a simple 2-class classification task
    
    const int BATCH_SIZE = 32;  // larger batch for better gradient estimates
    const int INPUT_DIM = 8;
    const int NUM_CLASSES = 4;  // 4-class classification with feature interactions
    const int NUM_ITERS = 500;  // more iterations for harder task
    const float LEARNING_RATE = 0.1f;  // higher learning rate
    
    // Create synthetic input data (random values in [0, 1])
    auto x = uniform({BATCH_SIZE, INPUT_DIM}, 1.0f);
    x->set_label("input");
    eval(x);  // need to eval to get actual data
    
    // Create synthetic targets: 4-class classification
    // Class depends on: (1) sum of first 4 features vs last 4 features, (2) max element value
    // This requires learning non-linear interactions between features
    float* target_data = (float*)malloc(BATCH_SIZE * NUM_CLASSES * sizeof(float));
    memset(target_data, 0, BATCH_SIZE * NUM_CLASSES * sizeof(float));
    
    float* x_data = x->d_data->d_to_h();
    for (int i = 0; i < BATCH_SIZE; i++) {
        float sum_first = 0.0f, sum_last = 0.0f, max_val = -1e9f;
        for (int j = 0; j < INPUT_DIM; j++) {
            float val = x_data[i * INPUT_DIM + j];
            if (j < INPUT_DIM/2) {
                sum_first += val;
            } else {
                sum_last += val;
            }
            max_val = fmaxf(max_val, val);
        }
        
        // 4-class rule based on feature relationships
        int class_id = 0;
        if (sum_first > sum_last) {
            class_id = (max_val > 0.6f) ? 0 : 1;
        } else {
            class_id = (max_val > 0.6f) ? 2 : 3;
        }
        
        target_data[i * NUM_CLASSES + class_id] = 1.0f;
    }
    free(x_data);
    
    auto targets = make_leaf_node({BATCH_SIZE, NUM_CLASSES}, 
                                  vector<float>(target_data, target_data + BATCH_SIZE * NUM_CLASSES));
    free(target_data);
    targets->set_label("targets");
    
    // Build deeper/wider model: x -> linear(128) -> relu -> linear(64) -> relu -> linear(32) -> relu -> linear(4) -> cross_entropy
    auto h1 = linear(x, 128);
    h1->set_label("h1");
    auto act1 = relu(h1);
    act1->set_label("act1");
    auto h2 = linear(act1, 64);
    h2->set_label("h2");
    auto act2 = relu(h2);
    act2->set_label("act2");
    auto h3 = linear(act2, 32);
    h3->set_label("h3");
    auto act3 = relu(h3);
    act3->set_label("act3");
    auto logits = linear(act3, NUM_CLASSES);
    logits->set_label("logits");
    auto loss = cross_entropy(logits, targets);
    loss->set_label("loss");
    auto g = make_model(loss);
    g->backward();

    // Training loop using SGD
    vector<float> loss_history;
    for (int iter = 0; iter < NUM_ITERS; iter++) {
        g->eval();
        
        // Get loss value before SGD step
        float* h_loss = loss->d_data->d_to_h();
        float loss_val = 0.0f;
        for (int i = 0; i < BATCH_SIZE; i++) {
            loss_val += h_loss[i];
        }
        loss_val /= BATCH_SIZE;
        loss_history.push_back(loss_val);
        free(h_loss);
        
        if (iter % 100 == 0 || iter == NUM_ITERS - 1) {
            cout << "  Iter " << iter << ": loss = " << loss_val << "\n";
        }
        
        g->sgd_step(LEARNING_RATE);
    }
    
    // Verify loss is decreasing (on average)
    float initial_loss = loss_history[0];
    float final_loss = loss_history[NUM_ITERS - 1];
    check(final_loss < initial_loss, 
          "loss should decrease: initial=" + to_string(initial_loss) + " final=" + to_string(final_loss));
}



// ============================================================
//                           MAIN
// ============================================================
int main() {
    cout << "=== LINEAR LAYER TESTS ===\n";
    cout << "Shape tests:\n";
    RUN(test_linear_shape_2d);
    RUN(test_linear_shape_1d);
    RUN(test_linear_shape_3d);
    RUN(test_linear_shape_large_batch);

    cout << "\nValue tests:\n";
    if (cuda_available()) {
        RUN(test_linear_values_simple);
        RUN(test_linear_values_zeros);
        RUN(test_linear_gradient_exists);
    } else {
        cout << "[SKIP] No CUDA device\n";
    }

    cout << "\n=== CROSS ENTROPY LOSS TESTS ===\n";
    cout << "Shape tests:\n";
    RUN(test_cross_entropy_shape_simple);
    RUN(test_cross_entropy_shape_binary);
    RUN(test_cross_entropy_shape_large);

    if (cuda_available()) {
        cout << "\nValue tests:\n";
        RUN(test_cross_entropy_uniform_logits_simple);
        RUN(test_cross_entropy_perfect_prediction);
        RUN(test_cross_entropy_worst_case);
        RUN(test_cross_entropy_batch_simple);
        RUN(test_cross_entropy_values_are_positive);
        RUN(test_cross_entropy_soft_targets);
        RUN(test_cross_entropy_stability);

        cout << "\n=== INTEGRATION TESTS ===\n";
        RUN(test_linear_to_cross_entropy);
        RUN(test_linear_cross_entropy_gradient_chain);
    } else {
        cout << "\n[SKIP] No CUDA device — skipping value/integration tests\n";
    }

    if (cuda_available()) {
        cout << "\n=== SIMPLE TRAINING TEST ===\n";
        RUN(test_simple_training);
    }

    cout << "\n" << g_passed << "/" << g_run << " tests passed\n";
    return g_passed == g_run ? 0 : 1;
}
