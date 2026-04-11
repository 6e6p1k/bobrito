#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#include <chrono>

#include "../src/allocator.cuh"
#include "../src/evaluate.h"
#include "../src/nodes.h"
#include "../src/ops.h"

#define TIME(label, iters, code) \
    do { \
        cudaDeviceSynchronize(); \
        auto _t0 = std::chrono::high_resolution_clock::now(); \
        for (int _i = 0; _i < iters; _i++) { code; } \
        cudaDeviceSynchronize(); \
        auto _t1 = std::chrono::high_resolution_clock::now(); \
        auto _us = std::chrono::duration_cast<std::chrono::microseconds>(_t1 - _t0).count(); \
        std::cout << label << ": " << _us << " us total, " << _us / iters << " us/iter (" << iters << " iters)\n"; \
    } while(0)

using namespace std;

// ============================================================
//                    DATA COMPARSION
// ============================================================
void compare_data(std::shared_ptr<Node> a, std::shared_ptr<Node> b, float tol = 1e-5f) {
    int size_a = a->get_size();
    int size_b = b->get_size();
    if (size_a != size_b) {
        std::cout << "SIZE MISMATCH: " << size_a << " vs " << size_b << "\n";
        return;
    }

    float* h_a = a->d_data->d_to_h();
    float* h_b = b->d_data->d_to_h();

    int mismatches = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < size_a; i++) {
        float diff = std::abs(h_a[i] - h_b[i]);
        if (diff > tol) {
            if (mismatches < 5)
                std::cout << "  [" << i << "] " << h_a[i] << " vs " << h_b[i] << " (diff=" << diff << ")\n";
            mismatches++;
        }
        max_diff = std::max(max_diff, diff);
    }

    if (mismatches == 0)
        std::cout << "MATCH (max_diff=" << max_diff << ")\n";
    else
        std::cout << mismatches << "/" << size_a << " mismatches (max_diff=" << max_diff << ")\n";

    free(h_a);
    free(h_b);
}

// ============================================================
//                        MEM STATS
// ============================================================
struct MemSnapshot {
    size_t cuda_free;
    size_t cuda_total;
    string pool_stats;
};

static MemSnapshot snapshot() {
    cudaDeviceSynchronize();
    MemSnapshot s;
    cudaMemGetInfo(&s.cuda_free, &s.cuda_total);
    s.pool_stats = MemoryPool::call().stats();
    return s;
}

static void print_snapshot(const string& label, const MemSnapshot& s) {
    size_t used_mb  = (s.cuda_total - s.cuda_free) / (1024 * 1024);
    size_t free_mb  = s.cuda_free  / (1024 * 1024);
    size_t total_mb = s.cuda_total / (1024 * 1024);
    cout << "[" << label << "]\n";
    cout << "  CUDA: " << used_mb << " MB used / " << total_mb << " MB total"
         << " (" << free_mb << " MB free)\n";
    cout << "  Pool: " << s.pool_stats << "\n\n";
}

// ============================================================
//                TRAINING LOOP (APPROACH 1)
// ============================================================
vector<shared_ptr<Node>> train_1(int N_ITERS) {
    // simple 1-layer: loss = sum_ld(relu(x @ W + b))
    auto x = full({64, 784}, 0.5f);
    auto W = full({784, 10}, 0.1f);
    auto b = full({1, 10}, 0.0f);
    x->set_label("x"); x->requires_grad = false;
    W->set_label("W"); W->requires_grad = true;
    b->set_label("b"); b->requires_grad = true;

    // print_snapshot("INIT", snapshot());

    for (int i = 0; i < N_ITERS; i++) {
        // ── forward ──
        auto z    = add(matmul(x, W), b);
        auto act  = relu(z);
        auto loss = sum_ld(act);
        auto g = Graph(loss, true);
        g.backward();
        g.eval();
        g.sgd_step(1e-3);
        // g.zero_grads();

    }
    // print_snapshot("AFTER LOOP", snapshot());
    return {W, b};
}

// ============================================================
//                  TRAINING LOOP (APPROACH 2)
// ============================================================
vector<shared_ptr<Node>> train_2(int N_ITERS) {
    // simple 1-layer: loss = sum_ld(relu(x @ W + b))
    auto x = full({64, 784}, 0.5f);
    auto W = full({784, 10}, 0.1f);
    auto b = full({1, 10}, 0.0f);
    x->set_label("x"); x->requires_grad = false;
    W->set_label("W"); W->requires_grad = true;
    b->set_label("b"); b->requires_grad = true;

    // print_snapshot("INIT", snapshot());

    // first time to build 
    auto z    = add(matmul(x, W), b);
    auto act  = relu(z);
    auto loss = sum_ld(act);
    auto g = make_model(loss);
    g->backward();

    for (int i = 0; i < N_ITERS; i++) {
        // g.zero_grads();
        g->eval();
        g->sgd_step(1e-3);
    }
    z.reset();
    act.reset();
    loss.reset();
    g.reset();

    // print_snapshot("AFTER LOOP", snapshot());
    return {W, b};
}

int main() {
    auto vec1 = train_1(50);
    auto vec2 = train_2(50);
    
    for (int i = 0; i < vec1.size(); ++i) {
        if (vec1[i]->get_size() < 50) {
            print_node(vec1[i]);
            print_node(vec2[i]);
        }
        compare_data(vec1[i], vec2[i]);
        compare_data(vec1[i]->owned_grad, vec2[i]->owned_grad);
    }

    print_snapshot("MIDDLE", snapshot());

    TIME("DISPOSABLE", 500, train_1(15));
    TIME("NON DISPOSABLE", 500, train_2(15));
    print_snapshot("END", snapshot());
    return 0;
}

int _main() {
    int N_ITERS = 15;
    // simple 1-layer: loss = sum_ld(relu(x @ W + b))
    // auto x = full({64, 784}, 0.5f);
    // auto W = full({784, 10}, 0.1f);
    // auto b = full({1, 10}, 0.0f);
    // x->set_label("x"); x->requires_grad = false;
    // W->set_label("W"); W->requires_grad = true;
    // b->set_label("b"); b->requires_grad = true;

    // print_snapshot("INIT", snapshot());

    // // first time to build 
    // auto z    = add(matmul(x, W), b);
    // auto act  = relu(z);
    // auto loss = sum_ld(act);
    // auto g = model(loss);
    // g->backward();

    // for (int i = 0; i < N_ITERS; i++) {
    //     // g.zero_grads();
    //     g->eval();
    //     g->sgd_step(1e-3);
    // }
    // g->detach_grads_from_bwd_graph();
    // z.reset();
    // act.reset();
    // loss.reset();
    // g.reset();
    // print_snapshot("AFTER LOOP", snapshot());


    // simple 1-layer: loss = sum_ld(relu(x @ W + b))
    // auto x = full({64, 784}, 0.5f);
    // auto W = full({784, 10}, 0.1f);
    // auto b = full({1, 10}, 0.0f);
    // x->set_label("x"); x->requires_grad = false;
    // W->set_label("W"); W->requires_grad = true;
    // b->set_label("b"); b->requires_grad = true;

    // print_snapshot("INIT", snapshot());

    // for (int i = 0; i < N_ITERS; i++) {
    //     // ── forward ──
    //     auto z    = add(matmul(x, W), b);
    //     auto act  = relu(z);
    //     auto loss = sum_ld(act);
    //     auto g = Graph(loss);
    //     // g.zero_grads();
    //     g.backward();
    //     g.eval();
    //     g.sgd_step(1e-3);
    // }
    // print_snapshot("AFTER LOOP", snapshot());

    return 0;
}