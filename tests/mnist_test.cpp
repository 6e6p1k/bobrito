// MNIST training test
// Loads MNIST from .npy files (float32 images + int32 labels) at:
//   MNIST_DIR/mnist_train_X.npy  [60000, 784]
//   MNIST_DIR/mnist_train_y.npy  [60000]  int32
//   MNIST_DIR/mnist_test_X.npy   [10000, 784]
//   MNIST_DIR/mnist_test_y.npy   [10000]  int32
//
// Default dir: /tmp  (override with env var MNIST_DIR)
// Generate files with:
//   python3 -c "
//     from keras.datasets import mnist; import numpy as np
//     (tx,ty),(vx,vy) = mnist.load_data()
//     tx = tx.reshape(-1,784).astype(np.float32)/255.0
//     vx = vx.reshape(-1,784).astype(np.float32)/255.0
//     np.save('/tmp/mnist_train_X.npy', tx)
//     np.save('/tmp/mnist_train_y.npy', ty.astype(np.int32))
//     np.save('/tmp/mnist_test_X.npy',  vx)
//     np.save('/tmp/mnist_test_y.npy',  vy.astype(np.int32))
//   "

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <execinfo.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <csignal>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

static void segfault_handler(int sig) {
    void* array[30];
    int size = backtrace(array, 30);
    fprintf(stderr, "Signal %d (SIGSEGV/SIGABRT):\n", sig);
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    _exit(1);
}

#include "../src/evaluate.h"
#include "../src/nodes.h"
#include "../src/ops.h"

using namespace std;

// ============================================================
//                     TINY NPY LOADER
// ============================================================
// Supports only C-contiguous float32 ('<f4') and int32 ('<i4') arrays.

struct NpyArray {
    vector<int> shape;
    vector<float> data_f32;
    vector<int32_t> data_i32;
    bool is_float;
};

static string read_npy_header(ifstream& f, vector<int>& shape) {
    // magic
    char magic[7] = {};
    f.read(magic, 6);
    if (string(magic, 6) != "\x93NUMPY")
        throw runtime_error("not a .npy file");
    uint8_t major, minor;
    f.read((char*)&major, 1);
    f.read((char*)&minor, 1);

    uint32_t hlen;
    if (major == 1) {
        uint16_t h16;
        f.read((char*)&h16, 2);
        hlen = h16;
    } else {
        f.read((char*)&hlen, 4);
    }

    string header(hlen, '\0');
    f.read(header.data(), hlen);

    // find dtype
    string descr;
    auto dp = header.find("'descr'");
    if (dp == string::npos) dp = header.find("\"descr\"");
    if (dp != string::npos) {
        auto s = header.find('\'', dp + 7);
        if (s == string::npos) s = header.find('"', dp + 7);
        auto e = header.find('\'', s + 1);
        if (e == string::npos) e = header.find('"', s + 1);
        descr = header.substr(s + 1, e - s - 1);
    }

    // find shape
    auto sp = header.find("'shape'");
    if (sp == string::npos) sp = header.find("\"shape\"");
    if (sp != string::npos) {
        auto lp = header.find('(', sp);
        auto rp = header.find(')', lp);
        string stuple = header.substr(lp + 1, rp - lp - 1);
        shape.clear();
        size_t pos = 0;
        while (pos < stuple.size()) {
            while (pos < stuple.size() &&
                   (stuple[pos] == ' ' || stuple[pos] == ','))
                pos++;
            if (pos >= stuple.size()) break;
            size_t end = pos;
            while (end < stuple.size() && stuple[end] != ',') end++;
            string tok = stuple.substr(pos, end - pos);
            if (!tok.empty()) shape.push_back(stoi(tok));
            pos = end;
        }
    }
    return descr;
}

static NpyArray load_npy(const string& path) {
    ifstream f(path, ios::binary);
    if (!f) throw runtime_error("cannot open: " + path);

    NpyArray arr;
    string descr = read_npy_header(f, arr.shape);

    int n = 1;
    for (int d : arr.shape) n *= d;

    if (descr == "<f4" || descr == "=f4") {
        arr.is_float = true;
        arr.data_f32.resize(n);
        f.read((char*)arr.data_f32.data(), n * sizeof(float));
    } else if (descr == "<i4" || descr == "=i4") {
        arr.is_float = false;
        arr.data_i32.resize(n);
        f.read((char*)arr.data_i32.data(), n * sizeof(int32_t));
    } else {
        throw runtime_error("unsupported dtype: " + descr);
    }
    return arr;
}

// ============================================================
//                   ONE-HOT ENCODER
// ============================================================
static vector<float> one_hot(const vector<int32_t>& labels, int n_classes,
                             int start, int count) {
    vector<float> oh(count * n_classes, 0.0f);
    for (int i = 0; i < count; i++)
        oh[i * n_classes + labels[start + i]] = 1.0f;
    return oh;
}

// ============================================================
//                       HELPERS
// ============================================================
static bool cuda_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

// Copy a slice of host data into a leaf node.
static shared_ptr<Node> slice_to_node(const vector<float>& data, int row_start,
                                      int rows, int cols) {
    vector<float> slice(data.begin() + row_start * cols,
                        data.begin() + (row_start + rows) * cols);
    return make_leaf_node({rows, cols}, slice);
}

// Read argmax from a device node (downloads to host).
static vector<int> argmax_rows(shared_ptr<Node> node) {
    // node shape: [batch, classes]
    int rows = node->get_shape()[0];
    int cols = node->get_shape()[1];
    float* h = node->d_data->d_to_h();
    vector<int> preds(rows);
    for (int r = 0; r < rows; r++) {
        int best = 0;
        for (int c = 1; c < cols; c++)
            if (h[r * cols + c] > h[r * cols + best]) best = c;
        preds[r] = best;
    }
    free(h);
    return preds;
}

// ============================================================
//                    SOFTMAX CROSS-ENTROPY
// ============================================================
// Numerically stable:  softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
// NLL loss            = -sum(y_onehot * log(softmax)) / (batch * classes)
static shared_ptr<Node> softmax_cross_entropy(shared_ptr<Node> logits,
                                              shared_ptr<Node> y_oh,
                                              int batch_size, int n_classes) {
    auto mx    = max(logits, -1, true);          // [B, 1]
    auto shift = sub(logits, mx);                // [B, C] — broadcast sub
    auto ex    = exp(shift);                     // [B, C]
    auto s     = sum(ex, -1, true);              // [B, 1]
    auto sm    = mul(ex, pow(s, -1.0f));         // [B, C]  softmax

    // NLL: -sum_over_classes(y * log(sm))  then sum_over_batch, scaled
    // Use sum(..., dim, keepdims=true) to stay 2D throughout and avoid 0-d tensors.
    auto elem_nll = neg(mul(y_oh, log(sm)));     // [B, C]
    auto sum_cls  = sum(elem_nll, -1, true);     // [B, 1]  sum over classes
    auto sum_all  = sum(sum_cls, 0, true);       // [1, 1]  sum over batch

    float scale_val = 1.0f / (float)(batch_size * n_classes);
    auto scale = full({1, 1}, scale_val);
    scale->set_label("scale");

    auto loss = mul(sum_all, scale);             // [1, 1]
    loss->set_label("loss");
    return loss;
}

// ============================================================
//                         TRAIN
// ============================================================
struct MLP {
    shared_ptr<Node> W1, b1, W2, b2;
    int in_dim, hidden, out_dim;
};

static MLP init_mlp(int in_dim, int hidden, int out_dim) {
    MLP m;
    m.in_dim = in_dim; m.hidden = hidden; m.out_dim = out_dim;
    m.W1 = kaiming_normal({in_dim, hidden}); m.W1->set_label("W1"); m.W1->requires_grad = true;
    m.b1 = full({1, hidden}, 0.0f);          m.b1->set_label("b1"); m.b1->requires_grad = true;
    m.W2 = kaiming_normal({hidden, out_dim});m.W2->set_label("W2"); m.W2->requires_grad = true;
    m.b2 = full({1, out_dim}, 0.0f);         m.b2->set_label("b2"); m.b2->requires_grad = true;
    return m;
}

static shared_ptr<Node> forward(const MLP& m, shared_ptr<Node> x) {
    auto h   = relu(add(matmul(x, m.W1), m.b1));  // [B, hidden]
    auto out = add(matmul(h, m.W2), m.b2);         // [B, out_dim]
    out->set_label("logits");
    return out;
}

// ============================================================
//                         MAIN TEST
// ============================================================
int main() {
    signal(SIGSEGV, segfault_handler);
    signal(SIGABRT, segfault_handler);

    if (!cuda_available()) {
        cout << "[SKIP] No CUDA device available — skipping MNIST test\n";
        return 0;
    }

    // Locate data files
    const char* env_dir = getenv("MNIST_DIR");
    string data_dir = env_dir ? env_dir : "/tmp";

    auto try_load = [&](const string& name) -> NpyArray {
        string path = data_dir + "/" + name;
        ifstream probe(path);
        if (!probe) {
            cerr << "[SKIP] Missing file: " << path << "\n";
            cerr << "       Set MNIST_DIR env var or regenerate with the "
                    "header comment.\n";
            exit(0);
        }
        return load_npy(path);
    };

    cout << "Loading MNIST from " << data_dir << "...\n";
    cout.flush();
    auto train_X_arr = try_load("mnist_train_X.npy");
    cout << "Loaded train_X: " << train_X_arr.shape[0] << "x" << train_X_arr.shape[1] << "\n";
    cout.flush();
    auto train_y_arr = try_load("mnist_train_y.npy");
    cout << "Loaded train_y: " << train_y_arr.shape[0] << "\n";
    cout.flush();
    auto test_X_arr  = try_load("mnist_test_X.npy");
    auto test_y_arr  = try_load("mnist_test_y.npy");

    const int N_TRAIN  = train_X_arr.shape[0];   // 60000
    const int N_TEST   = test_X_arr.shape[0];    // 10000
    const int IN_DIM   = 784;
    const int HIDDEN   = 128;
    const int N_CLASS  = 10;
    const int BATCH    = 64;
    const int EPOCHS   = 3;
    const float LR     = 7e-4f;
    const int N_BATCHES = N_TRAIN / BATCH;

    cout << "  train: " << N_TRAIN << "  test: " << N_TEST << "\n";

    cout << "Initializing MLP...\n"; cout.flush();
    MLP mlp = init_mlp(IN_DIM, HIDDEN, N_CLASS);
    cout << "MLP initialized.\n"; cout.flush();

    // ── Training loop ──────────────────────────────────────
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float epoch_loss = 0.0f;

        for (int i = 0; i < N_BATCHES; i++) {
            int start = i * BATCH;

            if (epoch == 0 && i == 0) { cout << "Building first batch graph...\n"; cout.flush(); }

            // batch inputs
            auto x   = slice_to_node(train_X_arr.data_f32, start, BATCH, IN_DIM);
            x->set_label("x");

            auto y_data = one_hot(train_y_arr.data_i32, N_CLASS, start, BATCH);
            auto y   = make_leaf_node({BATCH, N_CLASS}, y_data);
            y->set_label("y");

            // forward + loss
            auto logits = forward(mlp, x);
            if (epoch == 0 && i == 0) { cout << "Forward done.\n"; cout.flush(); }
            auto loss   = softmax_cross_entropy(logits, y, BATCH, N_CLASS);
            if (epoch == 0 && i == 0) { cout << "Loss built. Shape ndim=" << loss->get_ndim() << "\n"; cout.flush(); }

            // backward + update
            if (epoch == 0 && i == 0) { cout << "Building Graph...\n"; cout.flush(); }
            auto g = Graph(loss, true);
            if (epoch == 0 && i == 0) { cout << "Graph built. fwd_order size=" << g.fwd_order.size() << "\n"; cout.flush(); }
            if (epoch == 0 && i == 0) { cout << "require_grad_nodes=" << g.require_grad_nodes.size() << "\n"; cout.flush(); }
            g.backward();
            if (epoch == 0 && i == 0) { cout << "Backward done. bwd_order size=" << g.bwd_order.size() << "\n"; cout.flush(); }
            g.eval();
            if (epoch == 0 && i == 0) { cout << "Eval done.\n"; cout.flush(); }
            g.sgd_step(LR);
            if (epoch == 0 && i == 0) { cout << "SGD step done.\n"; cout.flush(); }

            // read loss value
            float* h = loss->d_data->d_to_h();
            epoch_loss += h[0];
            free(h);
        }

        cout << "Epoch " << epoch + 1 << "/" << EPOCHS
             << " | avg loss: " << epoch_loss / N_BATCHES << "\n";
    }

    // ── Evaluation ─────────────────────────────────────────
    // Forward-only: temporarily disable requires_grad so eval() skips backward.
    mlp.W1->requires_grad = false;
    mlp.b1->requires_grad = false;
    mlp.W2->requires_grad = false;
    mlp.b2->requires_grad = false;

    int correct = 0;
    const int N_TEST_BATCHES = N_TEST / BATCH;

    for (int i = 0; i < N_TEST_BATCHES; i++) {
        int start = i * BATCH;

        auto x_test = slice_to_node(test_X_arr.data_f32, start, BATCH, IN_DIM);
        x_test->set_label("x_test");

        auto h   = relu(add(matmul(x_test, mlp.W1), mlp.b1));
        auto out = add(matmul(h, mlp.W2), mlp.b2);
        out->set_label("logits_test");

        eval(out);

        auto preds = argmax_rows(out);
        for (int j = 0; j < BATCH; j++)
            if (preds[j] == test_y_arr.data_i32[start + j]) correct++;
    }

    float acc = 100.0f * correct / (N_TEST_BATCHES * BATCH);
    cout << "\nTest accuracy: " << acc << "%\n";

    // After 3 epochs with lr=7e-4 expect at least 85% accuracy.
    const float MIN_ACC = 85.0f;
    if (acc < MIN_ACC) {
        cerr << "[FAIL] accuracy " << acc << "% < " << MIN_ACC << "%\n";
        return 1;
    }

    cout << "[PASS] accuracy " << acc << "% >= " << MIN_ACC << "%\n";
    return 0;
}
