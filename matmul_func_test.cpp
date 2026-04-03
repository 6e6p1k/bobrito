#include "node.h"
#include "graph_viz.h"
#include "backward.h"
#include "allocator.cuh"
#include "ops.cuh"
#include "utils.h"
#include "visualize_graph.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

using namespace std;

// ── helpers ──

static int tests_passed = 0;
static int tests_failed = 0;

// Create a source node and fill with specific values (host -> device)
static shared_ptr<SourceNode> make_filled(int ndim, int* shape, vector<float> data, bool requires_grad = false)
{
    auto node = make_shared<SourceNode>();
    node->kind = NodeKind::Source;
    node->h_meta = make_shape_data(ndim, shape);
    node->h_requiers_grad = requires_grad;
    node->param_count = requires_grad ? 1 : 0;
    node->d_content = make_shared<ContentData>();
    node->d_content->h_values_to_d(data.data(), (int)data.size());
    return node;
}

// Pull result back to host for checking
static vector<float> to_host(shared_ptr<Node> node)
{
    int n = utils::numel(node->h_meta->ndim, node->h_meta->shape());
    float* h = node->d_content->d_values_to_h();
    vector<float> out(h, h + n);
    delete[] h;
    return out;
}

static bool approx_eq(float a, float b, float eps = 1e-4f)
{
    return fabs(a - b) < eps;
}

static bool check(const vector<float>& got, const vector<float>& expected, const char* name)
{
    if (got.size() != expected.size()) {
        cout << "  FAIL " << name << ": size mismatch (got " << got.size()
             << ", expected " << expected.size() << ")" << endl;
        tests_failed++;
        return false;
    }
    for (int i = 0; i < (int)got.size(); i++) {
        if (!approx_eq(got[i], expected[i])) {
            cout << "  FAIL " << name << ": element [" << i << "] got "
                 << got[i] << ", expected " << expected[i] << endl;
            tests_failed++;
            return false;
        }
    }
    cout << "  PASS " << name << endl;
    tests_passed++;
    return true;
}

// ── reference CPU matmul for generating expected values ──

// C[m][n] = sum_k A[m][k] * B[k][n]
// a_shape: [M, K], b_shape: [K, N]
// if a_trans: physical A is [K, M], logical A is [M, K]
// if b_trans: physical B is [N, K], logical B is [K, N]
static vector<float> cpu_matmul(
    const vector<float>& a, int a_rows, int a_cols, bool a_trans,
    const vector<float>& b, int b_rows, int b_cols, bool b_trans)
{
    int M = a_trans ? a_cols : a_rows;
    int K = a_trans ? a_rows : a_cols;
    int K2 = b_trans ? b_cols : b_rows;
    int N = b_trans ? b_rows : b_cols;
    (void)K2; // should equal K

    vector<float> c(M * N, 0.0f);
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                float av = a_trans ? a[k * a_cols + m] : a[m * a_cols + k];
                float bv = b_trans ? b[n * b_cols + k] : b[k * b_cols + n];
                sum += av * bv;
            }
            c[m * N + n] = sum;
        }
    }
    return c;
}

// batched version: a is [batch, a_rows, a_cols], b is [b_rows, b_cols] (broadcast)
// or both batched
static vector<float> cpu_batched_matmul(
    const vector<float>& a, int batch, int a_rows, int a_cols, bool a_trans,
    const vector<float>& b, int b_batch, int b_rows, int b_cols, bool b_trans)
{
    int M = a_trans ? a_cols : a_rows;
    int K = a_trans ? a_rows : a_cols;
    int N = b_trans ? b_rows : b_cols;

    vector<float> c(batch * M * N, 0.0f);
    int a_stride = a_rows * a_cols;
    int b_stride = b_rows * b_cols;

    for (int bi = 0; bi < batch; bi++) {
        int a_off = bi * a_stride;
        int b_off = (b_batch > 1) ? bi * b_stride : 0; // broadcast if b_batch == 1

        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    float av = a_trans
                        ? a[a_off + k * a_cols + m]
                        : a[a_off + m * a_cols + k];
                    float bv = b_trans
                        ? b[b_off + n * b_cols + k]
                        : b[b_off + k * b_cols + n];
                    sum += av * bv;
                }
                c[bi * M * N + m * N + n] = sum;
            }
        }
    }
    return c;
}

// ── tests ──

void test_basic_2d()
{
    // A [2,3] @ B [3,2] = C [2,2]
    // A = [[1,2,3],[4,5,6]]
    // B = [[1,2],[3,4],[5,6]]
    vector<float> a_data = {1,2,3, 4,5,6};
    vector<float> b_data = {1,2, 3,4, 5,6};

    int sa[] = {2, 3};
    int sb[] = {3, 2};
    auto A = make_filled(2, sa, a_data);  A->label = "A";
    auto B = make_filled(2, sb, b_data);  B->label = "B";

    auto C = matmul(A, B);
    eval(C);

    auto expected = cpu_matmul(a_data, 2, 3, false, b_data, 3, 2, false);
    // expected: [22, 28, 49, 64]
    check(to_host(C), expected, "basic_2d");
}

void test_transpose_B()
{
    // A [2,3] @ B^T where B stored as [2,3] -> logical B is [3,2]
    // C = A @ B^T -> [2,2]
    vector<float> a_data = {1,2,3, 4,5,6};
    vector<float> b_data = {1,2,3, 4,5,6}; // stored [2,3], transposed -> logical [3,2]

    int sa[] = {2, 3};
    int sb[] = {2, 3}; // physical shape
    auto A = make_filled(2, sa, a_data);  A->label = "A";
    auto B = make_filled(2, sb, b_data);  B->label = "B";
    B->transposed = true;

    auto C = matmul(A, B);
    eval(C);

    auto expected = cpu_matmul(a_data, 2, 3, false, b_data, 2, 3, true);
    // A @ B^T: row0=[1,2,3] dot [1,2,3]=14, dot [4,5,6]=32
    //          row1=[4,5,6] dot [1,2,3]=32, dot [4,5,6]=77
    check(to_host(C), expected, "transpose_B");
}

void test_transpose_A()
{
    // A stored as [3,2], transposed -> logical [2,3]
    // B [3,2]
    // C = A^T @ B -> [2,2]
    vector<float> a_data = {1,4, 2,5, 3,6}; // stored [3,2], transposed -> logical [2,3]
    vector<float> b_data = {1,2, 3,4, 5,6};

    int sa[] = {3, 2}; // physical
    int sb[] = {3, 2};
    auto A = make_filled(2, sa, a_data);  A->label = "A";
    auto B = make_filled(2, sb, b_data);  B->label = "B";
    A->transposed = true;

    auto C = matmul(A, B);
    eval(C);

    auto expected = cpu_matmul(a_data, 3, 2, true, b_data, 3, 2, false);
    check(to_host(C), expected, "transpose_A");
}

void test_transpose_both()
{
    // A stored as [3,2], transposed -> logical [2,3]
    // B stored as [2,3], transposed -> logical [3,2]
    // C = A^T @ B^T -> [2,2]
    vector<float> a_data = {1,4, 2,5, 3,6}; // physical [3,2]
    vector<float> b_data = {1,3,5, 2,4,6};  // physical [2,3]

    int sa[] = {3, 2};
    int sb[] = {2, 3};
    auto A = make_filled(2, sa, a_data);  A->label = "A";
    auto B = make_filled(2, sb, b_data);  B->label = "B";
    A->transposed = true;
    B->transposed = true;

    auto C = matmul(A, B);
    eval(C);

    auto expected = cpu_matmul(a_data, 3, 2, true, b_data, 2, 3, true);
    check(to_host(C), expected, "transpose_both");
}

void test_batched_no_transpose()
{
    // A [2, 2, 3] @ B [2, 3, 2] -> C [2, 2, 2]
    vector<float> a_data = {
        1,2,3, 4,5,6,       // batch 0
        7,8,9, 10,11,12     // batch 1
    };
    vector<float> b_data = {
        1,2, 3,4, 5,6,      // batch 0
        7,8, 9,10, 11,12    // batch 1
    };

    int sa[] = {2, 2, 3};
    int sb[] = {2, 3, 2};
    auto A = make_filled(3, sa, a_data);  A->label = "A";
    auto B = make_filled(3, sb, b_data);  B->label = "B";

    auto C = matmul(A, B);
    eval(C);

    auto expected = cpu_batched_matmul(a_data, 2, 2, 3, false, b_data, 2, 3, 2, false);
    check(to_host(C), expected, "batched_no_transpose");
}

void test_batched_broadcast_B()
{
    // A [2, 2, 3] @ B [3, 2] (broadcast) -> C [2, 2, 2]
    vector<float> a_data = {
        1,2,3, 4,5,6,       // batch 0
        7,8,9, 10,11,12     // batch 1
    };
    vector<float> b_data = {1,2, 3,4, 5,6}; // single, broadcast across batches

    int sa[] = {2, 2, 3};
    int sb[] = {3, 2};
    auto A = make_filled(3, sa, a_data);  A->label = "A";
    auto B = make_filled(2, sb, b_data);  B->label = "B";

    auto C = matmul(A, B);
    eval(C);

    auto expected = cpu_batched_matmul(a_data, 2, 2, 3, false, b_data, 1, 3, 2, false);
    check(to_host(C), expected, "batched_broadcast_B");
}

void test_batched_transpose_A()
{
    // A stored [2, 3, 2], transposed -> logical [2, 2, 3]
    // B [2, 3, 2]
    // C = A^T @ B -> [2, 2, 2]
    vector<float> a_data = {
        1,4, 2,5, 3,6,          // batch 0: physical [3,2], logical [2,3]
        7,10, 8,11, 9,12        // batch 1
    };
    vector<float> b_data = {
        1,2, 3,4, 5,6,
        7,8, 9,10, 11,12
    };

    int sa[] = {2, 3, 2}; // physical
    int sb[] = {2, 3, 2};
    auto A = make_filled(3, sa, a_data);  A->label = "A";
    auto B = make_filled(3, sb, b_data);  B->label = "B";
    A->transposed = true;

    auto C = matmul(A, B);
    eval(C);

    auto expected = cpu_batched_matmul(a_data, 2, 3, 2, true, b_data, 2, 3, 2, false);
    check(to_host(C), expected, "batched_transpose_A");
}

void test_batched_transpose_B_broadcast()
{
    // A [2, 2, 3]
    // B stored [2, 3], transposed -> logical [3, 2], broadcast across batch
    // C = A @ B^T -> [2, 2, 2]
    vector<float> a_data = {
        1,2,3, 4,5,6,
        7,8,9, 10,11,12
    };
    vector<float> b_data = {1,2,3, 4,5,6}; // physical [2,3], logical [3,2]

    int sa[] = {2, 2, 3};
    int sb[] = {2, 3}; // physical
    auto A = make_filled(3, sa, a_data);  A->label = "A";
    auto B = make_filled(2, sb, b_data);  B->label = "B";
    B->transposed = true;

    auto C = matmul(A, B);
    eval(C);

    auto expected = cpu_batched_matmul(a_data, 2, 2, 3, false, b_data, 1, 2, 3, true);
    check(to_host(C), expected, "batched_transpose_B_broadcast");
}

void test_square_identity()
{
    // A [3,3] @ I [3,3] = A
    vector<float> a_data = {1,2,3, 4,5,6, 7,8,9};
    vector<float> i_data = {1,0,0, 0,1,0, 0,0,1};

    int s[] = {3, 3};
    auto A = make_filled(2, s, a_data);  A->label = "A";
    auto I = make_filled(2, s, i_data);  I->label = "I";

    auto C = matmul(A, I);
    eval(C);

    check(to_host(C), a_data, "square_identity");
}

void test_nonsquare_tall_wide()
{
    // A [4,2] @ B [2,5] -> C [4,5]
    vector<float> a_data = {1,2, 3,4, 5,6, 7,8};
    vector<float> b_data = {1,2,3,4,5, 6,7,8,9,10};

    int sa[] = {4, 2};
    int sb[] = {2, 5};
    auto A = make_filled(2, sa, a_data);  A->label = "A";
    auto B = make_filled(2, sb, b_data);  B->label = "B";

    auto C = matmul(A, B);
    eval(C);
    
    auto expected = cpu_matmul(a_data, 4, 2, false, b_data, 2, 5, false);
    check(to_host(C), expected, "nonsquare_tall_wide");
}

// int main()
// {
//     cout << "\n=== Matmul Tests ===" << endl;

//     test_basic_2d();
//     test_transpose_B();
//     test_transpose_A();
//     test_transpose_both();
//     test_batched_no_transpose();
//     test_batched_broadcast_B();
//     test_batched_transpose_A();
//     test_batched_transpose_B_broadcast();
//     test_square_identity();
//     test_nonsquare_tall_wide();

//     cout << "\nResults: " << tests_passed << " passed, " << tests_failed << " failed" << endl;
//     return tests_failed > 0 ? 1 : 0;
// }