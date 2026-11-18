// ALL OPS
#include <cublas_v2.h>
#include <cub/cub.cuh>
#include <curand.h>

#include <string.h>
#include <memory>
#include <stdexcept>
#include <string>

#include "allocator.cuh"
#include "kernels.cuh"
#include "nodes.h"
#include "ops.h"
#include "primitives.h"
#include "shape_data.h"
#include "stream.h"
#include "utils.h"
#include "primitives.h"

#define SCALAR(n) (static_pointer_cast<ScalarNode>(n))


using namespace std;
static int g_seed = 0;

void CUDA_LAST_ERROR() {
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
}

// ============================================================
//                         COPY HELPER
// ============================================================
LeafPrim prim_leaf;
IdentityPrim prim_identity;

void IdentityPrim::allocate(Node* node) {
    node->d_data = node->inputs[0]->d_data;
}

shared_ptr<Node> copy_node_no_prim(shared_ptr<Node> node) {
    auto out = make_shared<Node>();
    out->prim = &prim_identity;
    auto shape = new int[node->get_ndim()];
    for (int i = 0; i < node->get_ndim(); ++i)
        shape[i] = node->get_shape()[i];
    out->h_meta = make_shape_data(node->get_ndim(), shape);
    out->is_bwd = true;
    out->inputs.push_back(node);
    out->set_label(node->label + ".copy()");
    return out;
}

// ============================================================
//                         LEAF INITIATION
// ============================================================
shared_ptr<Node> make_leaf_node(int* shape, int ndim,
                                shared_ptr<ContentData> device_data) {
    shared_ptr<Node> out_node = make_node(&prim_leaf, {}, shape, ndim);
    if (!device_data || !(device_data->is_allocated()))
        throw logic_error("Empty data was received!");
    out_node->d_data = device_data;
    return out_node;
}

shared_ptr<Node> make_leaf_node(vector<int> shape,
                                shared_ptr<ContentData> device_data) {
    return make_leaf_node(shape.data(), shape.size(), device_data);
}

shared_ptr<Node> make_leaf_node(int* shape, int ndim, float* h_data) {
    int size = utils::numel(ndim, shape);
    shared_ptr<ContentData> deviceData = make_shared<ContentData>();
    deviceData->allocate(size);
    deviceData->h_to_d(h_data, size);

    return make_leaf_node(shape, ndim, deviceData);
}

shared_ptr<Node> make_leaf_node(vector<int> shape, vector<float> h_data) {
    return make_leaf_node(shape.data(), shape.size(), h_data.data());
}

shared_ptr<Node> _empty_leaf_node(vector<int> shape) {
    shared_ptr<Node> out_node =
        make_node(&prim_leaf, {}, shape.data(), shape.size());
    shared_ptr<ContentData> d_data = make_shared<ContentData>();
    out_node->d_data = d_data;
    return out_node;
}

// ============================================================
//                            UNARY
// ============================================================
template <typename KernelOp>
static void unary_eval(Node* node) {
    int n = node->get_size();
    kernels::unary<KernelOp><<<GRID_SIZE(n), BLOCK_SIZE, 0, bobr::current_stream()>>>(
        node->inputs[0]->get_data(), node->get_data(), n);
}

shared_ptr<Node> unary_helper(Primitive* prim, shared_ptr<Node> in_node,
                              string label) {
    shared_ptr<Node> out_node =
        make_node(prim, {in_node}, in_node->get_shape(), in_node->get_ndim());
    out_node->set_label(label);
    return out_node;
}

// ========================= RELU ==========================
ReluPrim prim_relu;

void ReluPrim::eval(Node* node) { unary_eval<cuReLU>(node); }

void ReluPrim::vjp(shared_ptr<Node> fwd, shared_ptr<Node> grad,
                   shared_ptr<Node> out[]) {
    auto A = copy_node_no_prim(fwd->inputs[0]);
    out[0] = bwd_relu(grad, A);
}

shared_ptr<Node> relu(shared_ptr<Node> a) {
    return unary_helper(&prim_relu, a, a->label + ".relu()");
}

// ======================= BWD RELU ========================
BwdReluPrim prim_bwd_relu;

void BwdReluPrim::eval(Node* node) {
    int n = node->get_size();
    kernels::bwd_relu<<<GRID_SIZE(n), BLOCK_SIZE, 0, bobr::current_stream()>>>(
        node->inputs[0]->get_data(), node->inputs[1]->get_data(),
        node->get_data(), n);
}

shared_ptr<Node> bwd_relu(shared_ptr<Node> grad, shared_ptr<Node> fwd_node) {
    auto out_node = make_node(&prim_bwd_relu, {fwd_node, grad},
                              fwd_node->get_shape(), fwd_node->get_ndim());
    return out_node;
}

// ========================== NEG ==========================
NegPrim prim_neg;

void NegPrim::eval(Node* node) { unary_eval<cuNeg>(node); }

void NegPrim::vjp(shared_ptr<Node> fwd, shared_ptr<Node> bwd,
                  shared_ptr<Node> out[]) {
    out[0] = neg(bwd);
}

shared_ptr<Node> neg(shared_ptr<Node> a) {
    return unary_helper(&prim_neg, a, "-" + a->label);
}

// ========================== EXP ==========================
ExpPrim prim_exp;

void ExpPrim::eval(Node* node) { unary_eval<cuExp>(node); }

void ExpPrim::vjp(shared_ptr<Node> fwd, shared_ptr<Node> bwd,
                  shared_ptr<Node> out[]) {
    auto e = copy_node_no_prim(fwd);
    out[0] = mul(e, bwd);
}

shared_ptr<Node> exp(shared_ptr<Node> a) {
    return unary_helper(&prim_exp, a, a->label + ".exp()");
}

// ========================= SQRT ==========================
SqrtPrim prim_sqrt;

void SqrtPrim::eval(Node* node) { unary_eval<cuSqrt>(node); }

shared_ptr<Node> sqrt(shared_ptr<Node> a) {
    return unary_helper(&prim_sqrt, a, a->label + ".sqrt()");
}

// ========================== LOG ==========================
LogPrim prim_log;

void LogPrim::eval(Node* node) { unary_eval<cuLog>(node); }

void LogPrim::vjp(shared_ptr<Node> fwd, shared_ptr<Node> bwd,
                  shared_ptr<Node> out[]) {
    auto A = copy_node_no_prim(fwd->inputs[0]);
    out[0] = div(bwd, A);
}

shared_ptr<Node> log(shared_ptr<Node> a) {
    return unary_helper(&prim_log, a, a->label + ".log()");
}

// ========================= FULL ==========================
FullPrim prim_full;

void FullPrim::eval(Node* node) {
    int n = node->get_size();
    float val = SCALAR(node->inputs[0])->value;
    kernels::fill<<<GRID_SIZE(n), BLOCK_SIZE, 0, bobr::current_stream()>>>(node->get_data(), val, n);
}

shared_ptr<Node> full(int* shape, int ndim, float fill_val) {
    shared_ptr<ScalarNode> scalar = make_scalar_node(fill_val, "fill_val");
    shared_ptr<Node> out_node = make_node(&prim_full, {scalar}, shape, ndim);
    out_node->set_label("full(" + to_string((int)fill_val) + ")");
    return out_node;
}

shared_ptr<Node> full(vector<int> shape, float fill_val) {
    auto node = _empty_leaf_node(shape);
    return full(shape.data(), shape.size(), fill_val);
}

// ======================== ARANGE =========================
ArangePrim prim_arange;

void ArangePrim::eval(Node* node) {
    int n = node->get_size();
    float start = SCALAR(node->inputs[0])->value;
    float step = SCALAR(node->inputs[1])->value;
    kernels::arange<<<GRID_SIZE(n), BLOCK_SIZE, 0, bobr::current_stream()>>>(node->get_data(), start,
                                                  step, n);
}

shared_ptr<Node> arange(int* shape, int ndim, float start, float step) {
    auto scalar_1 = make_scalar_node(start, "start");
    auto scalar_2 = make_scalar_node(step, "step");

    shared_ptr<Node> out_node =
        make_node(&prim_arange, {scalar_1, scalar_2}, shape, ndim);
    out_node->set_label("arange(start=" + to_string((int)start) +
                        ", step=" + to_string((int)step) + ")");
    return out_node;
}

shared_ptr<Node> arange(vector<int> shape, int start, int step) {
    return arange(shape.data(), shape.size(), start, step);
}

shared_ptr<Node> arange(int start, int end, int step) {
    vector<int> shape(1);
    shape = {(end - start) / step};
    auto out_node = arange(shape, start, step);
    out_node->set_label("arange(start=" + to_string(start) + ", end=" +
                        to_string(end) + ", step=" + to_string(step) + ")");
    return out_node;
}

// ======================== UNIFORM ========================
void _uniform(float* dst, float limit, int n) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetStream(gen, bobr::current_stream());
    curandSetPseudoRandomGeneratorSeed(gen, g_seed++);
    curandGenerateUniform(gen, dst, n);
    curandDestroyGenerator(gen);

    kernels::scale<<<GRID_SIZE(n), BLOCK_SIZE, 0, bobr::current_stream()>>>(dst, limit, n);
}

UniformPrim prim_uniform;

void UniformPrim::eval(Node* node) {
    int n = node->get_size();
    float limit = SCALAR(node->inputs[0])->value;
    _uniform(node->get_data(), limit, n);
}

shared_ptr<Node> uniform(int* shape, int ndim, float limit) {
    shared_ptr<ScalarNode> scalar = make_scalar_node(limit, "limit");
    shared_ptr<Node> out_node = make_node(&prim_uniform, {scalar}, shape, ndim);
    out_node->set_label("uniform(" + to_string((int)limit) + ")");
    return out_node;
}

shared_ptr<Node> xavier_uniform(int* shape, int ndim) {
    float limit = sqrtf(6.0f / (shape[0] + shape[ndim - 1]));
    return uniform(shape, ndim, limit);
}

shared_ptr<Node> uniform(vector<int> shape, float limit) {
    return uniform(shape.data(), shape.size(), limit);
}

shared_ptr<Node> xavier_uniform(vector<int> shape) {
    return xavier_uniform(shape.data(), shape.size());
}

// ======================== NORMAL =========================
void _normal(float* dst, float mean, float std, int n) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetStream(gen, bobr::current_stream());
    curandSetPseudoRandomGeneratorSeed(gen, g_seed++);
    curandGenerateNormal(gen, dst, n, mean, std);
    curandDestroyGenerator(gen);
}

NormalPrim prim_normal;

void NormalPrim::eval(Node* node) {
    int n = node->get_size();
    if (n % 2 != 0) throw logic_error("Size should be even.");

    float mean = SCALAR(node->inputs[0])->value;
    float std = SCALAR(node->inputs[1])->value;

    _normal(node->get_data(), mean, std, n);
}

shared_ptr<Node> normal(int* shape, int ndim, float mean, float std) {
    shared_ptr<ScalarNode> scalar_mean = make_scalar_node(mean, "mean");
    shared_ptr<ScalarNode> scalar_std = make_scalar_node(std, "std");

    shared_ptr<Node> out_node = make_node(&prim_normal, {scalar_mean, scalar_std}, shape, ndim);
    out_node->set_label("uniform(" + to_string((int)mean) + to_string((int)std) + ")");
    return out_node;
}

shared_ptr<Node> kaiming_normal(int* shape, int ndim) {
    return normal(shape, ndim, 0, sqrtf(2.0 / shape[0]));
}

shared_ptr<Node> xavier_normal(int* shape, int ndim) {
    return normal(shape, ndim, 0, sqrtf(2.0 / (shape[0] + shape[ndim - 1])));
}

shared_ptr<Node> normal(vector<int> shape, float mean, float std) {
    return normal(shape.data(), shape.size(), mean, std);
}

shared_ptr<Node> kaiming_normal(vector<int> shape) {
    return kaiming_normal(shape.data(), shape.size());
}

shared_ptr<Node> xavier_normal(vector<int> shape) {
    return xavier_normal(shape.data(), shape.size());
}

// ============================================================
//                           ELEMWISE
// ============================================================
template <typename KernelOp>
static void elemwise_eval(Node* node) {
    BroadcastInfo info = broadcast::elemwise(node);
    int n = node->get_size();

    kernels::elemwise<KernelOp><<<GRID_SIZE(n), BLOCK_SIZE, 0, bobr::current_stream()>>>(
        node->inputs[0]->get_data(), node->inputs[1]->get_data(),
        node->get_data(), n, info);
}

shared_ptr<Node> elemwise_helper(Primitive* prim, shared_ptr<Node> in_node_1,
                                 shared_ptr<Node> in_node_2) {
    auto [out_shape, out_ndim] =
        check_shape_return_out_shape::elemwise(in_node_1, in_node_2);
    shared_ptr<Node> out_node =
        make_node(prim, {in_node_1, in_node_2}, out_shape, out_ndim);
    out_node->set_label(in_node_1->label + prim->symbol() + in_node_2->label);
    return out_node;
}

// ========================== ADD ==========================
AddPrim prim_add;

void AddPrim::eval(Node* node) { elemwise_eval<cuAdd>(node); }

void AddPrim::vjp(shared_ptr<Node> fwd, shared_ptr<Node> bwd,
                  shared_ptr<Node> out[]) {
    out[0] = bwd;
    out[1] = bwd;
}

shared_ptr<Node> add(shared_ptr<Node> a, shared_ptr<Node> b) {
    return elemwise_helper(&prim_add, a, b);
}

// ========================== SUB ==========================
SubPrim prim_sub;

void SubPrim::eval(Node* node) { elemwise_eval<cuSub>(node); }

void SubPrim::vjp(shared_ptr<Node> fwd, shared_ptr<Node> bwd,
                  shared_ptr<Node> out[]) {
    out[0] = bwd;
    out[1] = neg(bwd);
}

shared_ptr<Node> sub(shared_ptr<Node> a, shared_ptr<Node> b) {
    return elemwise_helper(&prim_sub, a, b);
}

// ========================== MUL ==========================
MulPrim prim_mul;

void MulPrim::eval(Node* node) { elemwise_eval<cuMul>(node); }

void MulPrim::vjp(shared_ptr<Node> fwd, shared_ptr<Node> bwd,
                  shared_ptr<Node> out[]) {
    auto A = copy_node_no_prim(fwd->inputs[0]);
    auto B = copy_node_no_prim(fwd->inputs[1]);

    out[0] = mul(bwd, B);
    out[1] = mul(bwd, A);
}

shared_ptr<Node> mul(shared_ptr<Node> a, shared_ptr<Node> b) {
    return elemwise_helper(&prim_mul, a, b);
}

// ========================== DIV ==========================
DivPrim prim_div;

void DivPrim::eval(Node* node) { elemwise_eval<cuDiv>(node); }

void DivPrim::vjp(shared_ptr<Node> fwd, shared_ptr<Node> bwd,
                  shared_ptr<Node> out[]) {
    auto A = copy_node_no_prim(fwd->inputs[0]);
    auto B = copy_node_no_prim(fwd->inputs[1]);
    out[0] = div(bwd, B);
    auto B_grad = bwd_div(A, B, bwd);
    out[1] = reduce_to_shape(B_grad, B->get_shape(), B->get_ndim());
}

shared_ptr<Node> div(shared_ptr<Node> a, shared_ptr<Node> b) {
    return elemwise_helper(&prim_div, a, b);
}

// ======================== BWD_DIV ========================
BwdDivPrim prim_bwd_div;

void BwdDivPrim::eval(Node* node) {
    int n = node->get_size();
    auto A = node->inputs[0];
    auto B = node->inputs[1];
    auto prev_grad = node->inputs[2];

    auto info = broadcast::elemwise(
        A->get_shape(), B->get_shape(),
        A->get_strides(), B->get_strides(),
        node->get_shape(), node->get_ndim(),
        A->get_ndim(), B->get_ndim());

    kernels::bwd_div<<<GRID_SIZE(n), BLOCK_SIZE, 0, bobr::current_stream()>>>(
        A->get_data(), B->get_data(), prev_grad->get_data(), node->get_data(), n, info);

    CUDA_LAST_ERROR();
}

shared_ptr<Node> bwd_div(shared_ptr<Node> A_copy, shared_ptr<Node> B_copy, shared_ptr<Node> bwd) {
    auto node = make_node(&prim_bwd_div, {A_copy, B_copy, bwd},
                          bwd->get_shape(), bwd->get_ndim());
    return node;
}

// ========================== POW ==========================
PowPrim prim_pow;

void PowPrim::eval(Node* node) {
    int n = node->get_size();
    kernels::pow<<<GRID_SIZE(n), BLOCK_SIZE, 0, bobr::current_stream()>>>(
        node->inputs[0]->get_data(), node->inputs[1]->get_data(), node->get_data(), n);
}

// c = a ^ scalar
// dc/da = dc/dc * dc/da = bwd * (a^(scalar-1) * scalar) = bwd * (c/a * scalar)
void PowPrim::vjp(shared_ptr<Node> fwd, shared_ptr<Node> bwd,
                  shared_ptr<Node> out[]) {
    out[0] = bwd_pow(fwd, bwd);
}

shared_ptr<Node> pow(shared_ptr<Node> a, shared_ptr<Node> exp) {
    shared_ptr<Node> out_node =
        make_node(&prim_pow, {a, exp}, a->get_shape(), a->get_ndim());
    return out_node;
}

// ======================== BWD POW ========================
BwdPowPrim prim_bwd_pow;

void BwdPowPrim::eval(Node* node) {
    int n = node->get_size();
    kernels::fused_bwd_pow<<<GRID_SIZE(n), BLOCK_SIZE, 0, bobr::current_stream()>>>(
        node->inputs[0]->get_data(), node->inputs[1]->get_data(), node->inputs[2]->get_data(),
        node->get_data(), n);
}

shared_ptr<Node> bwd_pow(shared_ptr<Node> fwd, shared_ptr<Node> bwd) {
    auto mantissa = copy_node_no_prim(fwd->inputs[0]);
    auto exponent = copy_node_no_prim(fwd->inputs[1]);
    shared_ptr<Node> out_node =
        make_node(&prim_bwd_pow, {mantissa, exponent, bwd},
                  mantissa->get_shape(), mantissa->get_ndim());
    return out_node;
}

// ======================== MATMUL =========================
MatMulPrim prim_matmul;

void MatMulPrim::eval(Node* node) {
    static cublasHandle_t handle = []() {
        cublasHandle_t h;
        cublasCreate(&h);
        cublasSetStream(h, bobr::current_stream());
        cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH);
        return h;
    }();

    float* a_data = node->inputs[0]->get_data();
    float* b_data = node->inputs[1]->get_data();
    float* res_data = node->get_data();
    float alpha = 1.0f, beta = 0.0f;

    bool a_t = node->inputs[0]->transposed;
    bool b_t = node->inputs[1]->transposed;

    int a_ndim = node->inputs[0]->h_meta->ndim;
    int b_ndim = node->inputs[1]->h_meta->ndim;
    int* a_shape = node->inputs[0]->h_meta->shape();
    int* b_shape = node->inputs[1]->h_meta->shape();

    // Physical storage: A is [a_rows, a_cols], B is [b_rows, b_cols]
    int a_rows = a_shape[a_ndim - 2];
    int a_cols = a_shape[a_ndim - 1];
    int b_rows = b_shape[b_ndim - 2];
    int b_cols = b_shape[b_ndim - 1];

    // Logical matmul: C = op(A) @ op(B)
    //   op(A) is [M, K], op(B) is [K, N], C is [M, N]
    int M = a_t ? a_cols : a_rows;
    int K = a_t ? a_rows : a_cols;
    int N = b_t ? b_rows : b_cols;

    // Batch
    int a_batch = 1, b_batch = 1;
    for (int i = 0; i < a_ndim - 2; i++) a_batch *= a_shape[i];
    for (int i = 0; i < b_ndim - 2; i++) b_batch *= b_shape[i];
    int batch = std::max(a_batch, b_batch);

    // Row-major trick: to get C = A @ B in row-major,
    // tell cuBLAS (column-major) to compute C^T = B^T @ A^T.
    //
    // A row-major [r, c] matrix in memory IS a column-major [c, r] matrix.
    // So cuBLAS sees:
    //   physical A as col-major [a_cols, a_rows]  (already transposed vs logical)
    //   physical B as col-major [b_cols, b_rows]  (already transposed vs logical)
    //
    // We want cuBLAS to compute: C^T = op'(B) @ op'(A)
    //
    // For B (cuBLAS's first matrix):
    //   cuBLAS sees physical B as col-major [b_cols, b_rows].
    //   We need logical B^T for the row-major trick.
    //   If b_t == false: logical B is [b_rows, b_cols], we need B^T = [b_cols, b_rows].
    //     cuBLAS already sees [b_cols, b_rows] → OP_N
    //   If b_t == true: logical B is [b_cols, b_rows], we need B^T = [b_rows, b_cols].
    //     cuBLAS sees [b_cols, b_rows], needs [b_rows, b_cols] → OP_T
    //
    // Same logic for A (cuBLAS's second matrix):
    //   If a_t == false: OP_N
    //   If a_t == true:  OP_T

    cublasOperation_t op_cublas_B = b_t ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_cublas_A = a_t ? CUBLAS_OP_T : CUBLAS_OP_N;

    // Leading dimensions: always the number of columns in physical
    // row-major storage. This equals the number of rows in cuBLAS's
    // column-major view.
    int ld_b = b_cols;
    int ld_a = a_cols;
    int ld_c = N;  // output is [M, N] in row-major, so ld = N

    int a_stride = a_rows * a_cols;
    int b_stride = b_rows * b_cols;
    int c_stride = M * N;

    if (b_ndim < a_ndim) b_stride = 0;
    if (a_ndim < b_ndim) a_stride = 0;

    // cuBLAS: C^T(N x M) = op'(B)(N x K) @ op'(A)(K x M)
    //   m = N, n = M, k = K
    cublasGemmStridedBatchedEx(
        handle, op_cublas_B, op_cublas_A, N, M, K, &alpha, b_data,
        CUDA_R_32F, ld_b, b_stride, a_data, CUDA_R_32F, ld_a, a_stride,
        &beta, res_data, CUDA_R_32F, ld_c, c_stride, batch,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void MatMulPrim::vjp(shared_ptr<Node> fwd, shared_ptr<Node> bwd,
                     shared_ptr<Node> out[]) {
    auto A = copy_node_no_prim(fwd->inputs[0]);
    auto B = copy_node_no_prim(fwd->inputs[1]);
    A->transposed = true;
    B->transposed = true;
    out[0] = matmul(bwd, B);
    out[1] = matmul(A, bwd);
}

shared_ptr<Node> matmul(shared_ptr<Node> a, shared_ptr<Node> b) {
    auto [out_shape, out_ndim] = check_shape_return_out_shape::matmul(a, b);
    shared_ptr<Node> out_node =
        make_node(&prim_matmul, {a, b}, out_shape, out_ndim);
    out_node->set_label(a->label + prim_matmul.symbol() + b->label);
    return out_node;
}

// ============================================================
//                          REDUCTION LD
// ============================================================
struct cubSum {
    static cudaError_t reduce(void* tmp, size_t& tmp_bytes, float* in,
                              float* out, int n, int* begin, int* end) {
        return cub::DeviceSegmentedReduce::Sum(tmp, tmp_bytes, in, out, n,
                                               begin, end, bobr::current_stream());
    }
};

struct cubMax {
    static cudaError_t reduce(void* tmp, size_t& tmp_bytes, float* in,
                              float* out, int n, int* begin, int* end) {
        return cub::DeviceSegmentedReduce::Max(tmp, tmp_bytes, in, out, n,
                                               begin, end, bobr::current_stream());
    }
};

struct cubMin {
    static cudaError_t reduce(void* tmp, size_t& tmp_bytes, float* in,
                              float* out, int n, int* begin, int* end) {
        return cub::DeviceSegmentedReduce::Min(tmp, tmp_bytes, in, out, n,
                                               begin, end, bobr::current_stream());
    }
};

template <typename KernelOp>
static void reduction_ld_eval(Node* node) {
    float* res_data = node->get_data();
    float* input_data = node->inputs[0]->get_data();
    int* shape = node->inputs[0]->h_meta->shape();
    int ndim = node->inputs[0]->h_meta->ndim;
    int size = node->inputs[0]->h_meta->size;

    int last_dim = shape[ndim - 1];
    int num_segments = size / last_dim;

    Block* b_ptr;
    MemoryPool::call().allocate(&b_ptr, (num_segments + 1) * sizeof(int));
    int* offsets = (int*)b_ptr->d_ptr;

    kernels::reduction_offsets<<<GRID_SIZE(num_segments + 1), BLOCK_SIZE, 0, bobr::current_stream()>>>(
        offsets, last_dim, num_segments);

    size_t temp_storage_bytes = 0;

    KernelOp::reduce(nullptr, temp_storage_bytes, input_data, res_data,
                     num_segments, offsets, offsets + 1);

    Block* temp_block;
    MemoryPool::call().allocate(&temp_block, temp_storage_bytes);

    KernelOp::reduce(temp_block->d_ptr, temp_storage_bytes, input_data,
                     res_data, num_segments, offsets, offsets + 1);

    MemoryPool::call().free(temp_block);
    MemoryPool::call().free(b_ptr);
}

shared_ptr<Node> reduction_ld_helper(Primitive* prim,
                                     shared_ptr<Node> in_node) {
    int out_ndim = in_node->get_ndim() - 1;
    int out_shape[out_ndim];
    for (int i = 0; i < out_ndim; ++i) out_shape[i] = in_node->get_shape()[i];
    shared_ptr<Node> out_node = make_node(prim, {in_node}, out_shape, out_ndim);
    out_node->set_label(in_node->label + "." + prim->symbol() + "()");
    return out_node;
}

// ===================== Max & Min Bwd =====================
MaxMinBwdPrim prim_max_min_bwd;

void MaxMinBwdPrim::eval(Node* node) {
    auto fwd_node = node->inputs[0];
    auto grad = node->inputs[1];
    int rows = fwd_node->get_size();
    int last_dim = fwd_node->inputs[0]->get_size() / rows;
    kernels::max_min_bwd_last_dim<<<GRID_SIZE(rows * last_dim), BLOCK_SIZE, 0, bobr::current_stream()>>>(
        fwd_node->inputs[0]->get_data(),  // original input [N, D]
        fwd_node->get_data(),              // max values [N]
        grad->get_data(),                  // upstream gradient [N]
        node->get_data(),                  // output gradient [N, D]
        rows, last_dim);
}

shared_ptr<Node> max_min_bwd(shared_ptr<Node> fwd_node, shared_ptr<Node> grad) {
    auto out_node = make_node(&prim_max_min_bwd, {fwd_node, grad},
                              fwd_node->inputs[0]->get_shape(), fwd_node->inputs[0]->get_ndim());
    return out_node;
}

// ========================= SUMLD =========================
SumLDPrim prim_sum_ld;

void SumLDPrim::eval(Node* node) { reduction_ld_eval<cubSum>(node); }

void SumLDPrim::vjp(shared_ptr<Node> fwd, shared_ptr<Node> bwd,
                    shared_ptr<Node> out[]) {
    out[0] = expand_ld(bwd, fwd->inputs[0]->get_shape(), fwd->inputs[0]->get_ndim());
}

shared_ptr<Node> sum_ld(shared_ptr<Node> a) {
    return reduction_ld_helper(&prim_sum_ld, a);
}

// ========================= MINLD =========================
MinLDPrim prim_min_ld;

void MinLDPrim::eval(Node* node) { reduction_ld_eval<cubMin>(node); }

void MinLDPrim::vjp(shared_ptr<Node> fwd, shared_ptr<Node> grad,
                    shared_ptr<Node> out[]) {
    out[0] = max_min_bwd(fwd, grad);
}

shared_ptr<Node> min_ld(shared_ptr<Node> a) {
    return reduction_ld_helper(&prim_min_ld, a);
}

// ========================= MAXLD =========================
MaxLDPrim prim_max_ld;

void MaxLDPrim::eval(Node* node) { reduction_ld_eval<cubMax>(node); }

void MaxLDPrim::vjp(shared_ptr<Node> fwd, shared_ptr<Node> grad,
                    shared_ptr<Node> out[]) {
    out[0] = max_min_bwd(fwd, grad);
}

shared_ptr<Node> max_ld(shared_ptr<Node> a) {
    return reduction_ld_helper(&prim_max_ld, a);
}

// ======================= EXPANDLD ========================
ExpandLDPrim prim_expand_ld;

void ExpandLDPrim::eval(Node* node) {
    int n = node->get_size();
    kernels::expand_last_dim<<<GRID_SIZE(n), BLOCK_SIZE, 0, bobr::current_stream()>>>(
        node->inputs[0]->get_data(), node->get_data(),
        node->get_shape()[node->get_ndim() - 1], n);
}

shared_ptr<Node> expand_ld(shared_ptr<Node> a, int* out_shape, int out_ndim) {
    bool same_ndim_case =
        (a->get_ndim() == out_ndim) && (a->get_shape()[out_ndim - 1] == 1);
    bool different_ndim_case = (a->get_ndim() == out_ndim - 1);
    if (!(same_ndim_case || different_ndim_case))
        throw std::logic_error("Cannot broadcast this.");
    for (int i = 0; i < out_ndim - 1; ++i)
        if (a->h_meta->shape()[i] != out_shape[i])
            throw std::logic_error("Cannot broadcast this.");

    auto out_node = make_node(&prim_expand_ld, {a}, out_shape, out_ndim);
    return out_node;
}

// ============================================================
//                           REDUCTION
// ============================================================
shared_ptr<Node> reduction_helper(Primitive* prim, shared_ptr<Node> a, int dim,
                                  bool keepdims) {
    // 0. handle negative indexing
    int ndim = a->get_ndim();
    dim = (dim + ndim) % ndim;

    if (dim == ndim - 1 && !keepdims) return reduction_ld_helper(prim, a);

    // For 2D tensors summing along dimension 0, use transpose for correct semantics
    if (ndim == 2 && dim == 0) {
        // [M, N] -> transpose to [N, M] -> sum last dim -> [N]
        auto transposed = transpose(a);
        auto summed = reduction_ld_helper(prim, transposed);
        if (keepdims) {
            int* res_shape = new int[2];
            res_shape[0] = 1;
            res_shape[1] = a->get_shape()[1];  // N
            return reshape(summed, res_shape, 2);
        }
        return summed;
    }

    // For other cases, use the original algorithm
    // 1. reshape, put the dim to the end
    int* shape_copy = new int[ndim];
    int val = a->get_shape()[dim];
    int j = 0;
    for (int i = 0; i < ndim; ++i)
        if (i != dim) shape_copy[j++] = a->h_meta->shape()[i];
    shape_copy[ndim - 1] = val;

    auto reshaped_node = reshape(a, shape_copy, ndim);

    // 2. make contiguous
    auto contiguous_node = contiguous(reshaped_node);

    // 3. perform last dim reduction
    auto node_after_sum = reduction_ld_helper(prim, contiguous_node);

    // 4. if keepdims = True, insert another dim
    if (keepdims) {
        int* res_shape = new int[ndim];
        for (int i = 0; i < ndim; ++i)
            res_shape[i] = (i == dim) ? 1 : a->get_shape()[i];
        std::shared_ptr<Node> node = reshape(node_after_sum, res_shape, ndim);
        return node;
    }
    return node_after_sum;
}

// reduce all dims
shared_ptr<Node> reduction_helper(Primitive* prim, shared_ptr<Node> a) {
    // 1. reshape to [size]
    int* new_shape = new int[1];
    new_shape[0] = a->get_size();
    auto reshaped_node = reshape(a, new_shape, 1);

    // 2. make contiguous
    auto contiguous_node = contiguous(reshaped_node);

    // 3. perform last dim reduction
    auto node_after_sum = reduction_ld_helper(prim, contiguous_node);

    return node_after_sum;
}

// ========================== SUM ==========================
SumPrim prim_sum;

void SumPrim::eval(Node* node) { reduction_ld_eval<cubSum>(node); }

shared_ptr<Node> sum(shared_ptr<Node> a, int dim, bool keepdims) {
    if (dim == 52 && keepdims == false) {
        return reduction_helper(&prim_sum_ld, a);
    }
    return reduction_helper(&prim_sum_ld, a, dim, keepdims);
}

// ========================== MIN ==========================
MinPrim prim_min;

void MinPrim::eval(Node* node) { reduction_ld_eval<cubSum>(node); }

shared_ptr<Node> min(shared_ptr<Node> a, int dim, bool keepdims) {
    if (dim == 52 && keepdims == false) {
        return reduction_helper(&prim_min_ld, a);
    }
    return reduction_helper(&prim_min_ld, a, dim, keepdims);
}

// ========================== MAX ==========================
MaxPrim prim_max;

void MaxPrim::eval(Node* node) { reduction_ld_eval<cubSum>(node); }

shared_ptr<Node> max(shared_ptr<Node> a, int dim, bool keepdims) {
    if (dim == 52 && keepdims == false) {
        return reduction_helper(&prim_max_ld, a);
    }
    return reduction_helper(&prim_max_ld, a, dim, keepdims);
}

// ======================= TRANSPOSE =======================
TransposePrim prim_transpose;

void TransposePrim::eval(Node* node) {
    int n = node->get_size();
    int last_dim = node->inputs[0]->get_shape()[node->inputs[0]->get_ndim() - 1];
    int second_last_dim = node->inputs[0]->get_shape()[node->inputs[0]->get_ndim() - 2];

    kernels::transpose<<<GRID_SIZE(n), BLOCK_SIZE, 0, bobr::current_stream()>>>(
        node->inputs[0]->get_data(), node->get_data(),
        second_last_dim, last_dim, n);
}

void TransposePrim::vjp(shared_ptr<Node> fwd, shared_ptr<Node> bwd,
                        shared_ptr<Node> out[]) {
    out[0] = transpose(bwd);
}

shared_ptr<Node> transpose(shared_ptr<Node> a) {
    int ndim = a->get_ndim();
    if (ndim < 2) throw std::logic_error("Cannot transpose tensor with < 2 dims");

    int* new_shape = new int[ndim];
    for (int i = 0; i < ndim; ++i) new_shape[i] = a->get_shape()[i];

    // Swap last two dimensions
    int tmp = new_shape[ndim - 1];
    new_shape[ndim - 1] = new_shape[ndim - 2];
    new_shape[ndim - 2] = tmp;

    shared_ptr<Node> out_node = make_node(&prim_transpose, {a}, new_shape, ndim);
    out_node->set_label(a->label + ".transpose()");
    return out_node;
}

// ====================== CONTIGUOUS =======================
ContiguousPrim prim_contiguous;

void ContiguousPrim::eval(Node* node) {
    int n = node->get_size();
    ContiguousInfo info = broadcast::contiguous(node);
    kernels::contiguous<<<GRID_SIZE(n), BLOCK_SIZE, 0, bobr::current_stream()>>>(
        node->inputs[0]->get_data(), node->get_data(), n, info);
}

void ContiguousPrim::vjp(shared_ptr<Node> fwd, shared_ptr<Node> bwd,
                         shared_ptr<Node> out[]) {
    out[0] = contiguous(bwd);
}

shared_ptr<Node> contiguous(shared_ptr<Node> a) {
    shared_ptr<Node> out_node =
        make_node(&prim_contiguous, {a}, a->get_shape(), a->get_ndim());
    out_node->set_label(a->label + "." + prim_contiguous.symbol() + "()");
    return out_node;
}

// ======================== RESHAPE ========================
ReshapePrim prim_reshape;

void ReshapePrim::vjp(shared_ptr<Node> fwd, shared_ptr<Node> bwd,
                      shared_ptr<Node> out[]) {
    out[0] = reshape(bwd, fwd->inputs[0]->get_shape(),
                     fwd->inputs[0]->get_ndim());
}

void ReshapePrim::allocate(Node* node) {
    node->d_data = node->inputs[0]->d_data;  // share buffer
}

shared_ptr<Node> reshape(shared_ptr<Node> a, int* new_shape, int new_ndim) {
    // quick check
    if (utils::numel(a->get_ndim(), a->get_shape()) !=
        utils::numel(new_ndim, new_shape))
        throw std::logic_error("Cannot broadcast this.");

    // compute new strides
    int* new_strides = new int[new_ndim];
    utils::compute_strides(new_strides, new_ndim, new_shape);

    shared_ptr<Node> out_node =
        make_node(&prim_reshape, {a}, new_shape, new_ndim);
    out_node->set_label(a->label + ".reshape()");
    return out_node;
}

// ==================== REDUCE TO SHAPE ====================
ReduceToShapePrim prim_reduce_to_shape;

shared_ptr<Node> reduce_to_shape(shared_ptr<Node> a, int* out_shape,
                                 int out_ndim) {
    int src_ndim = a->get_ndim();
    if (src_ndim == out_ndim) {
        bool match = true;
        for (int i = 0; i < src_ndim; i++)
            if (a->get_shape()[i] != out_shape[i]) {
                match = false;
                break;
            }
        if (match) return a;
    }
    auto out_node = a;

    // 1. left-pad target with 1s
    int pad = src_ndim - out_ndim;

    // 2. collapse extra leading dims
    for (int i = 0; i < pad; i++) out_node = sum(out_node, 0, false);

    // 3. same rank — reduce any remaining broadcast dims (keepdims)
    for (int i = 0; i < out_ndim; i++)
        if (out_shape[i] == 1 && out_node->h_meta->shape()[i] != 1)
            out_node = sum(out_node, i, true);

    return out_node;
}

shared_ptr<Node> reduce_to_shape(shared_ptr<Node> a, vector<int> out_shape) {
    return reduce_to_shape(a, out_shape.data(), out_shape.size());
}

// ======================== LINEAR =========================
shared_ptr<Node> linear(shared_ptr<Node> a, int out_dim) {
    int in_dim = a->get_shape()[a->get_ndim() - 1];
    int w_shape[2] = {in_dim, out_dim};
    int b_shape[2] = {1, out_dim};
    auto W = kaiming_normal(w_shape, 2);
    auto b = full(b_shape, 2, 0.0f);
    W->requires_grad = true;
    b->requires_grad = true;
    W->set_label("W");
    b->set_label("b");
    return add(matmul(a, W), b);
}

// ===================== CROSS ENTROPY =====================
SoftmaxPrim prim_softmax;

//  <===================== from logits ========================>
shared_ptr<Node> cross_entropy(shared_ptr<Node> logits, shared_ptr<Node> targets) {
    // stable log-softmax
    auto maxes = max(logits, -1, true);           // [batch, 1]
    auto shifted = sub(logits, maxes);      // [batch, classes]
    auto exp_shifted = exp(shifted);        // [batch, classes]
    auto sum_exp = sum(exp_shifted, -1, true);     // [batch, 1]
    auto log_sum = log(sum_exp);            // [batch, 1]
    auto log_softmax = sub(shifted, log_sum); // [batch, classes]
    log_softmax->set_label("sm_out");

    // cross entropy: -sum(targets * log_softmax) / batch
    auto prod = mul(targets, log_softmax);              // [batch, classes]
    auto sum_classes = sum(prod, -1, false);     // [batch]
    auto neg_sum_classes = neg(sum_classes);
    auto sum_all = sum(neg_sum_classes);
    auto loss = div(sum_all, full({1}, logits->get_shape()[0]));

    return loss;
}

//  <===================== from sm values ====================>
shared_ptr<Node> cross_entropy_softmax(shared_ptr<Node> probs, shared_ptr<Node> targets) {
    // CE = -sum(targets * log(probs)) / batch
    auto log_probs = log(probs);                       // [batch, classes]
    auto prod = mul(targets, log_probs);               // [batch, classes]
    auto sum_classes = sum(prod, -1, false);            // [batch]
    auto neg_sum_classes = neg(sum_classes);
    auto sum_all = sum(neg_sum_classes);
    auto loss = div(sum_all, full({1}, probs->get_shape()[0]));
    return loss;
}

// ======================== SOFTMAX ========================
shared_ptr<Node> softmax(shared_ptr<Node> logits) {
    auto maxes = max(logits, -1, true);        // [batch, 1]
    auto shifted = sub(logits, maxes);         // [batch, classes]
    auto exp_shifted = exp(shifted);           // [batch, classes]
    auto sum_exp = sum(exp_shifted, -1, true); // [batch, 1]
    auto result = div(exp_shifted, sum_exp);   // [batch, classes]
    result->set_label("softmax(" + logits->label + ")");
    return result;
}

// ============================================================
//                        FUSED OPS
// ============================================================
