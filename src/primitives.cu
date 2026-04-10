// ALL OPS 
#include "primitives.h"
#include "kernels.cuh"
#include "nodes.h"
#include <memory>
#include "utils.h"
#include <cublas_v2.h>
#include <cub/cub.cuh>
#include <pthread/qos.h>


#define GRID_SIZE(n) ((n + 255) / 256)
#define BLOCK_SIZE 256

using namespace std;

// -----------------------------------------------------------------
//                          UNARY
// -----------------------------------------------------------------
template <typename KernelOp>
static void unary_eval(Node* node) {
    int n = node->get_size();
    kernels::unary<KernelOp><<<GRID_SIZE(n), BLOCK_SIZE>>>(
        node->inputs[0]->get_data, node->get_data(), n);
}

shared_ptr<Node> unary_helper(Primitive* prim, shared_ptr<Node> in_node, char* label) {
    shared_ptr<Node> out_node = make_node(prim, {in_node}, in_node->get_shape(), in_node->get_ndim());
    out_node->set_label(label);
    return out_node;
}

// ============================================================
//                         RELU
// ============================================================
static struct : Primitive {
    void eval(Node* node) override { unary_eval<cuReLU>(node); }
    void vjp(shared_ptr<Node> fwd, shared_ptr<Node> dC,
             shared_ptr<Node> out[]) override {
        auto A = copy_fwd(fwd->inputs[0]);
        out[0] = bwd_relu(dC, A);
    }
    const char* name() override { return "ReLU"; }
    const char* symbol() override { return "relu"; }
} prim_relu;

shared_ptr<Node> relu(shared_ptr<Node> a) {
    return unary_helper(&prim_relu, a, a->label + ".relu()");
}

// ============================================================
//                         NEG
// ============================================================
static struct : Primitive {
    void eval(Node* node) override { unary_eval<cuNeg>(node); }
    void vjp(shared_ptr<Node> fwd, shared_ptr<Node> dC,
             shared_ptr<Node> out[]) override {

    }
    const char* name() override { return "Neg"; }
    const char* symbol() override { return "-"; }
} prim_neg;

shared_ptr<Node> neg(shared_ptr<Node> a) {

    return unary_helper(&prim_neg, a, "-" + a->label);
}

// ============================================================
//                         EXP
// ============================================================
static struct : Primitive {
    void eval(Node* node) override { unary_eval<cuExp>(node); }
    void vjp(shared_ptr<Node> fwd, shared_ptr<Node> dC,
             shared_ptr<Node> out[]) override {
                
    }
    const char* name() override { return "Exp"; }
    const char* symbol() override { return "exp"; }
} prim_exp;

shared_ptr<Node> exp(shared_ptr<Node> a) {
    return unary_helper(&prim_exp, a, a->label + ".exp()");
}

// ============================================================
//                         SQRT
// ============================================================
static struct : Primitive {
    void eval(Node* node) override { unary_eval<cuSqrt>(node); }
    void vjp(shared_ptr<Node> fwd, shared_ptr<Node> dC,
             shared_ptr<Node> out[]) override {
                
    }
    const char* name() override { return "Sqrt"; }
    const char* symbol() override { return "sqrt"; }
} prim_sqrt;

shared_ptr<Node> sqrt(shared_ptr<Node> a) {
    return unary_helper(&prim_sqrt, a, a->label + ".sqrt()");
}

// ============================================================
//                         LOG
// ============================================================
static struct : Primitive {
    void eval(Node* node) override { unary_eval<cuLog>(node); }
    void vjp(shared_ptr<Node> fwd, shared_ptr<Node> dC,
             shared_ptr<Node> out[]) override {
                
    }
    const char* name() override { return "Log"; }
    const char* symbol() override { return "log"; }
} prim_log;

shared_ptr<Node> log(shared_ptr<Node> a) {
    return unary_helper(&prim_log, a, a->label + ".log()");
}

// -----------------------------------------------------------------
//                          ELEMWISE
// -----------------------------------------------------------------
template <typename KernelOp>
static void elemwise_eval(Node* node)
{
    BroadcastInfo info = broadcast::elemwise(node);
    int n = node->get_size();
    kernels::elemwise<KernelOp><<<GRID_SIZE(n), BLOCK_SIZE>>>(
        node->inputs[0]->get_data(), node->inputs[1]->get_data(), 
        node->get_data(), info);
}

shared_ptr<Node> elemwise_helper(Primitive* prim, shared_ptr<Node> in_node_1, shared_ptr<Node> in_node_2) {
    auto [out_shape, out_ndim] = check_shape_return_out_shape::elemwise(in_node_1, in_node_2);
    shared_ptr<Node> out_node = make_node(prim, {in_node_1, in_node_2}, out_shape, out_ndim);
    out_node->set_label(in_node_1->label + prim->symbol() + in_node_2->label);
    return out_node;
}

// ============================================================
//                          ADD
// ============================================================
static struct : Primitive {
    void eval(Node* node) override { elemwise_eval<cuAdd>(node); }
    void vjp(shared_ptr<Node> fwd, shared_ptr<Node> dC,
             shared_ptr<Node> out[]) override {
        out[0] = skip(dC);
        out[1] = skip(dC);
    }
    const char* name() override { return "Add"; }
    const char* symbol() override { return "+"; }
} prim_add;

shared_ptr<Node> add(shared_ptr<Node> a, shared_ptr<Node> b) {
    return elemwise_helper(&prim_add, a, b);
}

// ============================================================
//                          MUL
// ============================================================
static struct : Primitive {
    void eval(Node* node) override { elemwise_eval<cuMul>(node); }
    void vjp(shared_ptr<Node> fwd, shared_ptr<Node> dC,
             shared_ptr<Node> out[]) override {
        auto B = copy_fwd(fwd->inputs[1]);
        auto A = copy_fwd(fwd->inputs[0]);
        out[0] = mul(dC, B);
        out[1] = mul(dC, A);
    }
    const char* name() override { return "Mul"; }
    const char* symbol() override { return "*"; }
} prim_mul;

shared_ptr<Node> mul(shared_ptr<Node> a, shared_ptr<Node> b) {
    return elemwise_helper(&prim_mul, a, b);
}

// ============================================================
//                        MATMUL
// ============================================================
static struct : Primitive {
    void eval(Node* node) override {
        static cublasHandle_t handle = [](){
            cublasHandle_t h;
            cublasCreate(&h);
            cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
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

        // Leading dimensions: always the number of columns in physical row-major storage.
        // This equals the number of rows in cuBLAS's column-major view.
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
        cublasGemmStridedBatchedEx(handle,
            op_cublas_B, op_cublas_A,
            N, M, K,
            &alpha,
            b_data, CUDA_R_32F, ld_b, b_stride,
            a_data, CUDA_R_32F, ld_a, a_stride,
            &beta,
            res_data, CUDA_R_32F, ld_c, c_stride,
            batch,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );
    }
    void vjp(shared_ptr<Node> fwd, shared_ptr<Node> dC,
             shared_ptr<Node> out[]) override {
        auto A = copy_fwd(fwd->inputs[0], /*transposed=*/true);
        auto B = copy_fwd(fwd->inputs[1], /*transposed=*/true);
        out[0] = matmul(dC, B);
        out[1] = matmul(A, dC);
    }
    const char* name() override { return "MatMul"; }
    const char* symbol() override { return "@"; }
} prim_matmul;

shared_ptr<Node> matmul(shared_ptr<Node> a, shared_ptr<Node> b) {
    auto [out_shape, out_ndim] = check_shape_return_out_shape::matmul(a, b);
    shared_ptr<Node> out_node = make_node(&prim_matmul, {a, b}, out_shape, out_ndim);
    out_node->set_label(a->label + prim_matmul.symbol() + b->label);
    return out_node;
}

// -----------------------------------------------------------------
//                          REDUCTION LD
// -----------------------------------------------------------------
struct cubSum {
    static cudaError_t reduce(void* tmp, size_t& tmp_bytes, float* in, float* out, int n, int* begin, int* end) 
    { return cub::DeviceSegmentedReduce::Sum(tmp, tmp_bytes, in, out, n, begin, end); }
};

struct cubMax {
    static cudaError_t reduce(void* tmp, size_t& tmp_bytes, float* in, float* out, int n, int* begin, int* end)
    { return cub::DeviceSegmentedReduce::Max(tmp, tmp_bytes, in, out, n, begin, end); }
};

struct cubMin {
    static cudaError_t reduce(void* tmp, size_t& tmp_bytes, float* in, float* out, int n, int* begin, int* end)
    { return cub::DeviceSegmentedReduce::Min(tmp, tmp_bytes, in, out, n, begin, end); }
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

    kernels::reduction_offsets<<<GRID_SIZE(num_segments + 1), BLOCK_SIZE>>>(offsets, last_dim, num_segments);

    size_t temp_storage_bytes = 0;

    ReduceOp::reduce(nullptr, temp_storage_bytes, input_data, res_data, num_segments, offsets, offsets + 1);

    Block* temp_block;
    MemoryPool::call().allocate(&temp_block, temp_storage_bytes);
    
    ReduceOp::reduce(temp_block->d_ptr, temp_storage_bytes, input_data, res_data, num_segments, offsets, offsets + 1);
    
    MemoryPool::call().free(temp_block);
    MemoryPool::call().free(b_ptr);
}

shared_ptr<Node> reduction_ld_helper(Primitive* prim, shared_ptr<Node> in_node) {
    int out_ndim = in_node->get_ndim() - 1;
    int out_shape[out_ndim];
    for (int i = 0; i < res_ndim; ++i) out_shape[i] = in_node->get_shape()[i];
    shared_ptr<Node> out_node = make_node(prim, {in_node}, in_node->get_shape(), in_node->get_ndim());
    out_node->set_label(in_node->label + "." + prim->symbol() + "()");
    return out_node;
}

// ============================================================
//                          SUMLD
// ============================================================
static struct : Primitive {
    void eval(Node* node) override { reduction_ld_eval<cubSum>(node); }
    void vjp(shared_ptr<Node> fwd, shared_ptr<Node> dC,
             shared_ptr<Node> out[]) override {
    }
    const char* name() override { return "SumLD"; }
    const char* symbol() override { return "sum_ld"; }
} prim_sum_ld;

shared_ptr<Node> sum_ld(shared_ptr<Node> a) {
    return reduction_ld_helper(&prim_sum_ld, a);
}

// ============================================================
//                          MINLD
// ============================================================
static struct : Primitive {
    void eval(Node* node) override { reduction_ld_eval<cubMin>(node); }
    void vjp(shared_ptr<Node> fwd, shared_ptr<Node> dC,
             shared_ptr<Node> out[]) override {
    }
    const char* name() override { return "MinLD"; }
    const char* symbol() override { return "min_ld"; }
} prim_min_ld;

shared_ptr<Node> min_ld(shared_ptr<Node> a) {
    return reduction_ld_helper(&prim_min_ld, a);
}

// ============================================================
//                          MINLD
// ============================================================
static struct : Primitive {
    void eval(Node* node) override { reduction_ld_eval<cubMax>(node); }
    void vjp(shared_ptr<Node> fwd, shared_ptr<Node> dC,
             shared_ptr<Node> out[]) override {
    }
    const char* name() override { return "MaxLD"; }
    const char* symbol() override { return "max_ld"; }
} prim_max_ld;

shared_ptr<Node> max_ld(shared_ptr<Node> a) {
    return reduction_ld_helper(&prim_max_ld, a);
}

// ============================================================
//                         CONTIGUOUS
// ============================================================
static struct : Primitive {
    void eval(Node* node) override { 
        int n = node->get_size();
        ContiguousInfo info = broadcast::contiguous(node);
        kernels::contiguous<<<GRID_SIZE(n), BLOCK_SIZE>>>(
            node->inputs[0]->get_data(), node->get_data(), n, info);
    }
    void vjp(shared_ptr<Node> fwd, shared_ptr<Node> dC,
             shared_ptr<Node> out[]) override {
    }
    const char* name() override { return "Contiguous"; }
    const char* symbol() override { return "contiguous"; }
} prim_contiguous;

shared_ptr<Node> contiguous(shared_ptr<Node> a) {
    shared_ptr<Node> out_node = make_node(&prim_contiguous, {a}, a->get_shape(), a->get_ndim());
    out_node->set_label(a->label + "." + prim_contiguous.symbol + "()");
    return out_node;
}

// ============================================================
//                         RESHAPE
// ============================================================
static struct : Primitive {
    void eval(Node* node) override {}
    void vjp(shared_ptr<Node> fwd, shared_ptr<Node> dC,
             shared_ptr<Node> out[]) override {}
    const char* name() override { return "Reshape"; }
    const char* symbol() override { return "reshape"; }
} prim_reshape;

shared_ptr<Node> reshape(shared_ptr<Node> a, int* new_shape, int new_ndim) {
    // quick check
    if (utils::numel(a->get_ndim(), a->get_shape()) != utils::numel(new_ndim, new_shape))
        throw std::logic_error("Cannot broadcast this.");

    // compute new strides
    int* new_strides = new int[new_shape];
    utils::compute_strides(res_strides, res_ndim, res_shape);

    shared_ptr<Node> out_node = make_node(&prim_reshape, {a}, new_shape, new_ndim);
    out_node->set_label(a->label + ".reshape()");
    return out_node;
}

// ============================================================
//                     SUM (complex shape logic)
// ============================================================
static struct : Primitive {
    void eval(Node* node) override { sum_last_dim_eval(node); }
    void vjp(shared_ptr<Node> fwd, shared_ptr<Node> dC,
             shared_ptr<Node> out[]) override {
        out[0] = expand_last_dim(dC, fwd->inputs[0]->h_meta->shape(),
                                      fwd->inputs[0]->h_meta->ndim);
    }
    const char* name() override { return "SumLD"; }
    const char* symbol() override { return "∑"; }
} prim_sum_ld;

// sum() has complex decomposition logic, that's fine — 
// it lives right next to its primitive
shared_ptr<Node> sum(shared_ptr<Node> a, int dim, bool keepdims) {
    int ndim = a->h_meta->ndim;
    dim = (dim + ndim) % ndim;
    if (dim == ndim - 1 && !keepdims)
        return make_node(&prim_sum_ld, {a}, ...);
    // transpose → contiguous → reduce_last_dim → reshape
    // ... your existing decomposition logic
}