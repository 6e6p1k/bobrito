#include <cuda_runtime_api.h>
#include <iostream>
#include <cublas_v2.h>
#include <ostream>
#include "allocator.cuh"
#include "node.h"
#include "utils.h"
#include "ops.cuh"
#include "kernels.cuh"
#include <cudart_platform.h>
#include <stdexcept>

#define CUDA_CHECK() { \
    cudaError_t err = cudaDeviceSynchronize(); \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

void compute_offset(int* dst, int last_dim, int num_segments)
{
    int gridDim = (num_segments + 1 + 255) / 256;
    int blockDim = 256;
    kernels::reduction_offsets<<<gridDim, blockDim>>>(dst, last_dim, num_segments);
}



namespace cu_ops
{

void add(InternalNode* res_node)
{
    BroadcastInfo info = broadcast::elemwise(res_node);
    float* res_data = res_node->get_data();
    float* a_data = res_node->inputs[0]->get_data();
    float* b_data = res_node->inputs[1]->get_data();
    int n = res_node->h_meta->size;

    int gridDim = (n + 255) / 256;
    int blockDim = 256;
    kernels::elemwise<cuAdd><<<gridDim, blockDim>>>(a_data, b_data, res_data, info);
    // CUDA_CHECK();
}

void sub(InternalNode* res_node)
{
    BroadcastInfo info = broadcast::elemwise(res_node);
    float* res_data = res_node->get_data();
    float* a_data = res_node->inputs[0]->get_data();
    float* b_data = res_node->inputs[1]->get_data();
    int n = res_node->h_meta->size;

    int gridDim = (n + 255) / 256;
    int blockDim = 256;
    kernels::elemwise<cuSub><<<gridDim, blockDim>>>(a_data, b_data, res_data, info);
    // CUDA_CHECK();
}

void mul(InternalNode* res_node)
{
    BroadcastInfo info = broadcast::elemwise(res_node);
    float* res_data = res_node->get_data();
    float* a_data = res_node->inputs[0]->get_data();
    float* b_data = res_node->inputs[1]->get_data();
    int n = res_node->h_meta->size;

    int gridDim = (n + 255) / 256;
    int blockDim = 256;
    kernels::elemwise<cuMul><<<gridDim, blockDim>>>(a_data, b_data, res_data, info);
    // CUDA_CHECK();
}

void div(InternalNode* res_node)
{
    BroadcastInfo info = broadcast::elemwise(res_node);
    float* res_data = res_node->get_data();
    float* a_data = res_node->inputs[0]->get_data();
    float* b_data = res_node->inputs[1]->get_data();
    int n = res_node->h_meta->size;

    int gridDim = (n + 255) / 256;
    int blockDim = 256;
    kernels::elemwise<cuDiv><<<gridDim, blockDim>>>(a_data, b_data, res_data, info);
    // CUDA_CHECK();
}

// void matmul(InternalNode* res_node) {}
void matmul(InternalNode* res_node) 
{
    // std::cout<<"Inside matmul --------------------------------\n";
    static cublasHandle_t handle = [](){
        cublasHandle_t h;
        cublasCreate(&h);
        cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
        return h;
    }();

    float* a_data = res_node->inputs[0]->get_data();
    float* b_data = res_node->inputs[1]->get_data();
    float* res_data = res_node->get_data();
    float alpha = 1.0f, beta = 0.0f;

    bool a_t = res_node->inputs[0]->transposed;
    bool b_t = res_node->inputs[1]->transposed;

    int a_ndim = res_node->inputs[0]->h_meta->ndim;
    int b_ndim = res_node->inputs[1]->h_meta->ndim;
    int* a_shape = res_node->inputs[0]->h_meta->shape();
    int* b_shape = res_node->inputs[1]->h_meta->shape();

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
    // CUDA_CHECK();
}

// void reduce_last_dim(InternalNode* res_node) {
//     float* res_data = res_node->get_data();
//     float* input_data = res_node->inputs[0]->get_data();
//     int* shape = res_node->inputs[0]->h_meta->shape();
//     int ndim = res_node->inputs[0]->h_meta->ndim;
//     int size = res_node->inputs[0]->h_meta->size;
    
//     int last_dim = shape[ndim - 1];
//     int num_segments = size / last_dim;
    
//     // std::cout << "reduce_last_dim: size=" << size << " last_dim=" << last_dim 
//     //           << " num_segments=" << num_segments << std::endl;

//     Block* b_ptr;
//     MemoryPool::call().allocate(&b_ptr, (num_segments + 1) * sizeof(int));
//     int* offsets = (int*)b_ptr->d_ptr;

//     int gridDim = (num_segments + 1 + 255) / 256;
//     int blockDim = 256;

//     kernels::reduction_offsets<<<gridDim, blockDim>>>(offsets, last_dim, num_segments);

//     Block* temp_storage_block;
//     size_t temp_storage_bytes = 0;
    
//     cub::DeviceSegmentedReduce::Sum(
//         nullptr, temp_storage_bytes,
//         input_data, res_data,
//         num_segments,
//         offsets, offsets + 1
//     );

//     MemoryPool::call().allocate(&temp_storage_block, temp_storage_bytes);
//     float* temp_storage = (float*)temp_storage_block->d_ptr;

//     cub::DeviceSegmentedReduce::Sum(
//         temp_storage, temp_storage_bytes,
//         input_data, res_data,
//         num_segments,
//         offsets, offsets + 1
//     );

//     MemoryPool::call().free(b_ptr);
//     MemoryPool::call().free(temp_storage_block);
//     // CUDA_CHECK();
// }

void sum_last_dim(InternalNode* res_node)
{
    return reduce_last_dim<cubSum>(res_node);
}

void max_last_dim(InternalNode* res_node)
{
    return reduce_last_dim<cubMax>(res_node);
}

void min_last_dim(InternalNode* res_node)
{
}

void skip(InternalNode* res_node)
{
    float* res_grad = res_node->get_data();
    float* a_grad = res_node->inputs[0]->get_data();
    int size = res_node->h_meta->size;

    int gridDim = (size + 255) / 256;
    int blockDim = 256;
    kernels::copy<<<gridDim, blockDim>>>(a_grad, res_grad, size);
    // CUDA_CHECK();
}

void fill(Node* node, float val)
{
    float* data = node->get_data();
    int size = node->h_meta->size;

    int gridDim = (size + 255) / 256;
    int blockDim = 256;
    kernels::fill<<<gridDim, blockDim>>>(data, val, size);
    // CUDA_CHECK();
}

void fill(SourceNode* node, float val)
{
    float* data = node->get_data();
    int size = node->h_meta->size;

    int gridDim = (size + 255) / 256;
    int blockDim = 256;
    kernels::fill<<<gridDim, blockDim>>>(data, val, size);
    // CUDA_CHECK();
}

void fill(InternalNode* node, float val)
{
    float* data = node->get_data();
    int size = node->h_meta->size;

    int gridDim = (size + 255) / 256;
    int blockDim = 256;
    kernels::fill<<<gridDim, blockDim>>>(data, val, size);
    // CUDA_CHECK();
}

void contiguous(InternalNode* res_node)
{
    ContiguousInfo info = broadcast::contiguous(res_node);
    float* res_data = res_node->get_data();
    float* data = res_node->inputs[0]->get_data();
    int size = res_node->h_meta->size;

    int gridDim = (size + 255) / 256;
    int blockDim = 256;

    kernels::contiguous<<<gridDim, blockDim>>>(data, res_data, size, info);
    // CUDA_CHECK();
}

void expand_last_dim(InternalNode* res_node)
{
    float* res_data = res_node->get_data();
    float* data = res_node->inputs[0]->get_data();
    int res_ndim = res_node->h_meta->ndim;
    int last_dim = res_node->h_meta->shape()[res_ndim - 1];
    int size = res_node->h_meta->size;

    int gridDim = (size + 255) / 256;
    int blockDim = 256;

    kernels::expand_last_dim<<<gridDim, blockDim>>>(data, res_data, last_dim, size);
    // CUDA_CHECK();
}

void relu(InternalNode* res_node)
{
    float* res_data = res_node->get_data();
    float* data = res_node->inputs[0]->get_data();
    int size = res_node->h_meta->size;

    int gridDim = (size + 255) / 256;
    int blockDim = 256;

    kernels::relu<<<gridDim, blockDim>>>(data, res_data, size);
    // CUDA_CHECK();
}

void bwd_relu(InternalNode* res_node)
{
    if (res_node->num_inputs != 2) throw std::logic_error("Unacceptable number of inputs.");
    if (res_node->inputs[0] == nullptr) throw std::logic_error("No first input detected.");
    if (res_node->inputs[1] == nullptr) throw std::logic_error("No second input detected.");
    auto bwd_node = res_node->inputs[0];
    auto fwd_node = res_node->inputs[1];

    if (!res_node->inputs[0]->is_data_allocated()) throw std::logic_error("First input doesn't have data allocated.");
    if (!res_node->inputs[1]->is_data_allocated()) throw std::logic_error("Second input doesn't have data allocated.");

    float* fwd_data = fwd_node->get_data();
    float* prev_grad = bwd_node->get_data();
    float* new_grad = res_node->get_data();
    int size = fwd_node->h_meta->size;

    int gridDim = (size + 255) / 256;
    int blockDim = 256;

    kernels::bwd_relu<<<gridDim, blockDim>>>(fwd_data, prev_grad, new_grad, size);
    // CUDA_CHECK();
}
}

// __global__ void contiguous(float* src, float* dst, int* shape, int* strides, int ndim, int size);
