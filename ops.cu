#include <iostream>
#include <cublas_v2.h>
#include <ostream>
#include "node.h"
#include "utils.h"
#include "ops.cuh"
#include "kernels.cuh"




namespace cu_ops
{

void add(InternalNode* res_node)
{
    BroadcastInfo info = broadcast::elemwise(res_node);
    float* res_data = res_node->d_content->values();
    float* a_data = res_node->inputs[0]->d_content->values();
    float* b_data = res_node->inputs[1]->d_content->values();
    int n = res_node->d_content->size();

    int gridDim = (n + 255) / 256;
    int blockDim = 256;
    kernels::elemwise<cuAdd><<<gridDim, blockDim>>>(a_data, b_data, res_data, info);
    cudaDeviceSynchronize();
}

void gadd(InternalNode* res_node)
{
    BroadcastInfo info = broadcast::elemwise(res_node);
    float* res_data = res_node->d_content->values();
    float* a_data = res_node->inputs[0]->d_content->values();
    float* b_data = res_node->inputs[1]->d_content->values();
    int n = res_node->d_content->size();

    int gridDim = (n + 255) / 256;
    int blockDim = 256;
    kernels::elemwise<cuAdd><<<gridDim, blockDim>>>(a_data, b_data, res_data, info);
    cudaDeviceSynchronize();
}

void mul(InternalNode* res_node)
{
    BroadcastInfo info = broadcast::elemwise(res_node);
    float* res_data = res_node->d_content->values();
    float* a_data = res_node->inputs[0]->d_content->values();
    float* b_data = res_node->inputs[1]->d_content->values();
    int n = res_node->d_content->size();

    int gridDim = (n + 255) / 256;
    int blockDim = 256;
    kernels::elemwise<cuMul><<<gridDim, blockDim>>>(a_data, b_data, res_data, info);
    cudaDeviceSynchronize();
}

void gmul(InternalNode* self)
{
    // BroadcastInfo info = broadcast::elemwise(res_node);
    // float* self_grad = self->d_content->grad();
    // cout<<"HUUUUUY"<<endl;
    // float* prev_grad = self->inputs[0]->d_content->grad();
    // cout<<"HUUUUUY"<<endl;
    // float* other_values = res_node->parallel->d_content->values(); 
    // int n = res_node->d_content->size();

    // int gridDim = (n + 255) / 256;
    // int blockDim = 256;
    // kernels::elemwise<cuAdd><<<gridDim, blockDim>>>(prev_grad, other_values, self_grad, info);
    cudaDeviceSynchronize();
}

void skip(InternalNode* res_node)
{
    float* res_grad = res_node->d_content->grad();
    float* a_grad = res_node->inputs[0]->d_content->grad();
    int size = res_node->d_content->size();

    int gridDim = (size + 255) / 256;
    int blockDim = 256;
    kernels::copy<<<gridDim, blockDim>>>(res_grad, a_grad, size);
    cudaDeviceSynchronize();
}

void fill(SourceNode* node, float val)
{
    float* data = node->d_content->values();
    int size = node->d_content->size();

    int gridDim = (size + 255) / 256;
    int blockDim = 256;
    kernels::fill<<<gridDim, blockDim>>>(data, val, size);
    cudaDeviceSynchronize();
}


// float* matmul_cuda_launch(
//     const float* A, const float* B, int batch, int M, int N, int K)
// {
//     // created once, reused for every call
//     static cublasHandle_t handle = [](){
//         cublasHandle_t h;
//         cublasCreate(&h);
//         cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);

//         // allocate 32MB workspace
//         void* workspace;
//         cudaMalloc(&workspace, 32 * 1024 * 1024);
//         cublasSetWorkspace(h, workspace, 32 * 1024 * 1024);

//         return h;
//     }();

//     float* res_ptr;
//     cudaMalloc(&res_ptr, batch * M * N * sizeof(float));

//     float alpha = 1.0f, beta = 0.0f;

//     cublasGemmStridedBatchedEx(handle,
//         CUBLAS_OP_N, CUBLAS_OP_N,
//         N, M, K,
//         &alpha,
//         B, CUDA_R_32F, N, K * N,
//         A, CUDA_R_32F, K, M * K,
//         &beta,
//         res_ptr, CUDA_R_32F, N, M * N,
//         batch,
//         CUBLAS_COMPUTE_32F,
//         CUBLAS_GEMM_DEFAULT_TENSOR_OP
//     );
//     return res_ptr;
// }
}