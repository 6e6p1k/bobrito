#include <iostream>
#include <cublas_v2.h>
#include <ostream>
#include "node.h"
#include "utils.h"
#include "ops.cuh"
#include "kernels.cuh"
#include <cudart_platform.h>




namespace cu_ops
{

void add(InternalNode* res_node)
{
    BroadcastInfo info = broadcast::elemwise(res_node);
    float* res_data = res_node->get_values();
    float* a_data = res_node->inputs[0]->get_values();
    float* b_data = res_node->inputs[1]->get_values();
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
    float* a_data = res_node->inputs[0]->get_values();
    float* b_data = res_node->inputs[1]->get_values();
    int n = res_node->d_content->size();

    int gridDim = (n + 255) / 256;
    int blockDim = 256;
    kernels::elemwise<cuAdd><<<gridDim, blockDim>>>(a_data, b_data, res_data, info);
    cudaDeviceSynchronize();
}

void mul(InternalNode* res_node)
{
    BroadcastInfo info = broadcast::elemwise(res_node);
    float* res_data = res_node->get_values();
    float* a_data = res_node->inputs[0]->get_values();
    float* b_data = res_node->inputs[1]->get_values();
    int n = res_node->d_content->size();

    // cout<<"\n";
    // cout<<"RESULT:"<<res_node->label<<" flipped="<<res_node->flipped<<endl;
    // cout<<"node a: "<<res_node->inputs[0]->label<<" flipped="<<res_node->inputs[0]->flipped<<endl;
    // cout<<"node b: "<<res_node->inputs[1]->label<<" flipped="<<res_node->inputs[1]->flipped<<endl;
    // cout<<"\n";
    
    int gridDim = (n + 255) / 256;
    int blockDim = 256;
    kernels::elemwise<cuMul><<<gridDim, blockDim>>>(a_data, b_data, res_data, info);
    cudaDeviceSynchronize();
}

void matmul(InternalNode* res_node)
{
    MatmulDims dims = broadcast::matmul(res_node);
    
    // created once, reused for every call
    static cublasHandle_t handle = [](){
        cublasHandle_t h;
        cublasCreate(&h);
        cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);

        // allocate 32MB workspace (check later)
        // void* workspace;
        // cudaMalloc(&workspace, 32 * 1024 * 1024);
        // cublasSetWorkspace(h, workspace, 32 * 1024 * 1024);

        return h;
    }();

    float* res_data = res_node->get_values();
    float* a_data = res_node->inputs[0]->get_values();
    float* b_data = res_node->inputs[1]->get_values();
    float alpha = 1.0f, beta = 0.0f;

    cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        dims.N, dims.M, dims.K,
        &alpha,
        b_data, CUDA_R_32F, dims.N, dims.K * dims.N,
        a_data, CUDA_R_32F, dims.K, dims.M * dims.K,
        &beta,
        res_data, CUDA_R_32F, dims.N, dims.M * dims.N,
        dims.batch,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
}

void skip(InternalNode* res_node)
{
    float* res_grad = res_node->get_values();
    float* a_grad = res_node->inputs[0]->get_values();
    int size = res_node->d_content->size();

    int gridDim = (size + 255) / 256;
    int blockDim = 256;
    kernels::copy<<<gridDim, blockDim>>>(a_grad, res_grad, size);
    cudaDeviceSynchronize();
}

void fill(Node* node, float val)
{
    float* data = node->get_values();
    int size = node->d_content->size();

    int gridDim = (size + 255) / 256;
    int blockDim = 256;
    kernels::fill<<<gridDim, blockDim>>>(data, val, size);
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

void fill(InternalNode* node, float val)
{
    float* data = node->d_content->values();
    int size = node->d_content->size();

    int gridDim = (size + 255) / 256;
    int blockDim = 256;
    kernels::fill<<<gridDim, blockDim>>>(data, val, size);
    cudaDeviceSynchronize();
}

}