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
#include <cub/cub.cuh>




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

// void matmul(InternalNode* res_node) {}
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

    bool a_transposed = res_node->inputs[0]->transposed;
    bool b_transposed = res_node->inputs[1]->transposed;
    cublasOperation_t a_cublas_op = a_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t b_cublas_op = b_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;


    cublasGemmStridedBatchedEx(handle,
        b_cublas_op, a_cublas_op,
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

//  float* src, float* dst, int* shape, int ndim, int n)
void reduce_last_dim(InternalNode* res_node) {
    float* res_data = res_node->d_content->values();
    float* input_data = res_node->inputs[0]->d_content->values();
    int* shape = res_node->inputs[0]->h_meta->shape();
    int ndim = res_node->inputs[0]->h_meta->ndim;
    int size = res_node->inputs[0]->d_content->size();
    
    int last_dim = shape[ndim - 1];
    int num_segments = size / last_dim;
    
    // build offsets
    
    Block* b_ptr;
    MemoryPool::call().allocate(&b_ptr, (num_segments + 1) * sizeof(int), BlockKind::MetaData);
    int* offsets = b_ptr->meta_ptr;
    // fill offsets: 0, last_dim, 2*last_dim, ...
    int gridDim = (num_segments + 1 + 255) / 256;
    int blockDim = 256;

    kernels::reduction_offsets<<<gridDim, blockDim>>>(offsets, last_dim, num_segments);
    cudaDeviceSynchronize();

    Block* temp_storage_block;
    size_t temp_storage_bytes = 0;
    
    // first call to get temp storage size
    cub::DeviceSegmentedReduce::Sum(
        nullptr, temp_storage_bytes,
        input_data, res_data,
        num_segments,
        offsets, offsets + 1
    );
    
    MemoryPool::call().allocate(&temp_storage_block, temp_storage_bytes);
    float* temp_storage = temp_storage_block->d_ptr; 

    // second call to actually reduce
    cub::DeviceSegmentedReduce::Sum(
        temp_storage, temp_storage_bytes,
        input_data, res_data,
        num_segments,
        offsets, offsets + 1
    );
    
    MemoryPool::call().free(b_ptr);
    MemoryPool::call().free(temp_storage_block);
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

void contiguous(InternalNode* res_node)
{
    ContiguousInfo info = broadcast::contiguous(res_node);
    float* res_data = res_node->get_values();
    float* data = res_node->inputs[0]->get_values();
    int size = res_node->d_content->size();
    
    int gridDim = (size + 255) / 256;
    int blockDim = 256;

    kernels::contiguous<<<gridDim, blockDim>>>(data, res_data, size, info);
    cudaDeviceSynchronize();
}
}

// __global__ void contiguous(float* src, float* dst, int* shape, int* strides, int ndim, int size);
