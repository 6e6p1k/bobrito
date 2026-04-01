#include <iostream>
#include <cublas_v2.h>

float* matmul_cuda_launch(
    const float* A, const float* B, int batch, int M, int N, int K)
{
    // created once, reused for every call
    static cublasHandle_t handle = [](){
        cublasHandle_t h;
        cublasCreate(&h);
        cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);

        // allocate 32MB workspace
        void* workspace;
        cudaMalloc(&workspace, 32 * 1024 * 1024);
        cublasSetWorkspace(h, workspace, 32 * 1024 * 1024);

        return h;
    }();

    float* res_ptr;
    cudaMalloc(&res_ptr, batch * M * N * sizeof(float));

    float alpha = 1.0f, beta = 0.0f;

    cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_32F, N, K * N,
        A, CUDA_R_32F, K, M * K,
        &beta,
        res_ptr, CUDA_R_32F, N, M * N,
        batch,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    return res_ptr;
}
