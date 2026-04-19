// nvrtc_hello.cpp
#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <vector>

#include "../src/utils.h"
#include "../src/allocator.cuh"
#include "../src/kernels.cuh"

#define NVRTC_CHECK(x) do { nvrtcResult r = (x); \
    if (r != NVRTC_SUCCESS) { std::cerr << "NVRTC err @" << __LINE__ << ": " << nvrtcGetErrorString(r) << "\n"; std::exit(1); } } while(0)
#define CU_CHECK(x) do { CUresult r = (x); \
    if (r != CUDA_SUCCESS) { const char* m; cuGetErrorString(r, &m); std::cerr << "CU err @" << __LINE__ << ": " << m << "\n"; std::exit(1); } } while(0)
#define CUDA_CHECK(x) do { cudaError_t r = (x); \
    if (r != cudaSuccess) { std::cerr << "CUDA err @" << __LINE__ << ": " << cudaGetErrorString(r) << "\n"; std::exit(1); } } while(0)

#define PATH "-I/home/hz/nothing/bobr-framework-cuda/src/"

const char* kernel_src = R"(
#define ELEMWISE_IDX_IMPL
#include "fused_kernels.cuh"

extern "C" __global__ void fused_k(float* a, float* b, float* out, int n, BroadcastInfo info) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= n) return;
    int a_idx, b_idx;
    elemwise_idx(out_idx, a_idx, b_idx, info);
    out[out_idx] = fmaxf(a[a_idx] + b[b_idx], 0.0f);
}
)";

int main() {
    CU_CHECK(cuInit(0));
    CUdevice device; CU_CHECK(cuDeviceGet(&device, 0));
    CUcontext ctx; CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, device));
    CU_CHECK(cuCtxSetCurrent(ctx));

    nvrtcProgram prog;
    NVRTC_CHECK(nvrtcCreateProgram(&prog, kernel_src, "fused.cu", 0, nullptr, nullptr));

    const char* opts[] = {
        "--gpu-architecture=sm_120",
        "-I/home/hz/nothing/bobr-framework-cuda/src/",  // <-- change this
        "-std=c++17",
    };
    nvrtcResult cr = nvrtcCompileProgram(prog, 3, opts);

    size_t log_size; NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &log_size));
    if (log_size > 1) {
        std::string log(log_size, 0);
        NVRTC_CHECK(nvrtcGetProgramLog(prog, log.data()));
        std::cerr << "NVRTC log:\n" << log << "\n";
    }
    if (cr != NVRTC_SUCCESS) std::exit(1);

    size_t cubin_size; NVRTC_CHECK(nvrtcGetCUBINSize(prog, &cubin_size));
    std::string cubin(cubin_size, 0);
    NVRTC_CHECK(nvrtcGetCUBIN(prog, cubin.data()));
    NVRTC_CHECK(nvrtcDestroyProgram(&prog));

    CUmodule module; CU_CHECK(cuModuleLoadData(&module, cubin.c_str()));
    CUfunction kernel; CU_CHECK(cuModuleGetFunction(&kernel, module, "fused_k"));

    // test: a = [-1, -1, ..., 2, 2, ...], b = [3, 3, ...], no broadcast (1D identity strides)
    const int N = 8;
    std::vector<float> h_a = {-5, -5, -5, -5, 2, 2, 2, 2};
    std::vector<float> h_b(N, 3.0f);
    std::vector<float> h_out(N, 0);
    float *d_a, *d_b, *d_out;

    CUDA_CHECK(cudaMalloc(&d_a, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N*sizeof(float), cudaMemcpyHostToDevice));

    BroadcastInfo info{};
    info.ndim = 1;
    info.out_shape[0] = N;
    info.a_strides[0] = 1;
    info.b_strides[0] = 1;

    int n = N;
    void* args[] = {&d_a, &d_b, &d_out, &n, &info};
    CU_CHECK(cuLaunchKernel(kernel, 1,1,1, N,1,1, 0, nullptr, args, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N*sizeof(float), cudaMemcpyDeviceToHost));
    for (float v : h_out) std::cout << v << " ";  // expect: 0 0 0 0 5 5 5 5
    std::cout << "\n";

    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_out));
    CU_CHECK(cuModuleUnload(module));
    return 0;
}

// 1. once at startup
void nvrtc_init() {
    CU_CHECK(cuInit(0));
    CUdevice device; CU_CHECK(cuDeviceGet(&device, 0));
    CUcontext ctx; CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, device));
    CU_CHECK(cuCtxSetCurrent(ctx));
}

struct Compiled { CUmodule module; CUfunction func; };
std::unordered_map<std::string, Compiled> cache;

CUfunction get_or_compile(const std::string& src) {
    auto it = cache.find(src);
    if (it != cache.end()) return it->second.func;

    // make prog
    nvrtcProgram prog;
    NVRTC_CHECK(nvrtcCreateProgram(&prog, src.c_str(), "fused.cu", 0, nullptr, nullptr));
    const char* opts[] = {"--gpu-architecture=sm_120", PATH, "-std=c++17"};
    nvrtcResult cr = nvrtcCompileProgram(prog, 3, opts);

    // nvrtc check
    size_t log_size; NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &log_size));
    if (log_size > 1) {
        std::string log(log_size, 0);
        NVRTC_CHECK(nvrtcGetProgramLog(prog, log.data()));
        std::cerr << "NVRTC log:\n" << log << "\n";
    }
    if (cr != NVRTC_SUCCESS) std::exit(1);

    // to cubin
    size_t cubin_size; NVRTC_CHECK(nvrtcGetCUBINSize(prog, &cubin_size));
    std::string cubin(cubin_size, 0);
    NVRTC_CHECK(nvrtcGetCUBIN(prog, cubin.data()));
    NVRTC_CHECK(nvrtcDestroyProgram(&prog));

    // load into gpu
    Compiled c;
    CU_CHECK(cuModuleLoadData(&c.module, cubin.c_str()));
    CU_CHECK(cuModuleGetFunction(&c.func, c.module, "fused_k"));
    cache[src] = c;
    return c.func;
}

// 3. launch: fast, called every step
void launch(CUfunction f, void** args, int n) {
    CU_CHECK(cuLaunchKernel(f, 
        GRID_SIZE(n),1,1, 
        BLOCK_SIZE,1,1, 
        0, nullptr, args, nullptr));
}