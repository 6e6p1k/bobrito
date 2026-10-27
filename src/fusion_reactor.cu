#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>

#include "kernels.cuh"
#include "fusion_reactor.cuh"

#define NVRTC_CHECK(x) do { nvrtcResult r = (x); \
    if (r != NVRTC_SUCCESS) { std::cerr << "NVRTC err @" << __LINE__ << ": " << nvrtcGetErrorString(r) << "\n"; std::exit(1); } } while(0)
#define CU_CHECK(x) do { CUresult r = (x); \
    if (r != CUDA_SUCCESS) { const char* m; cuGetErrorString(r, &m); std::cerr << "CU err @" << __LINE__ << ": " << m << "\n"; std::exit(1); } } while(0)
#define CUDA_CHECK(x) do { cudaError_t r = (x); \
    if (r != cudaSuccess) { std::cerr << "CUDA err @" << __LINE__ << ": " << cudaGetErrorString(r) << "\n"; std::exit(1); } } while(0)
#ifndef BOBR_SRC_DIR
#define BOBR_SRC_DIR "src"
#endif
#define NVRTC_INCLUDE_PATH "-I" BOBR_SRC_DIR

// Dispatches fusion ops
// Handles nvrtc workflow
FusionReactor& FusionReactor::call() {
    static FusionReactor instance;
    return instance;
}

CUfunction FusionReactor::make_kernel(const std::string& src) {
    auto it = cache.find(src);
    if (it != cache.end()) return it->second.func;

    auto prog = make_compile_prog(src, "kernel_x");
    auto cubin = prog_to_cubin(prog);
    Compiled compiled = cubin_to_compiled_struct(cubin);
    update_cache(src, compiled);
    return compiled.func;
}

void FusionReactor::launch_kernel(CUfunction func, void** args, int n) {
    CU_CHECK(
        cuLaunchKernel(
            func, 
            GRID_SIZE(n),1,1, 
            BLOCK_SIZE,1,1, 
            0, nullptr, args, nullptr
        ));
}

FusionReactor::FusionReactor() {
    nvrtc_init();
}

void FusionReactor::nvrtc_init() {
    // initiates nvrtc, called ones on startup
    CU_CHECK(cuInit(0));
    CU_CHECK(cuDeviceGet(&device, 0));
    CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, device));
    CU_CHECK(cuCtxSetCurrent(ctx));
}        

void FusionReactor::log_check(nvrtcProgram prog, nvrtcResult cr) {
    size_t log_size; NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &log_size));
    if (log_size > 1) {
        std::string log(log_size, 0);
        NVRTC_CHECK(nvrtcGetProgramLog(prog, log.data()));
        std::cerr << "NVRTC log:\n" << log << "\n";
    }
    if (cr != NVRTC_SUCCESS) std::exit(1);
}

std::string FusionReactor::prog_to_cubin(nvrtcProgram prog) {
    size_t cubin_size; NVRTC_CHECK(nvrtcGetCUBINSize(prog, &cubin_size));
    std::string cubin(cubin_size, 0);
    NVRTC_CHECK(nvrtcGetCUBIN(prog, cubin.data()));
    NVRTC_CHECK(nvrtcDestroyProgram(&prog));
    return cubin;
}

Compiled FusionReactor::cubin_to_compiled_struct(std::string cubin) {
    Compiled c;
    CU_CHECK(cuModuleLoadData(&c.module, cubin.c_str()));
    CU_CHECK(cuModuleGetFunction(&c.func, c.module, "fused_k"));
    return c;
}

void FusionReactor::update_cache(std::string kernel_code, Compiled comp) {
    cache[kernel_code] = comp;
}

nvrtcProgram FusionReactor::make_compile_prog(std::string kernel_code, char* name) {
#ifndef NVRTC_ARCH_FLAG
#define NVRTC_ARCH_FLAG "--gpu-architecture=sm_120"
#endif
    static const char* opts[] = {NVRTC_ARCH_FLAG, NVRTC_INCLUDE_PATH, "-std=c++17"};
    nvrtcProgram prog;
    NVRTC_CHECK(
        nvrtcCreateProgram(
            &prog, kernel_code.c_str(), name, 
            0, nullptr, nullptr
    ));
    nvrtcResult cr = nvrtcCompileProgram(prog, 3, opts);
    log_check(prog, cr);

    return prog;
}