#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <functional>
#include <vector>

#include "nodes.h"
#include "kernels.cuh"
#include "stream.h"

// ============================================================
//                         NVRTC HANDLER
// ============================================================
#define NVRTC_CHECK(x) do { nvrtcResult r = (x); \
    if (r != NVRTC_SUCCESS) { std::cerr << "NVRTC err @" << __LINE__ << ": " << nvrtcGetErrorString(r) << "\n"; std::exit(1); } } while(0)
#define CU_CHECK(x) do { CUresult r = (x); \
    if (r != CUDA_SUCCESS) { const char* m; cuGetErrorString(r, &m); std::cerr << "CU err @" << __LINE__ << ": " << m << "\n"; std::exit(1); } } while(0)
#define CUDA_CHECK(x) do { cudaError_t r = (x); \
    if (r != cudaSuccess) { std::cerr << "CUDA err @" << __LINE__ << ": " << cudaGetErrorString(r) << "\n"; std::exit(1); } } while(0)
#define PATH "-I/home/hz/nothing/bobr-framework-cuda/src/"

// ========================= INIT ==========================
void nvrtc_init() {
    // initiates nvrtc, called ones on startup
    CU_CHECK(cuInit(0));
    CUdevice device; CU_CHECK(cuDeviceGet(&device, 0));
    CUcontext ctx; CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, device));
    CU_CHECK(cuCtxSetCurrent(ctx));
}

// ==================== COMPILE & CACHE =====================
// called on the first batch (or on .compile())
// if (func in cache) -> retrieve, output 
// else -> compile and cache the func
struct Compiled { CUmodule module; CUfunction func; };


// ======================= KERNEL LAUNCH ====================
// [FAST] launches the kernel from existing function
// called every iteration
void launch(CUfunction f, void** args, int n) {
    CU_CHECK(cuLaunchKernel(f, 
        GRID_SIZE(n),1,1, 
        BLOCK_SIZE,1,1, 
        0, bobr::current_stream(), args, nullptr));
}

// Dispatches fusion ops
// Handles nvrtc workflow
class FusionReactor {

    std::unordered_map<std::string, Compiled> cache;
    CUdevice device;
    CUcontext ctx;

    CUfunction get_or_compile(const std::string& src) {
        auto it = cache.find(src);
        if (it != cache.end()) return it->second.func;
    
        // make prog
        nvrtcProgram prog;
        NVRTC_CHECK(
            nvrtcCreateProgram(
                &prog, src.c_str(), "fused.cu", 
                0, nullptr, nullptr
            ));

        const char* opts[] = {"--gpu-architecture=sm_120", PATH, "-std=c++17"};
        nvrtcResult cr = nvrtcCompileProgram(prog, 3, opts);
        log_check(prog, cr);

        // to cubin
        auto cubin = prog_to_cubin(prog);
    
        // load into gpu
        Compiled c;
        CU_CHECK(cuModuleLoadData(&c.module, cubin.c_str()));
        CU_CHECK(cuModuleGetFunction(&c.func, c.module, "fused_k"));
        cache[src] = c;
        return c.func;
    }

    private:
    
    FusionReactor() {
        nvrtc_init();
    }

    void nvrtc_init() {
        // initiates nvrtc, called ones on startup
        CU_CHECK(cuInit(0));
        CU_CHECK(cuDeviceGet(&device, 0));
        CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, device));
        CU_CHECK(cuCtxSetCurrent(ctx));
    }        

    void log_check(nvrtcProgram prog, nvrtcResult cr) {
        size_t log_size; NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &log_size));
        if (log_size > 1) {
            std::string log(log_size, 0);
            NVRTC_CHECK(nvrtcGetProgramLog(prog, log.data()));
            std::cerr << "NVRTC log:\n" << log << "\n";
        }
        if (cr != NVRTC_SUCCESS) std::exit(1);
    }

    std::string prog_to_cubin(nvrtcProgram prog) {
        size_t cubin_size; NVRTC_CHECK(nvrtcGetCUBINSize(prog, &cubin_size));
        std::string cubin(cubin_size, 0);
        NVRTC_CHECK(nvrtcGetCUBIN(prog, cubin.data()));
        NVRTC_CHECK(nvrtcDestroyProgram(&prog));
        return cubin;
    }
    

};


// struct FusionRule {
//     std::vector<PrimKind> pattern;   // e.g. {MatMul, Add, Relu}
//     PrimKind replacement;             // e.g. MatMulAddRelu
//     // optional: predicate for shape/dtype constraints
//     std::function<bool(const std::vector<Node*>&)> check;
// };