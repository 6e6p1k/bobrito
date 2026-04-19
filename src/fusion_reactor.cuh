#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <functional>
#include <vector>

struct Compiled { 
    CUmodule module; 
    CUfunction func;
 };

// Dispatches fusion ops
// Handles nvrtc workflow
class FusionReactor {
    public:
        static FusionReactor& call();
        CUfunction make_kernel(const std::string& src);
        void launch_kernel(CUfunction func, void** args, int n);
    
    private:
        std::unordered_map<std::string, Compiled> cache;
        CUdevice device;
        CUcontext ctx;

        FusionReactor();
        FusionReactor(const FusionReactor&) = delete;
        FusionReactor& operator=(const FusionReactor&) = delete;

        void nvrtc_init();
        void log_check(nvrtcProgram prog, nvrtcResult cr);
        std::string prog_to_cubin(nvrtcProgram prog);
        Compiled cubin_to_compiled_struct(std::string cubin);
        void update_cache(std::string kernel_code, Compiled comp);
        nvrtcProgram make_compile_prog(std::string kernel_code, char* name);
};