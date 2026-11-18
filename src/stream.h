#pragma once
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <functional>

namespace bobr {
    cudaStream_t current_stream(); // lazily creates one non-blocking stream on first use
    void synchronize(); // synchronizes the main_stream, trows on error
}

class CapturedStep {
    cudaGraphExec_t exec_ = nullptr;

    public:
    void capture(const std::function<void()>& body);
    void replay();
    ~CapturedStep();
    CapturedStep() = default;
    // Remove copy constructor
    CapturedStep(const CapturedStep& other) = delete;
    CapturedStep& operator=(const CapturedStep&) = delete;

};
