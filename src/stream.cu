#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdexcept>
#include <string>
#include <iostream>

#include "stream.h"


cudaStream_t bobr::current_stream() {
    static cudaStream_t s = [] {
        cudaStream_t tmp;
        cudaError_t err = cudaStreamCreateWithFlags(&tmp, cudaStreamNonBlocking);    
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("Error creating a stream: ") + cudaGetErrorString(err)
            );
        }
        return tmp;
    }();
    return s;
}

void bobr::synchronize() {
    cudaError_t err = cudaStreamSynchronize(current_stream());
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("Error syncing a stream: ") + cudaGetErrorString(err)
        );
    }
    return;
}

void CapturedStep::capture(const std::function<void()>& body) {
    // every launch on s is recorded 
    cudaGraph_t graph = nullptr;
    cudaError_t err;
    
    // BEGIN RECORDING
    err = cudaStreamBeginCapture(bobr::current_stream(), cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) throw std::runtime_error(
        std::string("Failed to begin CUDA Graph capture: ") + cudaGetErrorString(err)
    );

    // RECORD (if body throws, catch it and stop the recording before throwing) 
    try { body(); } 
    catch (...) {
        // end the capture so the stream is usable again, then drop whatever was recorded
        cudaStreamEndCapture(bobr::current_stream(), &graph);
        if (graph) cudaGraphDestroy(graph);
        throw;
    }

    // STOP THE RECORDING (if body doesn't throw)
    err = cudaStreamEndCapture(bobr::current_stream(), &graph);
    if (err != cudaSuccess) throw std::runtime_error(
        std::string("Graph was not captured, error ending the capture: ") + cudaGetErrorString(err)
    );

    // WRITE EXEC & destroy the graph
    cudaGraphExec_t tmp;
    tmp = exec_;
    exec_ = nullptr;
    err = cudaGraphInstantiate(&exec_, graph, nullptr, nullptr, 0);
    if (err != cudaSuccess) {
        cudaGraphDestroy(graph);
        exec_ = tmp;
        throw std::runtime_error(std::string("Failed instantiating the graph, previous version is restored: ") + cudaGetErrorString(err));
    }
    if (tmp) cudaGraphExecDestroy(tmp);
    cudaGraphDestroy(graph);

}

void CapturedStep::replay() {
    cudaError_t err = cudaGraphLaunch(exec_, bobr::current_stream());
    if (err != cudaSuccess) throw std::runtime_error(
        std::string("CUDA Graph launch failed: ") + cudaGetErrorString(err)
    );
}

CapturedStep::~CapturedStep() {
    if (exec_) cudaGraphExecDestroy(exec_);
}