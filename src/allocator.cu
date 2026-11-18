#include <cstddef>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <cstdint>

#include "allocator.cuh"
#include "stream.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// <Block struct>
void Block::allocate_block(void) {
    cudaError_t err = cudaMallocAsync(&d_ptr, size, bobr::current_stream());
    if (err != cudaSuccess) throw std::runtime_error(
        std::string("Error allocating a block: ") + cudaGetErrorString(err)
    );
}

void Block::free_block(void) {
    cudaError_t err = cudaFreeAsync(d_ptr, bobr::current_stream());
    switch (err) {
        case cudaSuccess: return;
        case cudaErrorCudartUnloading: return;

        default: throw std::runtime_error(
            std::string("Error freeing a block: ") + cudaGetErrorString(err)
        );
    }
}
// </Block struct>

// <Memory pool class>
MemoryPool& MemoryPool::call() {
    static MemoryPool instance;
    return instance;
}

// in MemoryPool
std::string MemoryPool::stats() {
    return "used: " + std::to_string(total_allocations) + ", bytes used: " + std::to_string(total_bytes_in_use);
}


void MemoryPool::free(Block* b_ptr)
{
    b_ptr->free_block();

    // <statistics>
    allocated_blocks.erase(b_ptr);
    total_allocations--;
    total_bytes_in_use -= b_ptr->size;
    // </statistics>

    delete b_ptr;
}


void MemoryPool::empty_cache(void)
{
    cudaError_t err = cudaMemPoolTrimTo(pool, 0);
    if (err != cudaSuccess) throw std::runtime_error(
        std::string("Failed to empty cache: ") + cudaGetErrorString(err)
    );
}

void MemoryPool::allocate(Block** dst, size_t bytes) {
    Block* b = new Block();
    b->size = bytes;
    b->allocate_block();
    #ifdef BOBR_ALLOC_DEBUG
        cudaMemsetAsync(b->d_ptr, 0xFF, bytes, bobr::current_stream());
    #endif
    *dst = b;

    // for stats & debugging purposes
    total_allocations++;
    total_bytes_in_use += bytes;
    allocated_blocks.insert(b);
}

// private constructor
MemoryPool::MemoryPool()
{
    cudaDeviceGetDefaultMemPool(&pool, 0);
    uint64_t keep_all = UINT64_MAX;
    cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &keep_all);
}

MemSnapshot snapshot() {
    cudaDeviceSynchronize();
    MemSnapshot s;
    cudaMemGetInfo(&s.cuda_free, &s.cuda_total);
    s.pool_stats = MemoryPool::call().stats();
    return s;
}

std::string print_block(const Block* block) {
    std::ostringstream oss;
    oss << "\tBlock ptr: " << block << "\n"
        << "\tBlock size: " << block->size << " bytes"
        << " (" << block->size / sizeof(float) << " floats" << ")" << "\n"
        << "\tData ptr: " << block->d_ptr << "\n"
        << "\tData (first 12 floats): ";
        int display_size = (int)std::min(block->size / sizeof(float), (size_t)12);
        float tmp[display_size];
        cudaStreamSynchronize(bobr::current_stream());
        cudaMemcpy(tmp, block->d_ptr, std::min(block->size, display_size * sizeof(float)), cudaMemcpyDeviceToHost);
        oss << "[";
        for (int i = 0; i < display_size; i++) {
            oss << std::fixed << std::setprecision(6) << tmp[i] << " ";
        }
        oss << "]\n";
    return oss.str();
}

std::string allocated_blocks_data_string(const std::unordered_set<Block*>& blocks) {
    std::ostringstream oss;
    oss << "\nAllocated blocks ======================\n";
    bool first = true;
    for (const Block* block : blocks) {
        if (!first) oss << '\n';
        first = false;
        oss << print_block(block);
    }
    oss << "======================================\n";
    return oss.str();
}

std::string make_snapshot_string(const std::string& label, const MemSnapshot& s) {
    size_t used_mb  = (s.cuda_total - s.cuda_free) / (1024 * 1024);
    size_t free_mb  = s.cuda_free  / (1024 * 1024);
    size_t total_mb = s.cuda_total / (1024 * 1024);
    std::ostringstream oss;
    oss << "[" << label << "]\n";
    oss << "  CUDA: " << used_mb << " MB used / " << total_mb << " MB total"
        << " (" << free_mb << " MB free)\n";
    oss << "  Pool: " << s.pool_stats << "\n\n";
    return oss.str();
}

void print_snapshot(const std::string& label, const MemSnapshot& s) {
    std::cout << make_snapshot_string(label, s);
}

void print_allocated_blocks() {
    const auto& blocks = MemoryPool::call().allocated_blocks;
    std::cout << allocated_blocks_data_string(blocks);
}
// </Memory pool class>


// <ContentData>
void ContentData::allocate(size_t size)
{
    if (!b_ptr) {
        MemoryPool::call().allocate(&b_ptr, size * sizeof(float));
        _size = size;
    } else {
        throw std::logic_error("Block is already allocated!");
    }
}

float* ContentData::data()
{
    check();
    return (float*)b_ptr->d_ptr;
}

size_t ContentData::size()
{
    if (_size == 0) throw std::logic_error("Nothing was allocated yet. nowhere to get size.");
    return _size;
}

size_t ContentData::bytes()
{
    return size() * sizeof(float);
}

float* ContentData::d_to_h()
{
    check();
    float* dst = (float*)malloc(_size * sizeof(float));
    gpuErrchk( cudaMemcpyAsync(dst, b_ptr->d_ptr, _size * sizeof(float),
    cudaMemcpyDeviceToHost, bobr::current_stream()
    ) );
    bobr::synchronize();
    return dst;
}

void ContentData::h_to_d(float* h_data, size_t size)
{
    if (!b_ptr) allocate(size);
    check();
    size_t bytes = size * sizeof(float);
    if (bytes > b_ptr->size) throw std::length_error("h_to_d: " + std::to_string(bytes) + " bytes do not fit a " + std::to_string(b_ptr->size) + "-byte block");
    gpuErrchk( cudaMemcpyAsync(b_ptr->d_ptr, h_data, bytes,
        cudaMemcpyHostToDevice, bobr::current_stream()) );
}

void ContentData::d_to_d(float* d_data, size_t size)
{
    if (!b_ptr) allocate(size);
    check();
    size_t bytes = size * sizeof(float);
    if (bytes > b_ptr->size) throw std::length_error("d_to_d: " + std::to_string(bytes) + " bytes do not fit a " + std::to_string(b_ptr->size) + "-byte block");
    gpuErrchk( cudaMemcpyAsync(b_ptr->d_ptr, d_data, bytes,
         cudaMemcpyDeviceToDevice, bobr::current_stream()) );
}

void ContentData::check(void)
{
    if (!b_ptr) throw std::logic_error("ContentData was never allocated.");
    if (!(b_ptr->d_ptr)) throw std::logic_error("ContentData is empty.");
    if (!(b_ptr->size)) throw std::logic_error("Size is not specified.");
}

bool ContentData::is_allocated(void)
{
    if (b_ptr == nullptr) return false;
    return true;
}
// </ContentData>
