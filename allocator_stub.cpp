// Stub for testing without CUDA - provides MemoryPool and ContentData symbols
#include "allocator.cuh"

MemoryPool& MemoryPool::call() {
    static MemoryPool* dummy = nullptr;
    return *dummy;
}

MemoryPool::MemoryPool() {}

void MemoryPool::allocate(Block** dst, size_t bytes) {}
void MemoryPool::free(Block* d_ptr) {}
OutStatus MemoryPool::reset() { return OutStatus::Success; }
int MemoryPool::pick_grid_tier(size_t bytes) { return 0; }
Grid* MemoryPool::make_connect_grid(int tier) { return nullptr; }

// ContentData stubs
float* ContentData::values() { return nullptr; }
float* ContentData::grad() { return nullptr; }
size_t ContentData::size() { return 0; }
float* ContentData::d_values_to_h() { return nullptr; }
float* ContentData::d_grad_to_h() { return nullptr; }
void ContentData::h_values_to_d(float*, int) {}
void ContentData::h_grad_to_d(float*, int) {}
void ContentData::check_values() {}
void ContentData::check_grad() {}