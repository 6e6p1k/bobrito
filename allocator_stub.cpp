// Stub for testing without CUDA - provides MemoryPool symbols
#include "allocator.cuh"

MemoryPool& MemoryPool::call() {
    static MemoryPool* dummy = nullptr;
    // Should never be called in tests
    return *dummy;
}

MemoryPool::MemoryPool() {}

void MemoryPool::data_to_device(Block** d_ptr, float* h_data, int size) {}
void MemoryPool::free(Block* d_ptr) {}
OutStatus MemoryPool::reset() { return OutStatus::Success; }
void MemoryPool::allocate(Block** dst, size_t bytes) {}
int MemoryPool::pick_grid_tier(size_t bytes) { return 0; }
Grid* MemoryPool::make_connect_grid(int tier) { return nullptr; }
