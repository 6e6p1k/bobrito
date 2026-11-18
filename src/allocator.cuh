#pragma once

#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <string>
#include <unordered_set>

struct Block
{
    void* d_ptr = nullptr;
    size_t size = 0;
    void allocate_block(void);
    void free_block(void);
};

class MemoryPool
{
    cudaMemPool_t pool;
    int total_allocations = 0;
    size_t total_bytes_in_use = 0;

public:
    std::unordered_set<Block*> allocated_blocks;

    // access
    static MemoryPool& call();

    // debugging
    std::string stats(void);

    // allocate block on device
    void allocate(Block** dst, size_t bytes);

    // releases block on device
    void free(Block* d_ptr);

    // release unused cached memory
    void empty_cache(void);

    // prevent copies
    MemoryPool(const MemoryPool&) = delete;
    void operator=(const MemoryPool&) = delete;

private:
    // private constructor
    MemoryPool();
};

// ===============    MEMORY UTILITIES   ==================
struct MemSnapshot {
    size_t cuda_free;
    size_t cuda_total;
    std::string pool_stats;
};
MemSnapshot snapshot();
std::string make_snapshot_string(const std::string& label, const MemSnapshot& s);
void print_snapshot(const std::string& label, const MemSnapshot& s);
void print_allocated_blocks();
std::string print_block(const Block* block);

// ===============    CONTENT DATA   ==================
struct ContentData
{
    Block* b_ptr = nullptr;
    size_t _size = 0; // element count; 0 = nothing allocated

    float* data();
    size_t size();
    size_t bytes();
    void allocate(size_t size);
    float* d_to_h();
    void h_to_d(float* h_data, size_t size);
    void d_to_d(float* d_data, size_t size);
    bool is_allocated(void);

    ~ContentData()
    {
        if (b_ptr != nullptr) {
            try {
                MemoryPool::call().free(b_ptr);
            } catch (const std::exception& e) {
                std::cerr << "failed to free a block: " << e.what() << std::endl;
            }
        }
    }

    ContentData() = default;

    // make it non-copyable
    ContentData(const ContentData&) = delete;
    ContentData& operator=(const ContentData&) = delete;

    ContentData(ContentData&& other) noexcept
        : b_ptr(other.b_ptr), _size(other._size)
    {
        other.b_ptr = nullptr;
        other._size = 0;
    }

    ContentData& operator=(ContentData&& other) noexcept
    {
        ContentData tmp(std::move(other));
        std::swap(b_ptr, tmp.b_ptr);
        std::swap(_size, tmp._size);
        return *this;
    }

    private:
    void check(void);
};
