#pragma once

#include <alloca.h>
#include <cstddef>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

struct Grid;

enum struct OutStatus {BlockFound, BlockNotFound, Success, Error};

struct Block 
{
    void* d_ptr = nullptr;
    size_t size = 0;
    Block* next_free = nullptr;
    Grid* mom = nullptr;
    OutStatus allocate_block(void);
    OutStatus free_block(void);
};

struct Grid
{
    int max_blocks_num;
    int blocks_in_use = 0;
    bool is_free = true;
    Block* next_free_block = nullptr;
    Grid* next_free_grid = nullptr;
    Block blocks[0];

    static Grid* init(int max_blocks_num, int size_per_block);

    OutStatus get_free(Block** dst);

    void free_block(Block* block);

    OutStatus free_grid(void);

    private:
    OutStatus free_single_grid(void);
};


class MemoryPool 
{
    int total_allocations = 0;
    size_t total_bytes_in_use = 0;

public:
    std::vector<Block*> allocated_blocks;

    // access
    static MemoryPool& call();

    // debugging
    std::string stats(void);

    // allocate block on device
    void allocate(Block** dst, size_t bytes);

    // releases block on device
    void free(Block* d_ptr);

    // reset all allocations 
    OutStatus reset(void);

    // prevent copies
    MemoryPool(const MemoryPool&) = delete;
    void operator=(const MemoryPool&) = delete;

private:
    // private constructor
    MemoryPool();

    int pick_grid_tier(size_t bytes);

    Grid* make_connect_grid(int tier);

    // private variables
    Grid* grids[5];
    static constexpr size_t MICRO_BLOCK  = 64;       // 64B
    static constexpr size_t TINY_BLOCK   = 1 << 10;  // 1KB
    static constexpr size_t SMALL_BLOCK  = 1 << 16;  // 64KB
    static constexpr size_t MEDIUM_BLOCK = 1 << 20;  // 1MB
    static constexpr size_t LARGE_BLOCK  = 1 << 24;  // 16MB
    static constexpr size_t TIER_SIZES[5] = {64, 1<<10, 1<<16, 1<<20, 1<<24};
    static constexpr int BLOCKS_PER_TIER[5] = {64, 256, 128, 64, 32};
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
    int _size = -1;

    float* data();
    int size();
    size_t bytes();
    void allocate(int size);
    float* d_to_h();
    void h_to_d(float* h_data, int size);
    void d_to_d(float* d_data, int size);
    bool is_allocated(void);

    ~ContentData()
    {
        if (b_ptr) MemoryPool::call().free(b_ptr);
    }

    private:
    void check(void);
};