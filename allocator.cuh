#pragma once

#include <cstddef>
#include <iostream>
#include <memory>

struct Grid;

using namespace std;
enum struct OutStatus {BlockFound, BlockNotFound, Success, Error};

struct Block 
{
    float* d_ptr;
    size_t size;
    Block* next_free = nullptr;
    Grid* mom = nullptr;

    OutStatus allocate_block(void);
    OutStatus free_block(void);
};

struct Grid
{
    int max_blocks_num;
    bool is_free = true;
    Block* next_free_block = nullptr;
    Grid* next_free_grid = nullptr;
    Block blocks[0];

    static Grid* init(int max_blocks_num, int size_per_block);

    OutStatus find_free(Block** dst, size_t min_size);

    OutStatus get_free(Block** dst);

    void free_block(Block* block);

    OutStatus free_grid(void);

    private:
    OutStatus free_single_grid(void);
};


class MemoryPool 
{
public:
    // access
    static MemoryPool& call();

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
    Grid* grids[4];
    static constexpr size_t TINY_BLOCK   = 1 << 10;  // 1KB
    static constexpr size_t SMALL_BLOCK  = 1 << 16;  // 64KB
    static constexpr size_t MEDIUM_BLOCK = 1 << 20;  // 1MB
    static constexpr size_t LARGE_BLOCK  = 1 << 24;  // 16MB
    static constexpr size_t TIER_SIZES[4] = {1<<10, 1<<16, 1<<20, 1<<24};
    static constexpr int BLOCKS_PER_TIER[4] = {256, 128, 64, 32};
};

struct ContentData
{
    Block* values_b_ptr = nullptr;
    Block* grad_b_ptr = nullptr;

    float* values();
    float* grad();
    int size();
    size_t bytes();
    void allocate_values(int size);
    void allocate_values(void);
    void allocate_grad(int size);
    void allocate_grad(void);
    void allocate(int size);
    float* d_values_to_h();
    float* d_grad_to_h();
    void h_values_to_d(float* h_data, int size);
    void h_grad_to_d(float* h_data, int size);

    ~ContentData()
    {
        if (values_b_ptr) MemoryPool::call().free(values_b_ptr);
        if (grad_b_ptr) MemoryPool::call().free(grad_b_ptr);
    }

    private:
    void check_values(void);
    void check_grad(void);
};





















