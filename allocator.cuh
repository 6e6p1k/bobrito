#pragma once

#include <alloca.h>
#include <cstddef>
#include <iostream>
#include <memory>

struct Grid;

enum struct OutStatus {BlockFound, BlockNotFound, Success, Error};
enum struct BlockKind {Data, MetaData};

struct Block 
{
    float* d_ptr;
    int* meta_ptr;
    size_t size;
    BlockKind kind = BlockKind::Data;
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

    static Grid* init(int max_blocks_num, int size_per_block, BlockKind kind = BlockKind::Data);

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
    void allocate(Block** dst, size_t bytes, BlockKind kind = BlockKind::Data);

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

    int pick_grid_tier(size_t bytes, BlockKind kind);

    Grid* make_connect_grid(int tier, BlockKind kind);

    // private variables
    Grid* grids[5];
    static constexpr size_t MICRO_BLOCK  = 64;       // 64B
    static constexpr size_t TINY_BLOCK   = 1 << 10;  // 1KB
    static constexpr size_t SMALL_BLOCK  = 1 << 16;  // 64KB
    static constexpr size_t MEDIUM_BLOCK = 1 << 20;  // 1MB
    static constexpr size_t LARGE_BLOCK  = 1 << 24;  // 16MB
    static constexpr size_t TIER_SIZES[5] = {64, 1<<10, 1<<16, 1<<20, 1<<24};
    static constexpr int BLOCKS_PER_TIER[5] = {64, 256, 128, 64, 32};
    static constexpr BlockKind BLOCK_KIND[5] = {BlockKind::MetaData, BlockKind::Data, BlockKind::Data, BlockKind::Data, BlockKind::Data};
};

struct ContentData
{
    Block* values_b_ptr = nullptr;
    Block* grad_b_ptr = nullptr;
    int _size;

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
    bool is_grad(void);

    ~ContentData()
    {
        if (values_b_ptr) MemoryPool::call().free(values_b_ptr);
        if (grad_b_ptr) MemoryPool::call().free(grad_b_ptr);
    }

    private:
    void check_values(void);
    void check_grad(void);
};





















