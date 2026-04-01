#pragma once

#include <cstddef>
#include <iostream>
#include <memory>
#include <mutex>

struct Grid;

using namespace std;
enum struct OutStatus {BlockFound, BlockNotFound, Success, Error};

struct Block 
{
    float* d_ptr;
    size_t size;
    int content_size;
    Block* next_free = nullptr;
    Grid* mom = nullptr;

    OutStatus allocate_block(void)
    {
        if (cudaMalloc(&d_ptr, size) != cudaSuccess)
        {
            return OutStatus::Error;
        }
        return OutStatus::Success;
    }

    OutStatus copy_data_to_block(const float* h_data, const size_t size)
    {
        if (cudaMemcpy(d_ptr, h_data, size, cudaMemcpyHostToDevice) != cudaSuccess)
        {
            return OutStatus::Error;
        }
        this->content_size = (int)size;
        return OutStatus::Success;
    }

    OutStatus free_block(void)
    {
        if (cudaFree(d_ptr) != cudaSuccess)
        {
            return OutStatus::Error;
        }
        return OutStatus::Success;
    }
};

struct Grid
{
    int max_blocks_num;
    bool is_free = true;
    Block* next_free_block = nullptr;
    Grid* next_free_grid = nullptr;
    Block blocks[0];

    static Grid* init(int max_blocks_num, int size_per_block)
    {
        // size of the structure itself
        size_t total_bytes = sizeof(Grid) + sizeof(Block) * max_blocks_num;

        Grid* grid = (Grid*)malloc(total_bytes);
        if (grid) 
        {
            grid->max_blocks_num = max_blocks_num;
            
            // init linked list of free blocks
            Block* raw = grid->blocks;
            // set every block but the last one
            for (int i = 0; i < max_blocks_num - 1; i++) {
                raw[i].next_free = &raw[i+1];
                raw[i].size = size_per_block;
                raw[i].allocate_block();
                raw[i].mom = grid;
            }
            // set the last one
            raw[max_blocks_num-1].next_free = nullptr;
            raw[max_blocks_num-1].size = size_per_block;
            raw[max_blocks_num-1].allocate_block();
            raw[max_blocks_num-1].mom = grid;
            // set next free to the first block
            grid->next_free_block = &raw[0];
        }
        return grid;
    }

    OutStatus find_free(Block** dst, size_t min_size)
    {
        Block* prev = nullptr;
        Block* curr = next_free_block;
        
        while (curr != nullptr)
        {
            if (curr->size >= min_size)
            {
                // unlink
                if (prev == nullptr)
                    next_free_block = curr->next_free;  // was head
                else
                    prev->next_free = curr->next_free;  // middle/tail
                
                *dst = curr;
                return OutStatus::BlockFound;
            }
            prev = curr;
            curr = curr->next_free;
        }
        return OutStatus::BlockNotFound;
    }

    OutStatus get_free(Block** dst)
    {
        if (!is_free) { return OutStatus::BlockNotFound; }
        Block* curr = next_free_block;
        Block* next = next_free_block->next_free;
        *dst = curr;
        next_free_block = next;
        if (next_free_block == nullptr) is_free = false;
        return OutStatus::BlockFound;
    }

    void free_block(Block* block)
    {
        block->next_free = next_free_block;  // point to current head
        next_free_block = block;             // become new head
        is_free = true;
    }

    OutStatus free_grid(void)
    {
        Grid* curr = this;
        Grid* next = next_free_grid;
        while (next != nullptr)
        {
            if (curr->free_single_grid() != OutStatus::Success) return OutStatus::Error;
            free(curr);
            curr = next;
            next = next->next_free_grid;
        }
        if (curr->free_single_grid() != OutStatus::Success) return OutStatus::Error;
        free(curr);
        return OutStatus::Success;
    }

    private:
    OutStatus free_single_grid(void)
    {
        for (int i = 0; i < max_blocks_num; i++)
        {
            if (blocks[i].free_block() != OutStatus::Success)
            {
                return OutStatus::Error;
            }
        }
        return OutStatus::Success;
    }
};


class MemoryPool 
{
public:
    // access
    static MemoryPool& call() {
        static MemoryPool instance; 
        return instance;
    }

    // 
    void data_to_device(Block** d_ptr, float* h_data, int size)
    {
        size_t bytes = (size_t)size * sizeof(float);
        allocate(d_ptr, bytes);
        (*d_ptr)->copy_data_to_block(h_data, bytes);
    }

    void free(Block* d_ptr)
    {
        d_ptr->mom->free_block(d_ptr);
    }

    // remove all allocations 
    OutStatus reset(void) 
    {
        for (int i = 0; i < 4; ++i)
        {
            if (grids[i]->free_grid() != OutStatus::Success) return OutStatus::Error;
            grids[i] = Grid::init(BLOCKS_PER_TIER[i], TIER_SIZES[i]);
        }
        return OutStatus::Success;

    }

    // prevent copies
    MemoryPool(const MemoryPool&) = delete;
    void operator=(const MemoryPool&) = delete;

private:
    // private constructor
    MemoryPool() 
    {
        grids[0] = Grid::init(256, TINY_BLOCK);
        grids[1] = Grid::init(128, SMALL_BLOCK);
        grids[2] = Grid::init(64,  MEDIUM_BLOCK);
        grids[3] = Grid::init(32,  LARGE_BLOCK);
    } 

    void allocate(Block** dst, size_t bytes) {
        int tier = pick_grid_tier(bytes);
        Grid* grid = grids[tier];
        while (grid != nullptr)
        {
            if (grid->is_free)
            {
                grid->get_free(dst);
                return;
            }
            grid = grid->next_free_grid;
        }
        grid = make_connect_grid(tier);
        grid->get_free(dst);
    }
    
    int pick_grid_tier(size_t bytes) 
    {
        if (bytes <= TINY_BLOCK)   return 0;
        if (bytes <= SMALL_BLOCK)  return 1;
        if (bytes <= MEDIUM_BLOCK) return 2;
        return 3;
    }

    Grid* make_connect_grid(int tier)
    {
        // make grid
        Grid* new_grid = Grid::init(BLOCKS_PER_TIER[tier], TIER_SIZES[tier]);

        // link new grid into the beginning
        Grid* tmp = grids[tier]->next_free_grid;
        grids[tier]->next_free_grid = new_grid;
        new_grid->next_free_grid = tmp;
        
        // output grid
        return new_grid;
    }

    // private variables
    Grid* grids[4];
    static constexpr size_t TINY_BLOCK   = 1 << 10;  // 1KB
    static constexpr size_t SMALL_BLOCK  = 1 << 16;  // 64KB
    static constexpr size_t MEDIUM_BLOCK = 1 << 20;  // 1MB
    static constexpr size_t LARGE_BLOCK  = 1 << 24;  // 16MB
    static constexpr size_t TIER_SIZES[4] = {1<<10, 1<<16, 1<<20, 1<<24};
    static constexpr int BLOCKS_PER_TIER[4] = {256, 128, 64, 32};
};























