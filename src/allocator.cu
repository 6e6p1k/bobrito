#include <cmath>
#include <cstddef>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <stdexcept>
#include "allocator.cuh"


// <Block struct>
OutStatus Block::allocate_block(void)
{
    cudaError_t status = cudaMalloc(&d_ptr, size);
    if (status)
    {
        std::cout << "cudaMalloc FAILED: " << cudaGetErrorString(status) << " size=" << size << std::endl;
        return OutStatus::Error;
    }
    return OutStatus::Success;
}

OutStatus Block::free_block(void)
{
    cudaError_t status = cudaFree(d_ptr);
    if (status)
    {
        return OutStatus::Error;
    }
    return OutStatus::Success;
}
// </Block struct>

// <Grid struct>
Grid* Grid::init(int max_blocks_num, int size_per_block)
{
    size_t total_bytes = sizeof(Grid) + sizeof(Block) * max_blocks_num;

    Grid* grid = (Grid*)malloc(total_bytes);
    if (grid)
    {
        grid->max_blocks_num = max_blocks_num;
        grid->is_free = true;
        grid->next_free_block = nullptr;
        grid->next_free_grid = nullptr;

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

OutStatus Grid::get_free(Block** dst)
{
    if (!is_free) { return OutStatus::BlockNotFound; }
    Block* curr = next_free_block;
    Block* next = next_free_block->next_free;
    *dst = curr;
    next_free_block = next;
    blocks_in_use++;
    if (next_free_block == nullptr) is_free = false;
    return OutStatus::BlockFound;
}

void Grid::free_block(Block* block)
{
    block->next_free = next_free_block;  // point to current head
    next_free_block = block;             // become new head
    is_free = true;
    blocks_in_use--;
}

OutStatus Grid::free_grid(void)
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

OutStatus Grid::free_single_grid(void)
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
// </Grid struct>


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
    b_ptr->mom->free_block(b_ptr);

    // for stats & debugging purposes
    total_allocations--;
    total_bytes_in_use -= b_ptr->size;
}


OutStatus MemoryPool::reset(void) 
{
    // Resets all allocations and allocates new starting pack of grids. 
    for (int i = 0; i < 5; ++i)
    {
        if (grids[i]->free_grid() != OutStatus::Success) return OutStatus::Error;
        grids[i] = Grid::init(BLOCKS_PER_TIER[i], TIER_SIZES[i]);
    }
    return OutStatus::Success;
}

void MemoryPool::allocate(Block** dst, size_t bytes) {
    int tier = pick_grid_tier(bytes);
    Grid* grid = grids[tier];
    while (grid != nullptr)
    {
        if (grid->is_free)
        {
            OutStatus status = grid->get_free(dst);

            // for stats & debugging purposes
            total_allocations++;
            total_bytes_in_use += TIER_SIZES[tier];
            
            return;
        }
        grid = grid->next_free_grid;
    }
    // not free grids -> make a new one
    grid = make_connect_grid(tier);
    OutStatus status = grid->get_free(dst);
    
    // for stats & debugging purposes
    total_allocations++;
    total_bytes_in_use += TIER_SIZES[tier];
}

// private constructor
MemoryPool::MemoryPool() 
{
    grids[0] = Grid::init(256, MICRO_BLOCK);
    grids[1] = Grid::init(256, TINY_BLOCK);
    grids[2] = Grid::init(128, SMALL_BLOCK);
    grids[3] = Grid::init(64,  MEDIUM_BLOCK);
    grids[4] = Grid::init(32,  LARGE_BLOCK);
} 
    
int MemoryPool::pick_grid_tier(size_t bytes)
{
    if (bytes <= MICRO_BLOCK)  return 0;
    if (bytes <= TINY_BLOCK)   return 1;
    if (bytes <= SMALL_BLOCK)  return 2;
    if (bytes <= MEDIUM_BLOCK) return 3;
    return 4;
}

Grid* MemoryPool::make_connect_grid(int tier)
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
constexpr size_t MemoryPool::TIER_SIZES[];
constexpr int MemoryPool::BLOCKS_PER_TIER[];
// </Memory pool class>


// <ContentData>
void ContentData::allocate(int size)
{
    if (!b_ptr)
    {
        MemoryPool::call().allocate(&b_ptr, size * sizeof(float));
        _size = size;
    }
}

float* ContentData::data()
{
    check();
    return (float*)b_ptr->d_ptr;
}

int ContentData::size()
{
    if (_size == -1) throw std::logic_error("Nothing was allocated yet. nowhere to get size.");
    else return _size;
}

size_t ContentData::bytes()
{
    return size() * sizeof(float);
}

float* ContentData::d_to_h()
{
    check();
    float* dst = (float*)malloc(b_ptr->size);
    cudaMemcpy(dst, b_ptr->d_ptr, b_ptr->size, cudaMemcpyDeviceToHost);
    return dst;
}

void ContentData::h_to_d(float* h_data, int size)
{
    size_t bytes = size * sizeof(float);
    if (!b_ptr)
    {
        MemoryPool::call().allocate(&b_ptr, bytes);
    }
    cudaMemcpy(b_ptr->d_ptr, h_data, bytes, cudaMemcpyHostToDevice);
}

void ContentData::check(void)
{
    if (!b_ptr) throw std::logic_error("ContentData was never allocated.");
    if (!(b_ptr->d_ptr)) throw std::logic_error("ContentData is empty.");
    if (!(b_ptr->size)) throw std::logic_error("Size is not specified.");
}

bool ContentData::is_allocated(void)
{
    if (!b_ptr) { std::cout<<"no block ptr\n"; return false; }
    if (!(b_ptr->d_ptr)) { std::cout<<"block but no data\n"; return false; }
    return true;
}
// </ContentData>
