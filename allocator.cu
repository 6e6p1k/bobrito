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
    cudaError_t status;
    if (kind==BlockKind::Data) status = cudaMalloc(&d_ptr, size);
    else status = cudaMalloc(&meta_ptr, size);

    if (status)
    {
        return OutStatus::Error;
    }
    return OutStatus::Success;
}

OutStatus Block::free_block(void)
{
    cudaError_t status;
    if (kind==BlockKind::Data) status = cudaFree(d_ptr);
    else status = cudaFree(meta_ptr);

    if (status)
    {
        return OutStatus::Error;
    }
    return OutStatus::Success;
}
// </Block struct>

// <Grid struct>
Grid* Grid::init(int max_blocks_num, int size_per_block, BlockKind kind)
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
        raw->kind = kind;
        // set every block but the last one
        for (int i = 0; i < max_blocks_num - 1; i++) {
            raw[i].next_free = &raw[i+1];
            raw[i].size = size_per_block;
            raw[i].allocate_block();
            raw[i].mom = grid;
            raw[i].kind = kind;
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
    if (next_free_block == nullptr) is_free = false;
    return OutStatus::BlockFound;
}

void Grid::free_block(Block* block)
{
    block->next_free = next_free_block;  // point to current head
    next_free_block = block;             // become new head
    is_free = true;
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

void MemoryPool::free(Block* b_ptr)
{
    b_ptr->mom->free_block(b_ptr);
}

// remove all allocations 
OutStatus MemoryPool::reset(void) 
{
    for (int i = 0; i < 5; ++i)
    {
        if (grids[i]->free_grid() != OutStatus::Success) return OutStatus::Error;
        grids[i] = Grid::init(BLOCKS_PER_TIER[i], TIER_SIZES[i], BLOCK_KIND[i]);
    }
    return OutStatus::Success;

}

void MemoryPool::allocate(Block** dst, size_t bytes, BlockKind kind) {
    int tier = pick_grid_tier(bytes, kind);
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

    // not free grids -> make a new one
    grid = make_connect_grid(tier, kind);
    OutStatus status = grid->get_free(dst);    
}

// private constructor
MemoryPool::MemoryPool() 
{
    grids[0] = Grid::init(256, MICRO_BLOCK, BlockKind::MetaData); // micro block for metadata
    grids[1] = Grid::init(256, TINY_BLOCK);
    grids[2] = Grid::init(128, SMALL_BLOCK);
    grids[3] = Grid::init(64,  MEDIUM_BLOCK);
    grids[4] = Grid::init(32,  LARGE_BLOCK);
} 
    
int MemoryPool::pick_grid_tier(size_t bytes, BlockKind kind) 
{
    if (bytes <= MICRO_BLOCK && kind == BlockKind::MetaData)    
        return 0; // solely for metadata 
    if (bytes <= TINY_BLOCK)   return 1;
    if (bytes <= SMALL_BLOCK)  return 2;
    if (bytes <= MEDIUM_BLOCK) return 3;
    return 4;
}

Grid* MemoryPool::make_connect_grid(int tier, BlockKind kind)
{
    // make grid
    Grid* new_grid = Grid::init(BLOCKS_PER_TIER[tier], TIER_SIZES[tier], kind);

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
void ContentData::allocate_values(int size)
{
    if (!values_b_ptr)
    {
        MemoryPool::call().allocate(&values_b_ptr, size * sizeof(float));
        _size = size;
    }
}

void ContentData::allocate_values(void)
{
    if (!values_b_ptr)
    {
        if (!grad_b_ptr) throw std::logic_error("Grad was never allocated. unknown size.");
        MemoryPool::call().allocate(&values_b_ptr, size() * sizeof(float));
    }
    
}

void ContentData::allocate_grad(int size)
{
    if (!grad_b_ptr)
    {
        MemoryPool::call().allocate(&grad_b_ptr, size * sizeof(float));
        _size = size;
    }
}

void ContentData::allocate_grad(void)
{
    if (!grad_b_ptr)
    {
        if (!values_b_ptr) throw std::logic_error("Values were never allocated. unknown size.");
        MemoryPool::call().allocate(&grad_b_ptr, size() * sizeof(float));
    }
    
}

void ContentData::allocate(int size)
{
    allocate_values(size);
    allocate_grad();
}

float* ContentData::values()
{
    // Returns pointer to values on device.
    check_values();
    return values_b_ptr->d_ptr;
}

float* ContentData::grad()
{
    // Returns pointer to grad on device.
    check_grad();
    return grad_b_ptr->d_ptr;
}

int ContentData::size()
{   
    if (!_size) throw std::logic_error("Nothing was allocated yet. nowhere to get size.");
    else return _size;
}

size_t ContentData::bytes()
{
    // Returns size of grad / values in bytes.
    return size() * sizeof(float);
}

float* ContentData::d_values_to_h()
{
    check_values();
    float* dst = (float*)malloc(values_b_ptr->size);
    cudaMemcpy(dst, values_b_ptr->d_ptr, values_b_ptr->size, cudaMemcpyDeviceToHost);
    return dst;
}

float* ContentData::d_grad_to_h()
{
    check_grad();
    float* dst = (float*)malloc(grad_b_ptr->size);
    cudaMemcpy(dst, grad_b_ptr->d_ptr, grad_b_ptr->size, cudaMemcpyDeviceToHost);
    return dst;
}

void ContentData::h_values_to_d(float* h_data, int size)
{
    size_t bytes = size * sizeof(float);
    if (!values_b_ptr)
    {
        MemoryPool::call().allocate(&values_b_ptr, bytes);
    }
    cudaMemcpy(values_b_ptr->d_ptr, h_data, bytes, cudaMemcpyHostToDevice);
}

void ContentData::h_grad_to_d(float* h_data, int size)
{
    size_t bytes = size * sizeof(float);
    if (!grad_b_ptr)
    {
        MemoryPool::call().allocate(&grad_b_ptr, bytes);
    }
    cudaMemcpy(grad_b_ptr->d_ptr, h_data, bytes, cudaMemcpyHostToDevice);
}

void ContentData::check_values(void)
{
    if (!values_b_ptr) throw std::logic_error("Values were never allocated.");
    if (!(values_b_ptr->d_ptr)) throw std::logic_error("Values is empty.");
    if (!(values_b_ptr->size)) throw std::logic_error("Size is not specified.");
}

void ContentData::check_grad(void)
{
    if (!grad_b_ptr) throw std::logic_error("Grad was never allocated.");
    if (!(grad_b_ptr->d_ptr)) throw std::logic_error("Grad is empty.");
}

bool ContentData::is_grad(void)
{
    if (!grad_b_ptr) return false;
    if (!(grad_b_ptr->d_ptr)) return false;
    return true;
}
// </ContentData>
