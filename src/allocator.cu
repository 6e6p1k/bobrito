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

#include "allocator.cuh"

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
            if (raw[i].allocate_block() != OutStatus::Success)
                throw std::runtime_error("Grid::init: cudaMalloc failed for a " + std::to_string(size_per_block) + "-byte block");
            raw[i].mom = grid;
        }
        // set the last one
        raw[max_blocks_num-1].next_free = nullptr;
        raw[max_blocks_num-1].size = size_per_block;
        if (raw[max_blocks_num-1].allocate_block() != OutStatus::Success)
            throw std::runtime_error("Grid::init: cudaMalloc failed for a " + std::to_string(size_per_block) + "-byte block");
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
    cudaMemset(b_ptr->d_ptr, 0, b_ptr->size);
    b_ptr->mom->free_block(b_ptr);

    // for stats & debugging purposes
    // std::cout << "\nALLOCATOR. BLOCK FREED ===============\n";
    // std::cout << print_block(b_ptr);
    // std::cout << "========================================\n\n";
    allocated_blocks.erase(std::remove(allocated_blocks.begin(), allocated_blocks.end(), b_ptr), allocated_blocks.end());
    total_allocations--;
    total_bytes_in_use -= b_ptr->size;
}


OutStatus MemoryPool::reset(void) 
{
    // Resets all allocations and allocates new starting pack of grids. 
    for (int i = 0; i < NUM_TIERS; ++i)
    {
        if (grids[i] && grids[i]->free_grid() != OutStatus::Success) return OutStatus::Error;
        grids[i] = i < EAGER_TIERS ? Grid::init(BLOCKS_PER_TIER[i], TIER_SIZES[i]) : nullptr;
    }
    return OutStatus::Success;
}

void MemoryPool::allocate(Block** dst, size_t bytes) {
    int tier = pick_grid_tier(bytes);
    if (grids[tier] == nullptr) grids[tier] = Grid::init(BLOCKS_PER_TIER[tier], TIER_SIZES[tier]);
    Grid* grid = grids[tier];
    while (grid != nullptr)
    {
        if (grid->is_free)
        {
            OutStatus status = grid->get_free(dst);

            // for stats & debugging purposes
            total_allocations++;
            total_bytes_in_use += TIER_SIZES[tier];
            allocated_blocks.push_back(*dst);
            // std::cout << "\nALLOCATOR. BLOCK ALLOCATION =============\n";
            // std::cout << print_block(*dst);
            // std::cout << "========================================\n\n";
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
    allocated_blocks.push_back(*dst);
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
    if (bytes <= LARGE_BLOCK)  return 4;
    if (bytes <= ULTRA_LARGE_BLOCK) return 5;
    throw std::length_error("MemoryPool: " + std::to_string(bytes) + " bytes requested, the largest block is "
                            + std::to_string(ULTRA_LARGE_BLOCK) + " bytes");
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
        int display_size = std::min((int)(block->size / sizeof(float)), 12);
        float tmp[display_size];
        cudaMemcpy(tmp, block->d_ptr, std::min(block->size, display_size * sizeof(float)), cudaMemcpyDeviceToHost);
        oss << "[";
        for (int i = 0; i < display_size; i++) {
            oss << std::fixed << std::setprecision(6) << tmp[i] << " ";
        }
        oss << "]\n"
        << "\tGrid ptr: " << block->mom << "\n";
    return oss.str();
}

std::string allocated_blocks_data_string(const std::vector<Block*>& blocks) {
    std::ostringstream oss;
    oss << "\nAllocated blocks ======================\n";
    for (const auto& block : blocks) {
        oss << print_block(block);
        if (block != blocks.back()) {
            oss << std::endl;
        }
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
    auto blocks = MemoryPool::call().allocated_blocks;
    std::cout << allocated_blocks_data_string(blocks);
}
// </Memory pool class>


// <ContentData>
void ContentData::allocate(int size)
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
    float* dst = (float*)malloc(_size * sizeof(float));
    gpuErrchk( cudaMemcpy(dst, b_ptr->d_ptr, _size * sizeof(float), cudaMemcpyDeviceToHost) );
    return dst;
}

void ContentData::h_to_d(float* h_data, int size)
{
    check();
    size_t bytes = size * sizeof(float);
    if (!b_ptr)
    {
        MemoryPool::call().allocate(&b_ptr, bytes);
    }
    if (bytes > b_ptr->size) throw std::length_error("h_to_d: " + std::to_string(bytes) + " bytes do not fit a " + std::to_string(b_ptr->size) + "-byte block");
    gpuErrchk( cudaMemcpy(b_ptr->d_ptr, h_data, bytes, cudaMemcpyHostToDevice) );
}

void ContentData::d_to_d(float* d_data, int size)
{
    size_t bytes = size * sizeof(float);
    if (!b_ptr)
    {
        MemoryPool::call().allocate(&b_ptr, bytes);
    }
    if (bytes > b_ptr->size) throw std::length_error("d_to_d: " + std::to_string(bytes) + " bytes do not fit a " + std::to_string(b_ptr->size) + "-byte block");
    gpuErrchk( cudaMemcpy(b_ptr->d_ptr, d_data, bytes, cudaMemcpyDeviceToDevice) );
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
