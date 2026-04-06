#include "../node.h"
#include "../utils.h"
#include <cub/cub.cuh>
#include <memory>
#include "../allocator.cuh"
#include "../node_helpers.h"



__global__ void reduction_offsets(int* offsets, int last_dim, int num_segments) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= num_segments) offsets[i] = i * last_dim;
}

void compute_offset(int* dst, int last_dim, int num_segments)
{
    int gridDim = (num_segments + 1 + 255) / 256;
    int blockDim = 256;
    reduction_offsets<<<gridDim, blockDim>>>(dst, last_dim, num_segments);
}


template <typename ReduceOp>
void reduce_last_dim(InternalNode* res_node) {
    float* res_data = res_node->get_data();
    float* input_data = res_node->inputs[0]->get_data();
    int* shape = res_node->inputs[0]->h_meta->shape();
    int ndim = res_node->inputs[0]->h_meta->ndim;
    int size = res_node->inputs[0]->h_meta->size;
    
    int last_dim = shape[ndim - 1];
    int num_segments = size / last_dim;
    
    Block* b_ptr;
    MemoryPool::call().allocate(&b_ptr, (num_segments + 1) * sizeof(int));
    int* offsets = (int*)b_ptr->d_ptr;
    compute_offset(offsets, last_dim, num_segments);

    size_t temp_storage_bytes = 0;

    ReduceOp::reduce(nullptr, temp_storage_bytes, input_data, res_data, num_segments, offsets, offsets + 1);

    Block* temp_block;
    MemoryPool::call().allocate(&temp_block, temp_storage_bytes);
    
    ReduceOp::reduce(temp_block->d_ptr, temp_storage_bytes, input_data, res_data, num_segments, offsets, offsets + 1);
    
    MemoryPool::call().free(temp_block);
    MemoryPool::call().free(b_ptr);
    // CUDA_CHECK();
}

struct cubSum {
    static cudaError_t reduce(void* tmp, size_t& tmp_bytes, float* in, float* out, int n, int* begin, int* end) {
        return cub::DeviceSegmentedReduce::Sum(tmp, tmp_bytes, in, out, n, begin, end);
    }
};

struct cubMax {
    static cudaError_t reduce(void* tmp, size_t& tmp_bytes, float* in, float* out, int n, int* begin, int* end) {
        return cub::DeviceSegmentedReduce::Max(tmp, tmp_bytes, in, out, n, begin, end);
    }
};

// CODE ABOVE COMPILED FINE 


void sum_last_dim(InternalNode* res_node)
{
    return reduce_last_dim<cubSum>(res_node);
}


std::shared_ptr<SourceNode> make_A()
{
    int shape[] = {2, 3, 4};
    int ndim = 3;
    float val = 3.22f;
    std::shared_ptr<SourceNode> node = make_filled_source_node(shape, ndim, val, true, "A");   
    return node;
}

std::shared_ptr<SourceNode> make_B()
{
    int shape[] = {2, 4, 5};
    int ndim = 3;
    float val = 1.488f;
    std::shared_ptr<SourceNode> node = make_filled_source_node(shape, ndim, val, true, "B");   
    return node;
}

int main(void) 
{ 
    auto A = make_A();
}