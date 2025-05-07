/*
CUDA kernels and a launcher for gradient clipping using the global L2 norm
In gradient clipping (especially for large-scale deep learning), we sometimes clip all gradients together by computing their global L2 norm:
    global_norm = sqrt(sum(grad^2))
*/


#include "utils/2_cuda_common_utils.h"
#include "utils/7_cuda_utils.cuh"
#include <assert.h>
#include <stddef.h>
#include <cuda_runtime_api.h>

/*
device function for:
    Each thread iterates through a strided subset of data, squaring and accumulating elements:
    Then performs a block-wide reduction using blockReduce<warpReduceSum>()
*/
template<class T>
__device__ float global_norm_squared_for_range(const T* data, size_t count) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_width = blockDim.x * gridDim.x;
    float accumulator = 0.f;
    for(size_t i = index; i < count; i += grid_width) {
        accumulator += (float)data[i] * (float)data[i]; // squaring
    }
    return blockReduce<warpReduceSum>(accumulator); // blockwide reduction
}

// First kernel where each CUDA block computes the sum of squares of a slice of data, and stores the partial result into the out[] array
template<class T>
__global__ void global_norm_squared_kernel(float* out, const T* data, size_t count, ptrdiff_t stride) {
    float block_sum = global_norm_squared_for_range(data + blockIdx.y * stride, count);
    // Processes multiple "slices" (e.g., for different gradients)
    // For each slice (blockIdx.y), launches gridDim.x blocks to process the full range
    // Each block computes partial squared sums, then:
    if(threadIdx.x == 0) {
        size_t out_index = blockIdx.y * gridDim.x + blockIdx.x;
        out[out_index] = out[out_index] + block_sum;
    }
}


/*
Aggregates all partial block sums from global_norm_squared_kernel() and Writes the final result (total L2 norm squared) into out[0].
*/
__global__ void global_norm_aggregate_kernel(float* out, size_t grid_size) {
    size_t index = threadIdx.x;
    //Each thread loads one value from the out[] buffer (up to grid_size total)
    float block_sum = (index < grid_size) ? out[index] : 0.f;
    float sum = blockReduce<warpReduceSum>(block_sum);
    if(threadIdx.x == 0) {
        out[0] = sum; 
    }
}

/*
helper function to calculate the maximum total number of CUDA blocks that will be launched across all tensors, so we can:
    Preallocate enough space in the out[] buffer
    Avoid overflows during parallel block reductions
*/
int get_max_num_block_sums(int* num_slices_all, int numel) {
    const int block_size = 512; // Each block has 512 threads (fixed)
    const int grid_size = deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount / block_size; // This is the maximum number of blocks the GPU can launch concurrently
    assert(grid_size > 0);
    int max_num_block_sums = 0;
    // For each gradient tensor, determine how it's sliced (e.g., per layer, or per head)
    for (int i = 0; i < numel; i++) {
        int num_slices = num_slices_all[i];
        const int gx = CEIL_DIV(grid_size, num_slices); // gx is how many blocks per slice
        const int gy = num_slices; // gy is the number of slices
        max_num_block_sums = max(max_num_block_sums, gx * gy); // gx × gy is the total number of blocks used for this tensor
    }
    return max_num_block_sums;
}

/*
kernel launch for globanormalization
*/
template<typename T>
void global_norm_squared(float* out, const T* values, size_t count, ptrdiff_t stride, int num_slices, int max_num_block_sums, bool reset, cudaStream_t stream) {
    const int block_size = 512;
    const int grid_size = deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount / block_size; // Calculate 2D Grid Dimensions
    assert(grid_size > 0); 

    const int gx = CEIL_DIV(grid_size, num_slices);
    const int gy = num_slices;

    assert(gx * gy < 1024);  // Because the next kernel will reduce these outputs in a single block (1024 threads max)

    if (reset) {
        cudaCheck(cudaMemsetAsync(out, 0, max_num_block_sums * sizeof(float), stream));
    }
    global_norm_squared_kernel<<<dim3(gx, gy), block_size, 0, stream>>>(out, values, count, stride);
    cudaCheck(cudaGetLastError());
}