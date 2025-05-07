/*
this file has some necessary utilitie for to be used in CUDA code.
*/

#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <type_traits>      
#include <cuda_runtime.h>
// importing nvtx tools for profiling with Nsight Systems / Compute
#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCudaRt.h>
#include <cuda_profiler_api.h>
// importing CUDA types for half precision and floating point
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "1_utils.h"

// Using Device properties
// defined as extern here because the individual kernels wish to use it
extern cudaDeviceProp deviceProp;

//defining warp size [typical warp sze on NVIDIA on Nvidia GPUs is 32 threads]
#define WARP_SIZE 32U

/* 
SOME CUDA CONCEPTS
-------------------------------------------------------------------------------------------
When you run CUDA kernels on a GPU, the SM executes blocks of threads
Each SM can hold a limited number of threads, registers, shared memory
Ideally you want many thread blocks ready to run on each SM at the same time. => This is called LATENCY HIDING

* Memory access is very slow compared to computation time
* If you have one block running, whenever the block waits for memory, the SM becomes idle
* If you have two or more block ready, when one block stalls(waits for memory), the SM can immediately switch to another block
* Switching is instant inside SMs
* No Overhead (like CPU context switch), this keeps the GPU busy

Why 2 blocks specifically: 
        * Modern GPUs like A100s, have very large SMs, big enough to handle 2 big thread blocks at the same time
        * 2 blocks together tries to make better use of resources
*/

//BELOW CODE TRIES TO FIT 2 BLOCKS per SMs, Since I am running on L40S I will get MAX_1024_THREADS_BLOCKS =1
//If you're running in A100 (SM80) or H100 (SM90+), you will get MAX_1024_THREADS_BLOCKS = 2

#if __CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900
#define MAX_1024_THREADS_BLOCKS 2
#else
#define MAX_1024_THREADS_BLOCKS 1
#endif

// for calcualting grid and block dimensions
// If M is already divisible by N , nothing changes
// If not, it pushes M high enough to make dvivision round up.

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// Define two compile-time constants called True and False
// They are based on `std::bool_constants` from <type traits>
// When writing templated or heavily optimized CUDA kernels (or C++ functions), Sometimes you want to pass a compile-time flag (true/false) as a function argument
// So that the compiler can optimize differently based on the flag
constexpr std::bool_constant<true> True; // compile tie TRUE
constexpr std::bool_constant<true> False; // Compile time FALSE

// CUDA error chekcing, basic for wrapping Cuda calls
inline void cudaCheck_(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR at file %s:%d:\n%s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

// define macro for convinience
#define cudaCheck(err) (cudaCheck_(err, __FILE__, __LINE__))

// function to safely free up GPU memory and resets the pointer to `nullptr`, if fails returns errors
template<class T>
inline void cudaFreeCheck(T** ptr, const char *file, int line) {
    cudaError_t error = cudaFree(*ptr);
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    *ptr = nullptr;
}


#define cudaFreeCheck(ptr) (cudaFreeCheck(ptr, __FILE__, __LINE__))

//CUDA Precision settings 

/* IF YOU DONOT KNOW C
enum in C is a way to group a set of named integer constants together,
so you can refer to meaningful names instead of just using numbers
*/
enum PrecisionMode {
    PRECISION_FP32, // 	32-bit floating point (normal float)
    PRECISION_FP16, // 	16-bit half-precision floating point (half)
    PRECISION_BF16 // 16-bit bfloat16 floating point (__nv_bfloat16)
};

// It choses the floating point type (float, half, or bfloat16) based on compile time macro
//  (ENABLE_FP32, ENABLEFP_16, OR none), also sets a `PRECISION_MODE` where precision was chosen

//CHECK IF ENABLE_FP32 is defined
#if defined(ENABLE_FP32)
typedef float floatX;
#define PRECISION_MODE PRECISION_FP32
//CHECK IF `ENABLE_fp16` is defined
#elif defined(ENABLE_FP16)
typedef half floatX;
#define PRECISION_MODE PRECISION_FP16
// else bfloat16 becomes default
#else 
typedef __nv_bfloat16 floatX;
#define PRECISION_MODE PRECISION_BF16
#endif

// Trying to define `__ldcs` and `__stcs` for bffloat16(__nv_bfloat16), uf the CUDA compier doesnot suppot them

/* SOME CUDA CONCEPTS
------------------------------------------------------------------------------------------
__ldcs = Load from memory using streaming cache hint
__stcs = Store to memory using streaming cache hint

Streaming Cache Hints tells the GPU: 
    "Load /Store this memory with caching optimized for streaming large data"
    which can make memory loads faster and cheaper for large tensor operations (like LLMs)

THis is needed because
    - In CUDA11 (and earlier), nvcc doesnot provide __ldcs or __stcs functions for bfloat16 types
    - In CUDA12 or newer, theses functions are properly available 
So for older CUDA versions, they need to define their own versions manually, but if you define your own when CUDA already provides
it --> compilation error ("function already exists") 

*/
#if defined(ENABLE_BF16) && (__CUDACC_VER_MAJOR__ < 12) && !((__CUDA_ARCH__ >= 800) || !defined(__CUDA_ARCH__))
// Trick the hardware into treating bfloat16 as just a 16-bit short
// Use fast streaming cache memory ops even on old CUDA compilers.
//Custom __ldcs for floatX (which is bf16 here), Reinterpret the pointer as unsigned short* , Load with existing __ldcs for unsigned short. Then wrap it back into a bfloat16.
__device__ floatX __ldcs(const floatX* address) {
    unsigned short bf = __ldcs(reinterpret_cast<const unsigned short*>(address));
    return __nv_bfloat16_raw{bf};
}

// Similarly, Custom __stcs for floatX, Extract the raw 16-bit representation from the __nv_bfloat16, Store it using the normal unsigned short __stcs.
__device__ void __stcs(floatX* address, floatX value) {
    __stcs(reinterpret_cast<unsigned short*>(address), ((__nv_bfloat16_raw)value).x);
}
#endif


// defines a small helper C++ class (NvtxRange)
// to automatically mark regions (create start/end marks) in a CUDA program for profiling
// uses NVTX (NVIDIA Tools Extension library) to create profiling markers —
// you can see these markers in Nsight Systems, Nsight Compute, Visual Profiler, etc.
class NvtxRange {
    public:
       NvtxRange(const char* s) { nvtxRangePush(s); }
       NvtxRange(const std::string& base_str, int number) {
           std::string range_string = base_str + " " + std::to_string(number);
           nvtxRangePush(range_string.c_str());
       }
       ~NvtxRange() { nvtxRangePop(); }
};

// define the macro for the convinience

#define NVTX_RANGE_FN() NvtxRange nvtx_range(__FUNCTION__)

// Utilities to read and write between CUDA Memory <-> files, using double buffering running on the 
//This function copies a blokc of memory from GPU device memory to a disk file
// It uses asynchronous memory transfers (cudaMemcpyAsync) + double buffering (two buffers alternate) + CUDA streams (overlap memory copy and disk write).
// End Goal: Save a large GPU tensor to a file very fast without stalling the GPU unnecessarily
inline void device_to_file(FILE* dest, void* src, size_t num_bytes, size_t buffer_size, cudaStream_t stream) {
    
    // Allocates page-locked (pinned) host memory of size 2 x buffer_size bytes as pinned memory is faster for `cudaMemcpyAsync`
    char* buffer_space;
    cudaCheck(cudaMallocHost(&buffer_space, 2*buffer_size));
    // split the buffer into two halves, while one buffer is reading, the other is writing.
    void* read_buffer = buffer_space; // read_buffer -> where the next GPU-to-CPU copy happens
    void* write_buffer = buffer_space + buffer_size; // write_buffer -> which wil be written to the file meanwhile

    // Prime the first read, i.e. First start by copying the first chunck from GPU memory to CPU pinned buffer, then 
    // synchronize using `cudaStramSynchronize` to make sure the first copy is complete.
    char* gpu_read_ptr = (char*)src;
    size_t copy_amount = std::min(buffer_size, num_bytes);
    cudaCheck(cudaMemcpyAsync(read_buffer, gpu_read_ptr, copy_amount, cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaStreamSynchronize(stream));

    // setup the remaining state, Tracks how many bytes are left to copy (`rest_bytes` ), and how big the last written chunk was (`write_buffer_size`)
    size_t rest_bytes = num_bytes - copy_amount;
    size_t write_buffer_size = copy_amount;
    gpu_read_ptr += copy_amount; // moves the GPU memory pointer forward

    std::swap(read_buffer, write_buffer);
    // whilwe there are still bytes left 
    while(rest_bytes > 0) {
        copy_amount = std::min(buffer_size, rest_bytes);
        //start async copy
        cudaCheck(cudaMemcpyAsync(read_buffer, gpu_read_ptr, copy_amount, cudaMemcpyDeviceToHost, stream));
        //while copying is happening , write the old buffer to file
        fwriteCheck(write_buffer, 1, write_buffer_size, dest);
        // Synchronize to make sure both buffer transfers complete
        cudaCheck(cudaStreamSynchronize(stream));
        // Swap buffers, necxt cycle , the read and write buffer switch roles
        // GPU memory -> CPU copy
        // CPU memory -> Disk Write
        std::swap(read_buffer, write_buffer);

        rest_bytes -= copy_amount;
        write_buffer_size = copy_amount;
        gpu_read_ptr += copy_amount;
    }

    // After the loop ends write the remaining data still left in `write buffer`
    fwriteCheck(write_buffer, 1, write_buffer_size, dest);

    //At the end free the pinned memory
    cudaCheck(cudaFreeHost(buffer_space));

}


// Function to file to device
// This function loads a block of data from a file on diskk into GPU device memory efficiently
// Again it follows the same principles like above
// Pinned host memory, Asynchronous Copy, Double buffering (alternating between two buffers), CUDA streams for overlapping operations
// End Goal: Load large file into GPU memory as fast as possible without stalling the GPU
inline void file_to_device(void* dest, FILE* src, size_t num_bytes, size_t buffer_size, cudaStream_t stream) {
    // Allocate 2 x buffer bytes of pinened (page-locked) host memory
    // Uses `cudaHostAllocWriteCombined`: This makes host memory optimized for write by CPU and read by GPU, imporves performance on Host-to-Device transfers
    char* buffer_space;
    cudaCheck(cudaMallocHost(&buffer_space, 2*buffer_size, cudaHostAllocWriteCombined));

    // Split the buffer into two halves as earlier to setup double buffering
    void* read_buffer = buffer_space;
    void* write_buffer = buffer_space + buffer_size;

    // Prime the first read; Reads the first chunk from disk into `read_buffer` synchronously.
    char* gpu_write_ptr = (char*)dest;
    size_t copy_amount = std::min(buffer_size, num_bytes);
    freadCheck(read_buffer, 1, copy_amount, src); // No asynchronous read from disk, as fread is a CPU blocking operations

    // remaining bytes left to read
    size_t rest_bytes = num_bytes - copy_amount;
    size_t write_buffer_size = copy_amount; // size of the chunk to copy next
    std::swap(read_buffer, write_buffer); // swap buffers so next cycle can happen cleanly

    //main loop
    while(rest_bytes > 0) {
        //start asynch copy from write_buffer to GPU memory
        copy_amount = std::min(buffer_size, rest_bytes);
        cudaCheck(cudaMemcpyAsync(gpu_write_ptr, write_buffer, write_buffer_size, cudaMemcpyHostToDevice, stream));
        gpu_write_ptr += write_buffer_size;
        //while async copy is running, simultaneously: Read the next chunk from the file into read_buffer
        freadCheck(read_buffer, 1, copy_amount, src);
        //wait for the copy to complete and synchronize
        cudaCheck(cudaStreamSynchronize(stream));
        //swap buffers 
        std::swap(read_buffer, write_buffer);
        //update the counters so the disk I/O abd PCIe I/O happens in parallel
        rest_bytes -= copy_amount;
        write_buffer_size = copy_amount;
    }

    // after the main loop ends
    //copy the remaining one buffer data
    cudaCheck(cudaMemcpyAsync(gpu_write_ptr, write_buffer, write_buffer_size, cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaStreamSynchronize(stream));
    //clean up the pinned memory
    cudaCheck(cudaFreeHost(buffer_space));
}


#endif