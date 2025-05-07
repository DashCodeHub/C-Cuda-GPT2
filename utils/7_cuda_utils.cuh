/*
Provides critical low-level CUDA utilities needed for:
    - Fast memory access (128-bit load/store)
    - Warp/block reductions
    - Datatype conversion (float, half, bf16)
    - Random number generation for stochastic rounding
    - Safe memory allocation, fallback to managed memory
These building blocks used all across `llm.c` CUDA Kernels
*/

// imports
#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include "2_cuda_common_utils.h"

/*
Packed128 is a small struct designed to group multiple elements together in memory 
(float, half, bfloat16, etc.) so that we can load and store 128 bits at once on the 
GPU, using fast CUDA instructions like LDG.128 and STS.128.
Because accessing wider chunks of memory (128-bit at once) is much faster than 
fetching small pieces individually on modern GPUs.

                            Usage
                            -----
load128 => Load 128 bits normally
load128cs => Load 128 bits with cache streaming hint
store128 =>  Store 128 bits normally
store128cs =>  Store 128 bits with cache straming hint
store128cg =>  Store 128 bits with cache straming hint bypassing L1
*/

// Template on ElementType, can be float, half, __nv_bfloat16
template<class ElementType>
// Ensures that the structure is aligned to 16 bytes (128 bits) in memory.
struct alignas(16) Packed128 { 
    // Member 1
    Packed128() = default; // default constructor

    // Member 2
    // Constructor that loads 128 bits from an int4 [a CUDA built-in type = four 32-bit integers = 128 bits]
    __device__ explicit Packed128(int4 bits) {
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch."); // static_assert ensures sizes match at compile time
        memcpy(&payload, &bits, sizeof(bits)); // copy the raw bits into payload - just bitwise reinterpretation not value conversion
    }

    //Member 3
    // Create a Packed128 where all entries are initialized to a given constant value
    __device__  static Packed128 constant(ElementType value) {
        Packed128 result;
        for(int k = 0; k < size; ++k) {
            result.payload[k] = value;
        }
        return result;
    }

    // Member 4
    // Create a Packed128 filled with 0s, built by calling constant(value) internally
    __device__ static Packed128 zeros() {
        return constant(0.f);
    }

    // Member 5
    // Create a Packed128 filled with 1s, built by calling constant(value) internally
    __device__ static Packed128 ones() {
        return constant(1.f);
    }

    // Member 6
    // Allows treating Packed128 like an array. Internally accessthe payload array
    __device__ ElementType& operator[](int index) {
        return payload[index];
    }
    __device__ const ElementType& operator[](int index) const {
        return payload[index];
    }

    // Member 7 
    // Packs the payload array back into a raw int4 bit-representation.
    // Needed when you want to store the Packed128 efficiently back to memory.
    // Again, uses memcpy() to ensure exact bit copying (no value reinterpretation)
    __device__ int4 get_bits() const {
        int4 bits;
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        memcpy(&bits, &payload, sizeof(bits));
        return bits;
    }
    
    // Member 8: Compile-time constant: Number of elements packed inside the 128-bit structure.
    // depends on the type
    /*
    ElementType | size
    float (32-bit) | 4
    half (16-bit) | 8
    __nv_bfloat16 (16-bit) | 8
    So Packed128<float> holds 4 floats and Packed128<half> holds 8 halfs
    */
    static constexpr const size_t size = sizeof(int4) / sizeof(ElementType);

    // Member 9: Data Payload -- The actual data storage, an array sized to exactly fill 128 bits
    ElementType payload[size];
};

//These functions efficiently load and store full 128-bit Packed128 vectors to
// and from aligned memory addresses in GPU global memory. Some of them hint to the 
// hardware about how to treat cache (streaming/bypass hints) for better performance.

// 1. load128 : Takes a pointer to an array of elements
//Bit-casts the memory at address into an int4.tells compiler treat 16 byte at once
// No special cache hint here -- normal memory accesss
template<class ElementType>
__device__ Packed128<ElementType> load128(const ElementType* address) {
    return Packed128<ElementType>{*reinterpret_cast<const int4*>(address)};
}

// 2. load128cs, same as load128, but uses __ldcs(): load with cache streaming hint
/* SOME CONCEPTS OF CUDA
What does streaming mean?
Tells the GPU: "Hey, I am only going to use this data once, so don't waste L1 cache storing it."
This helps avoid cache pollution during massive memory operations (like reading large matrices).
 Used for forward pass loads when you don't need to reuse the input again immediately.
*/
template<class ElementType>
__device__ Packed128<ElementType> load128cs(const ElementType* address) {
    return Packed128<ElementType>{__ldcs(reinterpret_cast<const int4*>(address))};
}

//3. store128: Takes a Packed128 object (value), Get its raw bits as an int4*
// Writes all 128 bits at once, normal store, no special cache hints
// Used for writing output where you might want it to stay cached (like residual connections)
template<class ElementType>
__device__ void store128(ElementType* target, Packed128<ElementType> value) {
    *reinterpret_cast<int4*>(target) = value.get_bits();
}

//4. store128cs: Same as store128, but uses `__stcs()` = store with cache streaming hint
// This data probably wont  be needed again soon, no need to keep it in L1 cache.
// Useful when storing intermediate results that will be immediately used by another kernel
template<class ElementType>
__device__ void store128cs(ElementType* target, Packed128<ElementType> value) {
    __stcs(reinterpret_cast<int4*>(target), value.get_bits());
}

// 5. store128cg: same but uses stcg
// stcg = store globally to L2 cache, bypassing L1 cache
/* SOME CONCEPTS ON CUDA
----------------------------
Store to global memory and L2 cache, but bypass L1 cache.
Good when you know many threads will reuse this data across SMs, not just locally.
Used for very large tensor writes where you want good L2 reuse, but avoid L1 congestion.
*/
template<class ElementType>
__device__ void store128cg(ElementType* target, Packed128<ElementType> value) {
    __stcg(reinterpret_cast<int4*>(target), value.get_bits());
}

// short-form typedefs and enum type to identify  the datatype of a tensor
typedef Packed128<float> f128;
typedef Packed128<floatX> x128;
enum class DType : uint8_t {
    FP32, FP16, BF16
};

// Given a datatype enum, returns the underlying number of bytes
// for a scalar of that type
size_t sizeof_dtype(DType type) {
    switch (type) {
        case DType::FP32:
            return sizeof(float);
        case DType::FP16:
            return sizeof(half);
        case DType::BF16:
            return sizeof(nv_bfloat16);
        default: // handle or get compiler warning
            fprintf(stderr, "Unknown datatype\n");
            exit(EXIT_FAILURE);
    }
}

DType dtype_of(float* f) { return DType::FP32; }
DType dtype_of(nv_bfloat16 * f) { return DType::BF16; }
DType dtype_of(half * f) { return DType::FP16; }

// Copy, cast functions

// device functions and the kernel to cast data between types

// declares a generic device function: takes a value val of type Ts and returns type Td
template<typename Td, typename Ts>
__device__ Td cast_value(Ts val);

// actual definition: float to float (no conversion needed)
template<>
__device__ float cast_value<float, float>(float val) {
    return val;
}

// actual definition: half to float using cuda intrinsic __half2float()
template<>
__device__ float cast_value<float, half>(half val) {
    return __half2float(val);
}

// actual definition: bflaot16 to float32 using cuda intrinsic __bfloat162float()
template<>
__device__ float cast_value<float, __nv_bfloat16>(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

// copy and cast kernel: actual CUDA kernel that copies and casts arrays
template<typename Td, typename Ts>
__global__ void copy_and_cast_kernel(Td* dst, const Ts* src, size_t n, ptrdiff_t stride_dst, ptrdiff_t stride_src) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx + stride_dst * blockIdx.y] = cast_value<Td, Ts>(src[idx + stride_src * blockIdx.y]);
    }
}


// sums value inside a single warp (32 threads)
__device__ inline float warpReduceSum(float val) {
    // In each iteration of the loop:
    // __shfl_xor_sync exchanges data across threads.
    // offset = 16, 8, 4, 2, 1 — the distance between threads that are exchanging values.
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset); // Exchanges a value across threads using XOR of thread indices,
                                                            // does not require shared memory or syncthreads inside a warp
    }
    // Every thread adds its neighbor's value to its own, By the end, all threads inside a warp hold the same sum.
    return val;
}

// finds maximum value inside a warp
// Instead of adding, it finds maximum between own value and neighbor's value
// After the loop, all threads inside the warp hold the same maximum value.
__device__ inline float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}


using reduction_func_t = float (*) (float);

// sums (or reduces) values across the entire CUDA block, even if block has many warps.
// More general reduction across a full CUDA block, even if block is 128, 256, 512, or 1024 threads
// proceeds in 3 stages
template<reduction_func_t warp_reduction>
__device__ inline float blockReduce(float val, bool final_sync=false, float out_of_bounds=0.0f) {
    __shared__ float shared_val[WARP_SIZE];
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    // Stage-1 => Intra Warp reduction: Each warp (32 threads) internally reduces using warpReduceSum or warpReduceMax
    float warp_val = warp_reduction(val);
    // Stage-2 => Inter-warp reduction: Each warp writes its result into shared memory at index warp_id, Only thread 0 inside each warp does the write.
    if (lane_id == 0) { 
        shared_val[warp_id] = warp_val; 
    }
    
    // Stage-3 => warp-Level Final reduction: After all warp results are written:
    // Threads 0, 1, 2, etc., load the warp results,
    // Then perform another warp reduction to get the final block result.
    __syncthreads();
    warp_val = (lane_id < num_warps) ? shared_val[lane_id] : out_of_bounds;
    float block_val = warp_reduction(warp_val);

    // Optionally do a final __syncthreads() if final_sync == true.
    if (final_sync) {
        __syncthreads();
    }
    return block_val;
}

// Perform a sum reduction over a large array in CUDA, but always produce the exact same result (deterministic) every time you run it.
/*
Why deterministic?

When you use many blocks and floating-point addition, the order of addition changes,
Due to associativity issues, results slightly vary (small differences),
This version ensures same thread schedule, same math order, hence same final result.
*/
template<class Float>
__global__ void global_sum_single_block_kernel(float* result, const Float* values, size_t count) {
    assert(gridDim.x == 1);     // Enforce that only one block is launched
                                // If multiple blocks reduce separately and then sum together, 
                                // the order of floating-point operations can vary
    
    
    // Thread-local sum: Each thread starts its own thread, 
    //Jumps ahead by blockDim.x steps each time (strided access)
    // Sums up the assigned elements locally into `thread_sum`
    float thread_sum = 0;
    for(size_t index = threadIdx.x; index < count; index += blockDim.x) {
        thread_sum += (float)values[index];
    }

    // Now reduce all the thread_sums across the blocks using:
    // First inside warp (with warp shuffles), Then across warps (using shared memory)
    float reduction = blockReduce<warpReduceSum>(thread_sum, true);
    if(threadIdx.x == 0) { // thread 0 writes the final reduced sum into *results
        *result = reduction; // only one thread writes, no race condition
    }
}

// Host side wrapper that Launches the kernel `global_sum_single_block_kernel`
// Always launches with exacty 1 block of 1024 threads
// Passes in the Cuda Stream 
template<class Float>
void global_sum_deterministic(float* result, const Float* values, int count, cudaStream_t stream) {
    global_sum_single_block_kernel<<<1, 1024, 0, stream>>>(result, values, count);
    cudaCheck(cudaGetLastError());
}

// CUDA UTILITIES ON MEMORY MANAGEMENT

/*
below function tries to allocate memory normally on the GPU using `cudaMalloc`, but if it fails
due to out of memory (OOM), it falls back to managed memory (cudaMallocManaged), so the program not crashes

*/
int cudaMallocConditionallyManaged(void** out, size_t bytes, const char *file, int line) {

    // First attempt to allocate the memory
    cudaError_t err = cudaMalloc(out, bytes);

    if(err == cudaErrorMemoryAllocation) {
        // check if the error was specifically OOM (Out of Memory)
        cudaGetLastError(); //reset the CUDA error state - else CUDA API calls might fail unexpectedly
        /* SOME CUDA CONCEPTS
        Managed memory is shared between GPU and CPU
        It is slower than device memory but it works wihout crashing
        */
        // Fall back to mamanged memory
        cudaCheck_(cudaMallocManaged(out, bytes), file, line);
        
        // Advise memory location: `cudaMemAdvise`
        // Tell CUDA prefer to put this memory on CPU side
        // Helps performance slightly by avaoiding GPU trying to pull too uch
        cudaCheck_(cudaMemAdvise(*out, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId), file, line);
        return 1;
    } else {
        cudaCheck_(err, file, line); // if no error
        return 0;
    }
}

// macro for convinience
#define cudaMallocConditionallyManaged(out, bytes)\
(cudaMallocConditionallyManaged((void**)out, bytes, __FILE__, __LINE__))


/*
Generate random numbers deterministically inside CUDA (works on device and host)
Use them for stochastic rounding when converting floating-point numbers (like from float32 → bfloat16)
WHY THIS CODE IS NEEDED
------------------------
Normal rounding always rounds to nearest.
Stochastic rounding randomly rounds up or down based on value — this improves numerical accuracy when reducing precision
t's especially useful when saving activations/gradients in mixed precision training
*/

// Tiny fast PRNG (Pseudo Random Number Generator)
// `constexpr` means the compiler can pre compute if argeuments are know at compile time
__device__ __host__ constexpr unsigned int SquirrelNoise5(unsigned int positionX, unsigned int seed)
{
    constexpr unsigned int SQ5_BIT_NOISE1 = 0xd2a80a3f;	// 11010010101010000000101000111111
    constexpr unsigned int SQ5_BIT_NOISE2 = 0xa884f197;	// 10101000100001001111000110010111
    constexpr unsigned int SQ5_BIT_NOISE3 = 0x6C736F4B; // 01101100011100110110111101001011
    constexpr unsigned int SQ5_BIT_NOISE4 = 0xB79F3ABB;	// 10110111100111110011101010111011
    constexpr unsigned int SQ5_BIT_NOISE5 = 0x1b56c4f5;	// 00011011010101101100010011110101
    unsigned int mangledBits = positionX;
    mangledBits *= SQ5_BIT_NOISE1;
    mangledBits += seed;
    mangledBits ^= (mangledBits >> 9);
    mangledBits += SQ5_BIT_NOISE2;
    mangledBits ^= (mangledBits >> 11);
    mangledBits *= SQ5_BIT_NOISE3;
    mangledBits ^= (mangledBits >> 13);
    mangledBits += SQ5_BIT_NOISE4;
    mangledBits ^= (mangledBits >> 15);
    mangledBits *= SQ5_BIT_NOISE5;
    mangledBits ^= (mangledBits >> 17);
    return mangledBits;
}
/*
This gives you a 2D random value based on:

indexX (threadIdx.x, etc.)

indexY (blockIdx.x, etc.)

seed (global random seed)
*/
__device__ __host__ constexpr unsigned int Get2dNoiseUint(int indexX, int indexY, unsigned int seed)
{   
    // Uses a large prime number (198491317) to mix X and Y indices
    constexpr unsigned int PRIME_NUMBER = 198491317u; 
    unsigned int x = static_cast<unsigned int>(indexX);
    unsigned int y = static_cast<unsigned int>(indexY);

    return SquirrelNoise5(x + (PRIME_NUMBER * y), seed); // 
}

// DIFFICULT TO EXPLAIN
// stochastic rounding built on top of Squirel Noise above (with seed updated per step via xorshift)
__device__ __forceinline__ void stochastic_rounding(float in, __nv_bfloat16 *out, unsigned int seed) {

    unsigned int random = Get2dNoiseUint(threadIdx.x, blockIdx.x * blockDim.x + blockIdx.y, seed);
    unsigned int threshold = random & 0xFFFF;
    unsigned int float_bits = __float_as_uint(in);
    unsigned int rounded_bits = float_bits & 0x0000FFFF;
    float_bits = (rounded_bits > threshold) ? (float_bits | 0xFFFF) : (float_bits  & ~0xFFFF);
    *out = __float2bfloat16_rn(__uint_as_float(float_bits));
}
__device__ __forceinline__ void stochastic_rounding(float in, half *out, unsigned int random) {
    *out = (float)in; 
}
__device__ __forceinline__ void stochastic_rounding(float in, float *out, unsigned int random) {
    *out = in; 
}

#endif