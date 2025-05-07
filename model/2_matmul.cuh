/*
MTRIX MULTIPLICATION HEADER FILE
---------------------------------
 - Forward and Backward matrix multiplication using CUDA and cuBLASLt
 - Can add Bias, GELU fusion, and gradient accumulation.
 - Integration with cuBLASLt, which allows customization of GPU Matmul pipelines
 - Specialized CUDA kernels for things like bias gradients and block-level reductions

*/
#pragma once

#include "2_cuda_common_utils.h"
#include "3_cublas_common_utils.h"
#include "7_cuda_utils.cuh"
#include "1_gelu.cuh"

#include <assert.h>
#include <type_traits>    


/*
wrapper for performing matrix multiplications (GEMMs) using NVIDIA's cuBLASLt API. 
It supports fusion with bias addition and activation functions like GELU, batching, 
transposes, and accumulation.
*/

void matmul_cublaslt(floatX* d, const floatX* a, const floatX* b, const floatX* bias,
    int m, int n, int k, cudaStream_t stream=0, bool transA=true, bool transB=false,
    int batch_count=0, size_t strideA=0, size_t strideB=0, size_t strideOut=0,
    bool accumulate=false, floatX* pre_gelu=NULL, bool backward=false)
{
    /* ARGUMENTS
    ---------------------
    - d: Output matrix
    - a, b: Input matrices
    - bias: Optional bias vector
    - m, n, k: Dimensions of the matrix multiplication (d = a × b)
    - stream: CUDA stream for async execution
    - transA/transB: Whether to transpose A or B
    - batch_count and stride*: For batched matrix multiplication
    - accumulate: Whether to accumulate into output (d += result)
    - pre_gelu: Optional storage for pre-activation GELU input
    - backward: Used during backpropagation to switch epilogues
    */

    // Initialize range and flags
    NVTX_RANGE_FN(); // annotating profiler
    bool has_bias = (bias != NULL); // optional bias to add 
    bool has_gelu = (pre_gelu != NULL); // optional gelu to add

    // cuBLASLt prefers 16-byte aligned memory for performance (SIMD-friendly). 
    // If not, it will error out.
    if(((uintptr_t)a % 16) != 0 || ((uintptr_t)b % 16) != 0 || ((uintptr_t)d % 16) != 0 || ((uintptr_t)bias % 16) != 0) {
    printf("All cuBLASLt pointers must be aligned!\n");
    exit(EXIT_FAILURE);
    }

    // Defines the data types and computational properties of the matmul.
    // cublas_compute: internal global compute mode (e.g., 32-bit accumulators)
    // CUDA_R_32F: scale type (FP32)
    cublasLtMatmulDesc_t operationDesc; 
    cublasCheck(cublasLtMatmulDescCreate(&operationDesc, cublas_compute, CUDA_R_32F));

    
    int returnedResults = 0; // Holds the number of valid algorithms found by cuBLASLt for the given matrix multiplication configuration.
    cublasLtMatmulPreference_t preference; // A preference handle that lets you tell cuBLASLt what kind of algorithms you prefer
    cublasLtMatmulHeuristicResult_t heuristic; // hold the algorithm configuration cuBLASLt selects

    // Specifies whether A or B is transposed. cuBLASLt allows flexible layout handling.
    cublasOperation_t opNoTranspose = CUBLAS_OP_N;
    cublasOperation_t opTranspose = CUBLAS_OP_T;
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, (transA)  ? &opTranspose : &opNoTranspose,   sizeof(opTranspose)));
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, (transB) ? &opTranspose   : &opNoTranspose, sizeof(opNoTranspose)));

    // Describes shapes and strides of A, B, C (input), and D (output) matrices. Adjusts based on transpose status.
    // For example, A (m×k) vs. Aᵀ (k×m).

    // These are handles (objects) that describe the shape, datatype, and memory layout of the matrices
    // cuBLASLt needs explicit layout information because it allows much more flexibility than normal
    cublasLtMatrixLayout_t ALayout; //left input matrix
    cublasLtMatrixLayout_t BLayout; // right input matrix
    cublasLtMatrixLayout_t DLayout; // intermediate output
    cublasLtMatrixLayout_t CLayout; // final output
    
    // Setting up A's Layout in memory
    if (transA) {
        cublasCheck(cublasLtMatrixLayoutCreate(&ALayout, CUBLAS_LOWP, k, m, k)); // If transA == true, meaning matrix A is transposed, the shape is (k × m).
    } else {
        cublasCheck(cublasLtMatrixLayoutCreate(&ALayout, CUBLAS_LOWP, m, k, m)); // If transA == false, A has shape (m × k).
    }

    // Same for B: setting up layout of B in the memory
    if (transB) {
    cublasCheck(cublasLtMatrixLayoutCreate(&BLayout, CUBLAS_LOWP, n, k, n));
    } else {
    cublasCheck(cublasLtMatrixLayoutCreate(&BLayout, CUBLAS_LOWP, k, n, k));
    }


    // Crating layout descriptors for C and D matrix
    // Both C and D have the same shape: mxn [mxk X kxn matrix multiplication]
    // Here the leading dimension is m , because cuBLAS expects column-major storage, so each "column" has m elements.
    cublasCheck(cublasLtMatrixLayoutCreate(&CLayout, (sizeof(floatX) == 1) ? CUDA_R_16BF : CUBLAS_LOWP, m, n, m)); // If sizeof(floatX) == 1, that suggests floatX is 8-bit (FP8), 
                                                                                                                // cuBLASLt needs intermediate C matrix to be in BF16 format (16-bit bfloat16), NOT in FP8
                                                                                                                // Because cuBLASLt internally needs higher precision for accumulation if you’re using very small datatypes like FP8, to prevent catastrophic rounding errors.
    // For final output matrix D, we always use the "normal" datatype (CUBLAS_LOWP), So if floatX is FP32, FP16, BF16, it uses that directly
    cublasCheck(cublasLtMatrixLayoutCreate(&DLayout, CUBLAS_LOWP, m, n, m));

    
    // This block activates strided batched GEMM (general matrix-matrix multiplication) if batch_count > 0.
    // In batched GEMM, you multiply many independent matrix pairs (Aᵢ × Bᵢ = Dᵢ) in a single kernel call for performance.
    if (batch_count) {
        cublasCheck(cublasLtMatrixLayoutSetAttribute(ALayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(BLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(CLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(DLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));

        cublasCheck(cublasLtMatrixLayoutSetAttribute(ALayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(BLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(CLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideOut, sizeof(strideOut)));
        cublasCheck(cublasLtMatrixLayoutSetAttribute(DLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideOut, sizeof(strideOut)));
    }

    // Creates a "preference" object to specify matmul tuning preferences
    // cuBLASLt tries multiple algorithms for matmul. Some may use more GPU memory (workspace) for speed.
    // cublaslt_workspace_size is usually 32 MB (hardcoded earlier). It allows the library to use up to that much temporary GPU memory to optimize performance.
    cublasCheck(cublasLtMatmulPreferenceCreate(&preference));
    cublasCheck(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&cublaslt_workspace_size, sizeof(cublaslt_workspace_size)));

    
    cublasLtEpilogue_t epilogue;
    // Epilogue tells cuBLASLt to do more than just a pure matmul
    if (has_gelu) {
        int64_t gelu_ld = m; // Configure a pointer (pre_gelu) where the intermediate values (before GELU) should be stored
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &gelu_ld, sizeof(gelu_ld)));
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &pre_gelu, sizeof(pre_gelu)));
        // Needed for backward pass to compute the derivative of GELU
        if (backward) {
            assert(!has_bias); // we shouldn't have any backward matmuls that use both GELU and bias
            epilogue = CUBLASLT_EPILOGUE_DGELU;
        } else {
            epilogue = has_bias ? CUBLASLT_EPILOGUE_GELU_AUX_BIAS : CUBLASLT_EPILOGUE_GELU_AUX;
        }
    } else if(has_bias){
        epilogue = backward ? CUBLASLT_EPILOGUE_BGRADB : CUBLASLT_EPILOGUE_BIAS; // Add bias (BIAS) in forward, Accumulate gradient w.r.t bias (BGRADB) in backward
    } else {
        epilogue = CUBLASLT_EPILOGUE_DEFAULT; // Just perform D = A * B
    }
    // set the epilogue
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    if (has_bias) {
        // Bias type is correct for FP8 (must be BF16).
        // Bias pointer is passed so cuBLASLt knows where to fetch it.
        cublasDataType_t bias_data_type = (sizeof(floatX) == 1) ? CUDA_R_16BF : CUBLAS_LOWP;
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_data_type, sizeof(bias_data_type)));
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    }

    // Specifies the datatype used for alpha and beta scaling factors in the matmul:
    cublasDataType_t scale_type = CUDA_R_32F; // Accumulating in FP32 is numerically safer even if A, B, C are FP16, BF16, or FP8.
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type))); // FP16 is only required if using CUBLAS_COMPUTE_16F, which is rare

    // Asks cuBLASLt to suggest the best algorithm for this matrix multiplication, given: Matrix sizes, Data types, Preferences, Optional features
    cublasLtMatmulAlgoGetHeuristic(cublaslt_handle, operationDesc, ALayout, BLayout, CLayout, DLayout,
                    preference, 1, &heuristic, &returnedResults);
    // If no algorithm is found, it fails
    if (returnedResults == 0) {
        printf("No cuBLASLt algorithm: m: %d, n: %d, k: %d, bias: %d\n", n, m, k, has_bias);
        exit(EXIT_FAILURE);
    }

    // Set alpha and beta Scaling Factors
    // alpha = 1.0: Full contribution from A × B.
    // beta = 0.0: Overwrite D = A × B
    // beta = 1.0: If accumulate is true, perform D += A × B
    const float alpha = 1.0f, beta = accumulate ? 1.0f : 0.0f;

    // launch the matricx multiplication, with optional fused bias, fused FELU, batching and striding if requested.
    cublasCheck(cublasLtMatmul(cublaslt_handle, operationDesc,
                &alpha, a, ALayout, b, BLayout, &beta, d, CLayout, d, DLayout,
                &heuristic.algo, cublaslt_workspace, cublaslt_workspace_size, stream));

    // Frees descriptors and layouts to avoid memory/resource leaks
    cublasCheck(cublasLtMatmulPreferenceDestroy(preference));
    cublasCheck(cublasLtMatmulDescDestroy(operationDesc));
    cublasCheck(cublasLtMatrixLayoutDestroy(ALayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(BLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(CLayout));
    cublasCheck(cublasLtMatrixLayoutDestroy(DLayout));
    cudaCheck(cudaGetLastError());
}


// small wrapper around matmul_cublaslt for the forward pass (keeping historical order of arguments)
void matmul_forward_cublaslt(floatX* out,
    floatX* inp, floatX* weight, floatX* bias,
    int B, int T, int C, int OC, cudaStream_t stream,
    floatX* pre_gelu=NULL, int gelu_fusion=1) {
    /*
    floatX is a generic type: could be float, half, or bfloat16.
    B = batch size, T = sequence length → total rows: B*T
    C = input channels, OC = output channels
    */
    if (gelu_fusion < 1 && pre_gelu) {
        // gelu fusion 
        matmul_cublaslt(pre_gelu, weight, inp, bias, OC, B*T, C, stream, true, false, 0, 0, 0, 0, false, NULL, false);
        gelu_forward(out, pre_gelu, B*T*OC, stream);
    } else {
        matmul_cublaslt(out, weight, inp, bias, OC, B*T, C, stream, true, false, 0, 0, 0, 0, false, pre_gelu, false);
    }
}

// parallel row-wise reduction: it adds multiple vectors (from src) together along a dimension, and stores the reduced result into dst
__global__ void reduce_add_sum_kernel(floatX* dst, const float* src, size_t n, size_t m) {
    /*
    - src: a 2D tensor of size [m][n] flattened as 1D (row-major).
    - dst: a 1D vector of length n, which accumulates the sum of all m rows.
    - n: number of columns (elements in each row).
    - m: number of rows.
    */
    const size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * f128::size; // Threads are grouped to work on f128::size consecutive elements (like a mini-vector).
    assert(n % x128::size == 0); // Ensure n is divisible by the vector size x128::size (alias of f128::size). Skip threads that go past the end.
    if (idx < n) {
        f128 acc;
        // A vector (f128) accumulator initialized to 0 for partial sum accumulation
        for(int k = 0; k < f128::size; ++k) {
            acc[k] = 0.f;
        }
        // For each row l, load f128-sized chunk at position idx + n * l (i.e., column idx of row l)
        for(int l = 0; l < m; ++l) {
            f128 s = load128(src + idx + n * l);
            for(int k = 0; k < f128::size; ++k) {
                acc[k] += s[k];
            }
        }
        // Add the accumulated sum to the existing value in dst (+=)
        // Cast to floatX in case you're mixing precision (e.g., accumulating in float, storing in half or bfloat16)
        for(int k = 0; k < f128::size; ++k) {
            dst[idx + k] = (floatX) ((float)dst[idx + k] + acc[k]);
        }
    }
}



// 

template<typename OutFloat, bool UseAuxBuffer>
// This kernel computes the bias gradient from the output gradient dout. It's designed for maximum parallelism, using warps and shared memory
__global__ void matmul_backward_bias_kernel9(OutFloat* dbias, const floatX* dout, int B, int T, int OC,
                                             std::bool_constant<UseAuxBuffer>) {
    /*
    OutFloat* dbias:	                Output buffer to store computed bias gradients
    const floatX* dout:	                Input gradient tensor (shape B × T × OC)
    int B:	                            Batch size
    int T:	                            Sequence length (time steps)
    int OC:	                            Output channels (number of features)
    std::bool_constant<UseAuxBuffer>:	Flag: whether to use an auxiliary buffer or write directly
    */
    // Threads are grouped in a blockDim = {bdx, bdy, bz} structure.
    // each thread is indexed as warp_d, warp_c, and block_d:
    // warp_d: fast inner loop dimension (x)
    // warp_c: controls the output channel slice (y)
    // block_d: stacks warps along z-axis
    constexpr const int bdx = 4;
    constexpr const int bdy = WARP_SIZE / bdx;
    assert(blockDim.x == bdx);
    assert(blockDim.y == bdy);
    
    // A warp computes a slice of the OC dimension
    // global_oc is the starting index for output channels this warp handles
    int warp_d = (int)threadIdx.x;
    int warp_c = (int)threadIdx.y;
    int block_d = (int)threadIdx.z;

    const int OC_per_warp = bdy * x128::size;                                            
    int local_oc = warp_c * x128::size;
    int global_oc = blockIdx.x * OC_per_warp + local_oc;

    // Each thread handles a particular (B,T) index flattened as BT 
    int local_bt = warp_d + bdx * block_d;
    int bt_per_block = bdx * blockDim.z;
    
    // Each thread accumulates x128::size floats (i.e., a packed vector).
    float accumulators[x128::size];
    for (int k = 0; k < x128::size; k++) {
        accumulators[k] = 0.0f;
    }
    // Loops over all rows of dout, accumulates OC slices in registers
    if(global_oc < OC) {
        for (int idx = blockIdx.y * bt_per_block + local_bt; idx < B * T; idx += gridDim.y * bt_per_block) {
            x128 packed_dout = load128(dout + global_oc + idx*OC);
            for (int k = 0; k < x128::size; k++) {
                accumulators[k] += (float)packed_dout[k];
            }
        }
    }

    // Stores intermediate sums per warp into shared memory for reduction
    __shared__ float sub_results[x128::size][WARP_SIZE][bdy];
    // Uses __shfl_down_sync to reduce across threads within a warp
    // Only one thread writes the result to shared memory
    for (int k = 0; k < x128::size; k++) {
        float v = accumulators[k];
        v += __shfl_down_sync(0xffffffff, v, 1, 4);
        v += __shfl_down_sync(0xffffffff, v, 2, 4);
        if(warp_d == 0) {
            sub_results[k][block_d][warp_c] = v;
        }
    }
    __syncthreads();

    // Aggregates the shared memory partials across blockDim.z
    // If UseAuxBuffer is true, stores to a buffer for later reduction
    for (int k = block_d; k < x128::size; k += blockDim.z) {
        float a = 0.f;
        for (int r = warp_d; r < blockDim.z; r += bdx) {
            float v = sub_results[k][r][warp_c];
            v += __shfl_down_sync(0xffffffff, v, 1, 4);
            v += __shfl_down_sync(0xffffffff, v, 2, 4);
            a += v;
        }
        if(warp_d == 0 && global_oc < OC) {
            if constexpr (!UseAuxBuffer) {
                dbias[global_oc + k] = (OutFloat)(a + (float)dbias[global_oc + k]);
            } else {
                dbias[global_oc + k + blockIdx.y * OC] = a;
            }
        }
    }
}

/*
wrapper for backward passes of matrix multiplication, handling gradients for input, weights, and optionally bias. Internally, it calls:
    matmul_backward_bias_kernel9 for bias gradient
    matmul_cublaslt for input and weight gradients
    gelu_backward_inplace if GELU activation is present and not fused
*/
void matmul_backward(floatX* dinp, floatX* dweight, floatX* dbias,floatX* dout, floatX* inp, floatX* weight,float* dbias_buffer,int B, int T, int C, int OC, cudaStream_t stream,
    floatX* pre_gelu=NULL, int gelu_fusion=1) {
    /*
    | **Argument**        | **Description**                                                                 |
    |---------------------|---------------------------------------------------------------------------------|
    | `floatX* dinp`      | Output gradient w.r.t. input                                                    |
    | `floatX* dweight`   | Output gradient w.r.t. weights                                                  |
    | `floatX* dbias`     | Output gradient w.r.t. bias (optional)                                         |
    | `floatX* dout`      | Upstream gradient from the next layer                                          |
    | `floatX* inp`       | Input activations used during the forward pass                                 |
    | `floatX* weight`    | Weights used during the forward pass                                           |
    | `float* dbias_buffer`| Temporary buffer for intermediate bias gradients                              |
    | `int B`             | Batch size                                                                     |
    | `int T`             | Sequence length or time steps                                                  |
    | `int C`             | Input feature dimension                                                        |
    | `int OC`            | Output feature dimension                                                       |
    | `cudaStream_t stream`| CUDA stream for kernel execution                                              |
    | `floatX* pre_gelu`  | Optional buffer to store pre-GELU activations                                  |
    | `int gelu_fusion`   | Flag controlling the use of fused GELU operations                              |
    */
    
    NVTX_RANGE_FN();


    if (dbias != NULL) { // only execute if the caller provides a non-null bias gradient pointer

        // Sets up CUDA block configuration, 4x8x(block_size/32) thread block
        const int block_size = deviceProp.maxThreadsPerMultiProcessor == 1536 ? 768 : 1024;

        // OC_per_warp: how many output channels each warp processes.
        // grid_size_x: splits output channels across blocks.
        // grid_size_y: chooses how many blocks to spawn based on available SMs.
        dim3 block_dim = {4, 8, (unsigned)block_size/WARP_SIZE};
        const int OC_per_warp = block_dim.y * x128::size; // 64 at BF16
        const int grid_size_x = CEIL_DIV(OC, OC_per_warp); // e.g. 12 horizontal blocks for 768 OCs at BF16
        const int grid_size_y = max(1, deviceProp.maxThreadsPerMultiProcessor * deviceProp.multiProcessorCount / (block_size * grid_size_x)); // full GPU!

        // Case:1 No auxillary buffer needed, accumulate directly into dbias
        if(grid_size_y == 1) {
            matmul_backward_bias_kernel9<<<dim3(grid_size_x, grid_size_y), block_dim, 0, stream>>>(dbias, dout, B, T, OC, False);
        cudaCheck(cudaGetLastError());
        } else {
            // cCase: 2  Needs accumulation from buffer, First stores per-block partial sums in dbias_buffer, Then reduces them into final dbias.
            matmul_backward_bias_kernel9<<<dim3(grid_size_x, grid_size_y), block_dim, 0, stream>>>(dbias_buffer, dout, B, T, OC, True);
            cudaCheck(cudaGetLastError());
            reduce_add_sum_kernel<<<CEIL_DIV(OC, 256 * f128::size), 256, 0, stream>>>(dbias, dbias_buffer, OC, grid_size_y);
            cudaCheck(cudaGetLastError());
        }
        // Prevent double bias computation (if matmul_cublaslt later fuses bias)
        dbias = NULL; 
    }

    //Computes dinp = dout × weight^T, No bias, no accumulation, Uses pre-GELU if fusion enabled
    matmul_cublaslt(dinp, weight, dout, NULL, C, B*T, OC, stream, false, false, 0, 0, 0, 0, false,
    gelu_fusion >= 2 ? pre_gelu : NULL, true);

    if (gelu_fusion < 2 && pre_gelu) {
        gelu_backward_inplace(dinp, pre_gelu, B*T*C, stream); // Applies backward GeLU manually only if not fused into matmul
    }

    // weight gradient computation, Computes dweight += inp^T × dout, Uses += instead of = to accumulate gradient, Fuses nothing
    matmul_cublaslt(dweight, inp, dout, NULL /*dbias*/, C, OC, B*T, stream, false, true, 0, 0, 0, 0,
    true /* accumulate */, NULL, true);
}