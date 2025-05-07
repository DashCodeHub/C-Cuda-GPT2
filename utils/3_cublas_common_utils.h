
/*
initilizes, configures and provide safety checks for cuBLAS (and cuBLASLt) operations in CUDA code
mkaes it easier to use cuBLAS/cuBLASLt withou repeating the boilerplate everywhere.
    Mainly, This file centralizes cuBLAS/cuBLASLt settings, workspace allocation, 
    precision handling, and error checking for the project.
*/

#ifndef CUBLAS_COMMON_H
#define CUBLAS_COMMON_H

//header files included
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h> // for basic blasss operation
#include <cublasLt.h> // cublasLt for optimized low-level GEMM (matrix multiplication) on newer GPUs
// mostly for tensor ops

// defines data type cuBLAS to be used internally
//matches with precision setting ealier we saw for floatX in `2_cuda_common.h`
/*
ENABLE_FP32 ->	CUDA_R_32F -> (32-bit float)
ENABLE_FP16 ->	CUDA_R_16F -> (16-bit half float)
Default (BF16) -> 	CUDA_R_16BF -> (16-bit bfloat16)
*/
#if defined(ENABLE_FP32)
#define CUBLAS_LOWP CUDA_R_32F
#elif defined(ENABLE_FP16)
#define CUBLAS_LOWP CUDA_R_16F
#else // default to bfloat16
#define CUBLAS_LOWP CUDA_R_16BF
#endif

/* SOME GPU CONCEPTS
----------------------------------------------------------------------------------
In high performance GPU libraries like cuBLAS and cuBLASLt, WROKSPACE MEMORY
refers to extra temporary  GPU memory that is used internally during certain operations like 
GEMM. This workspace is not your input or output
It is an extra scratchpad memory  that cuBLAS/cuBLASLt uses behind the scenes to make operations faster


*/

// assigning Gloabl variabls for cuBLAS
const size_t cublaslt_workspace_size = 32 * 1024 * 1024; //Size of temporary workspace for cuBLASLt GEMMs (matrix multiplications). Hardcoded to 32 MB (especially needed for Hopper GPUs like H100).
void* cublaslt_workspace = NULL; // Pointer to the workspace memory (to be allocated later)
cublasComputeType_t cublas_compute = CUBLAS_COMPUTE_32F; // What compute precision mode cuBLASLt should use (default = float32 compute).
cublasLtHandle_t cublaslt_handle; // The cuBLASLt context handle (must be initialized before use).


// cuBLAS Error Checking heloer
void cublasCheck(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, line);
        exit(EXIT_FAILURE);
    }
}
// defined macro for convinience
#define cublasCheck(status) { cublasCheck((status), __FILE__, __LINE__); }

#endif