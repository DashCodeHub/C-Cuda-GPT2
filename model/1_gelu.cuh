/*
Implements GeLU (Gaussian Error Linear Unit) non-linearity in CUDA both forward pass
and backward pass for use in neural networks like GPT-2. 
Specifically what is done here is:
    - Fast Approximate for GeLU function
    - Fully vectorized (loads / stores 128 bit chunks at once)
    - Designed for mixed precision (float16, bfloat16, or float32 depending `floatX`)

Specifically approximate gelu is caluclated using tanh activation and this does not cause much
degradation in accuracy. The Approximation gelu is given as:

        GeLU(x)=0.5x(1+sqrt(2/PI)*tanh((x+0.044715(x^3))))

    __NOTE__: From now on what ever functions we are going to develop for model, we will be developing
    both the for the FORWARD PASS and the BACKWARD PASS
*/
# pragma once
#include <assert.h>
#include "utils/2_cuda_common_utils.h"
#include "utils/7_cuda_utils.cuh"

// Macro to define gelu activation scaling factor which is sqrt(2/PI)
#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)


/*
GELU FORWARD PASS KERNEL
----------------------------------------
Each thread:
    - Loads 128-bit chunk from inp (using load128cs — "cache streaming"),
    - Applies approximate GeLU transformation on each value,
    - Stores result to out (with store128 — normal store), in case it is useful 
        for the data to be in the cache for the next operation after this GeLU
*/ 

__global__ void gelu_forward_kernel2(floatX* out, const floatX* inp) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size; // multiply by `x128::size` because each thread processes multiple elements at onece -- not just scalar but a vector of size x128::size values
    // for e.g. If x128::size = 4, then thread 0 handles elements 0–3, thread 1 handles 4–7 etc
    x128 packed_out;
    x128 packed_inp = load128cs(inp + idx); 
    for(int k = 0; k < packed_inp.size; ++k) {
        float xi = (float)packed_inp[k];
        float cube = 0.044715f * xi * xi * xi;
        packed_out[k] = (floatX)(0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube))));
    }
    store128(out + idx, packed_out);
}


/*
GELU BACKWARD PASS KERNEL
--------------------------------------
Each thread:
     - Loads inputs and upstream gradients (d_in_out).
     - Computes the derivative of GeLU using:
        tanh, cosh, sech², and a cubic term
     - Multiplies this gradient with d_in_out (in-place update).
     - Stores result.

If you do the math you can see the derivative willturn out to be:
    Let u(x) = sqrt(2/PI) * (x + 0.044715 * x^3)
    Let u'(x) = sqrt(2/PI) * (1 + 3 * 0.044715 * x^2)

    dGeLU(x)/dx = 0.5 * (1 + tanh(u(x))) + 0.5 * x * (1 - tanh(u(x))^2) * u'(x)

*/
__global__ void gelu_backward_inplace_kernel(floatX* d_in_out, const floatX* inp) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;

    x128 packed_dinp;
    x128 packed_inp = load128cs(inp + idx);
    x128 packed_dout = load128(d_in_out + idx);
    for (int k = 0; k < packed_inp.size; ++k) {
        float x = (float)packed_inp[k];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        packed_dinp[k] = (floatX)(local_grad * (float)packed_dout[k]);
    }
    store128(d_in_out + idx, packed_dinp); // used in backpropagation during training
}


// Launch the forward kernel, Uses block_size = 512, Total grid size: CEIL_DIV(N, block_size * x128::size)
// Requires N to be a multiple of the vectorized chunk size
void gelu_forward(floatX* out, const floatX* inp, int N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 512;
    assert(N % (block_size * x128::size) == 0);
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    gelu_forward_kernel2<<<grid_size, block_size, 0, stream>>>(out, inp);
    cudaCheck(cudaGetLastError());
}

// Launch backward kernel
// block_size = 128 (more compute-intensive), Same grid logic
void gelu_backward_inplace(floatX* d_in_out, const floatX* inp, const int N, cudaStream_t stream) {
    NVTX_RANGE_FN();
    const int block_size = 128;
    assert(N % (block_size * x128::size) == 0);
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    gelu_backward_inplace_kernel<<<grid_size, block_size, 0, stream>>>(d_in_out, inp);
    cudaCheck(cudaGetLastError());
}