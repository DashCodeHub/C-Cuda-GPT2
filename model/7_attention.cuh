/*
This is the most important part of the problem -- THe attention algorithm

file implements a full CUDA-based fallback attention mechanism for 
transformer models, specifically when cuDNN FlashAttention 
is not available. Here's a high-level breakdown of what 
this file does and how it’s structured:

Implements multi-head attention using custom CUDA kernels and cublasLt, covering:

    Permutations (QKV reshaping)
    Batched matrix multiplications (via matmul_cublaslt)
    Softmax (autoregressive)
    Backpropagation for attention

                                Kernels Breakdown
----------------------------------------------------------------------------
1. permute_kernel & permute_kernel_backward
    ----------------------------------------
    Convert QKV tensor from shape (B, T, 3, NH, d) to three separate tensors (B, NH, T, d).
    Used both in forward and backward passes to rearrange data.

2. unpermute_kernel & unpermute_kernel_backward
    --------------------------------------------
    Convert the multi-head output (B, NH, T, d) back to (B, T, C).
    Again, used in both directions of training.

3. softmax_forward_kernel5
    -------------------------------------------
    Compute softmax for each row of the attention logits (B, NH, T, T) using the online softmax algorithm.
    Works only on lower triangle (autoregressive mask)
    Fuses scaling and exponentiation
    Uses warp-level reduction

4. softmax_autoregressive_backward_inplace_kernel
    ---------------------------------------------
    Compute backward pass of softmax efficiently in-place (modifies datt).
    Ensures gradients respect autoregressive masking.



                            KERNEL LAUNCHES
------------------------------------------------------------------------------------

attention_forward()
------------------------------------
        Permute QKV from (B, T, 3C) → (B, NH, T, HS)
        QKᵀ matmul to get logits (T×T) using matmul_cublaslt
        Softmax with masking
        Softmax @ V gives output per head
        Unpermute back to (B, T, C)

attention_backward()
------------------------------------
        Backprop through unpermute
        Matmul to get datt
        Softmax gradient (in-place)
        Compute gradients wrt Q, K, V
        Permute gradients back into original shape

*/


#include "utils/2_cuda_common_utils.h"
#include "utils/7_cuda_utils.cuh"
#include "utils/3_cublas_common_utils.h"
#include "2_matmul.cuh"
#include <assert.h>

/*
Convert:

Input tensor inp of shape (B, N, 3, NH, d) into Output tensors q, k, v each of shape (B, NH, N, d)
done for efficient batched matrix multiplications
*/
__global__ void permute_kernel(floatX* q, floatX* k, floatX* v, const floatX* inp, int B, int N, int NH, int d) {
    
    /* ARGUMENTS
    inp: the combined QKV tensor (B × N × 3 × NH × d)
    q, k, v : outputs (B × NH × N × d)
    */

    // Launch 1D grid of threads over the full output size (i.e., total elements in each of q/k/v)
    // Each thread works on one (b, nh_, n, d_) index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * NH * N * d) { 
        return; 
    }

    // Index decomposition
    // This breaks the flat index idx into 4D indices: b: batch, nh_: head, n: token index, d_: feature
    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;
    // input index calculation
    // compute linear index for Q, in the packed input layout:
    // (B, N, 3, NH, d) as 5 D tensor
    // In memory, it's laid out as:inp[b][n][0][nh_][d_] = inp[(b * N * 3 * NH * d) + ...]
    int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
    
    // Q is at [..., 0, ..., ...]
    q[idx] = __ldcs(&inp[inp_idx]);
    // K is at [..., 1, ..., ...] → offset by NH * d
    k[idx] = __ldcs(&inp[inp_idx + NH * d]);
    // V is at [..., 2, ..., ...] → offset by 2 * NH * d
    v[idx] = __ldcs(&inp[inp_idx + 2 * (NH * d)]);
}


/*

this bwlow reverses the operation performed by permute_kernel. It is used in the backward pass 
of the attention mechanism to combine gradients of q, k, and v back into a single gradient tensor dinp
Given: dq, dk, dv of shape (B, NH, N, d) (gradients of query, key, value)
This kernel fills: dinp of shape (B, N, 3, NH, d) (combined gradient in original format)
*/
__global__ void permute_kernel_backward(floatX* dinp,const floatX* dq, const floatX* dk, const floatX* dv,int B, int N, int NH, int d) {
    /* ARGS:
    dinp: output gradient in packed QKV layout (shape: B × N × 3 × NH × d)
    dq/dk/dv: separate gradients (shape: B × NH × N × d)
    */
    // Flat index over all elements of dq/dk/dv (1 thread = 1 scalar value)
    // total threads: B * NH * N * d
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * NH * N * d) { 
        return; 
    }
    // Breaks flat index into: b = batch index, nh_ = head index, n = token index, d_ = dimension index
    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;
    // This computes the base location in dinp for the corresponding Q value at [b][n][0][nh_][d_].
    // Now we store=> Q: at offset 0, K: at offset + NH*d, V: at offset + 2*NH*d
    int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
    dinp[inp_idx] = dq[idx];
    dinp[inp_idx + NH * d] = dk[idx];
    dinp[inp_idx + 2 * (NH * d)] = dv[idx];
}

/*
 This kernel is responsible for reordering the tensor layout from the format used in attention computation to the standard transformer output format
Convert tensor layout from:inp shape: (B, NH, N, d)`(Batch, NumHeads, Sequence Length, HeadDim)
to:
out shape: (B, N, NH, d)`(Batch, Sequence Length, NumHeads, HeadDim)
This is necessary because:

Most transformer implementations expect output in (B, N, C) format, where C = NH × d
But attention matmuls are done per head, thus working with (B, NH, N, d) format
*/
__global__ void unpermute_kernel(floatX* inp, floatX *out, int B, int N, int NH, int d) {
    /* ARGS
    inp: pointer to input in (B, NH, N, d) layout
    out: pointer to output buffer in (B, N, NH, d) layout
    B, N, NH, d: dimensions (Batch, Tokens, Heads, HeadDim)
    */
    int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    // out[b][n][nh_][d_] <- inp[b][nh_][n][d_]
    if (idx >= B * NH * N * d) { return; }

    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;
    int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
    out[other_idx] = __ldcs(&inp[idx]);
}


/*
This is the reverse of the unpermute_kernel.
It maps gradients from the output layout (B, N, NH, d) back to the input layout (B, NH, N, d) for backpropagation.
*/
__global__ void unpermute_kernel_backward(floatX* dinp, const floatX *dout, int B, int N, int NH, int d) {
    // Each thread is responsible for one scalar element (like a float/bfloat16)
    // idx is a flat index over the entire (B, NH, N, d) tensor
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * NH * N * d) { return; }

    // Decomposes flat index into: b: batch index, nh_: head index, n: sequence position, d_: head dimension
    int b = idx / (NH * N * d);
    int rest = idx % (NH * N * d);
    int nh_ = rest / (N * d);
    rest = rest % (N * d);
    int n = rest / d;
    int d_ = rest % d;
    // This calculates the corresponding index in the dout tensor, which is in (B, N, NH, d) layout:
    // First iterate over batches b, Then sequence n, Then head nh_, Then dimension d_
    int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
    // Move the value from the permuted layout (dout) to the unpermuted layout (dinp)
    // Cast to floatX (could be __half, __nv_bfloat16, etc.)
    dinp[idx] = (floatX)dout[other_idx];
}

/*
Performs a row-wise softmax over each T×T attention matrix slice
Autoregressive: only computes and normalizes up to the diagonal (t ≤ current position)
Inputs/outputs are shaped as (N, T, T) where N = B × NH
*/
__global__ void softmax_forward_kernel5(floatX* out, float inv_temperature, const floatX* inp, int N, int T) {
    assert(T % 4  == 0);
    // Each warp (32 threads) processes one row of the attention matrix, enabling fast warp-level reductions
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    // idx refers to which row of the attention matrix we're computing
    // own_pos is the column index in the T×T matrix, but we only compute up to this position
    // Reverse iteration (gridDim.x - blockIdx.x - 1) helps with better cache reuse in backward pass
    int idx = (gridDim.x - blockIdx.x - 1) * num_warps + warp_id; 
    if(idx >= N * T) {
        return;
    }
    int own_pos = idx % T;
    int pos_by_4 = own_pos / 4;

    const floatX* x = inp + idx * T; //now operate on this row x of shape (T,). Each warp handles one such row

    // not INF, so we don't get NaNs accidentally when subtracting two values.
    const float flt_max = 340282346638528859811704183484516925440.0f; // to avoid including float.h
    float maxval = -flt_max;
    float sumval = 0.0f;

    // Max Reduction (Numerical Stability)
    // This finds the max value in the causal region of the row (up to own_pos) using warp-wise reduction, 
    // to ensure numerical stability during exponentiation.
    //Iterates over a row of attention logits in chunks of 4 (i.e., vectorized loop).
    // Keeps track of the maximum value so far (maxval) to ensure numerical stability.
    // Accumulates sumval = ∑ exp(xᵢ - max) efficiently as max changes
    const floatX* x_aligned = reinterpret_cast<const floatX*>(__builtin_assume_aligned(x, 16));
    for (int i = lane_id; i < pos_by_4; i += WARP_SIZE) {
        // Each thread in the warp (32 threads) gets a different chunk i, spaced WARP_SIZE apart.
        // We're operating only up to own_pos, but in chunks of 4 values per thread — hence pos_by_4
        
        // Load 4 values at once from memory using vectorized access
        // These are the 4 logits for positions 4*i, 4*i+1, ..., 4*i+3
        float regarray[4];
        for (int k = 0; k < 4; ++k) {
            regarray[k] = (float)x_aligned[4*i + k];
        }
        
        // Update the current maxval seen so far over all processed elements.
        // Store the previous maxval to use for correcting the already accumulated sumval
        float old_maxval = maxval;
        for(int k = 0; k < 4; ++k) {
            maxval = fmaxf(maxval, regarray[k]);
        }
        // When maxval changes (increases), we must rescale the sumval to account for the new max:
        // This avoids catastrophic cancellation or overflow when computing exp(xᵢ - max)
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        // Add the new elements' contribution to the softmax sum after max subtraction.
        for(int k = 0; k < 4; ++k) {
            sumval += expf(inv_temperature * (regarray[k] - maxval));
        }
    }
    // 4 * pos_by_4 is the starting index of the final chunk, lane_id goes from 0 to 31 (since it's a warp).
    // own_pos is the last index this thread is responsible for, ensures we only compute softmax on valid lower-triangular entries (autoregressive masking)
    if(4*pos_by_4 + lane_id <= own_pos) {
        float old_maxval = maxval;
        maxval = fmaxf(maxval, (float)x[4*pos_by_4 + lane_id]);
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        sumval += expf(inv_temperature * ((float)x[4*pos_by_4 + lane_id] - maxval));
    }

    // Each thread within a warp may have computed a different maxval for its part of the row
    float global_maxval = warpReduceMax(maxval);
    // Each thread’s sumval was computed relative to its local maxval, rescale the partial sum to be consistent with global_maxval
    sumval *= expf(inv_temperature * (maxval - global_maxval));
    // compute the final softmax denominator by summing all exponentiated values
    float sum = warpReduceSum(sumval);
    // Precompute the normalization constant to multiply the exponentials later
    float norm = 1.f / sum;

    // Each thread now computes the final softmax value for the elements it owns
    // __ldcs is a streaming read from global memory
    // ev is the exponentiated logit after subtracting global_maxval
    // Multiply it by norm to normalize
    // Write the result to out
    for (int i = lane_id; i <= own_pos; i += WARP_SIZE) {
        float ev = expf(inv_temperature * ((float)__ldcs(x + i) - global_maxval));
        __stcs(out + idx * T + i, (floatX)(ev * norm));
    }
}



/*
backward pass for autoregressive softmax (i.e., causal attention). 
It computes the gradient of the softmax output w.r.t. its pre-activation logits — and does so in-place
*/
__global__ void softmax_autoregressive_backward_inplace_kernel(floatX* datt, const floatX* att, int B, int T, int C, float scale) {
    // Launch config: each block handles 4 rows (T_per_block) per batch-head pair
    // Threads process elements of a single row
    constexpr const int BlockSize = 256;
    constexpr int T_per_block = 4;
    
    int t0 = T - 1 - T_per_block*blockIdx.x; // t0 is the starting row index (processed in reverse, important for reuse & cache)
    int idx = blockIdx.y; // idx is the (B * NH)th row of the attention matrix

    // Move pointer to the correct (B, H) slice of the attention map and its gradient.
    att += idx * T * T; 
    datt += idx * T * T;

    // Process up to T_per_block rows (t = timestep), Exit early if t < 0 (important at end-of-sequence).
    for(int to = 0; to < T_per_block; ++to) {
        int t = t0 - to;
        if(t < 0) {
            return;
        }
        const floatX* att_bth = att + t * T;
        const floatX* datt_bth = datt + t * T;
        floatX* dpreatt_bth = datt + t * T;

        float local_sum = 0;
        // We are computing: sumval = sum_i<=t(att_ti, datt_ti)
        // This is reused across all positions in the row t
        for (int t2 = threadIdx.x; t2 <= t; t2 += BlockSize) {
            local_sum += (float)att_bth[t2] * (float)datt_bth[t2];
        }
        local_sum = blockReduce<warpReduceSum>(local_sum);

        // backprop through softmax: If t3 ≤ t (causal region)
        // It calculates: dL/dx_i, this is the softmax gradient using the efficient trick
        // If t3 > t (future positions): We set the gradient to zero — causal masking
        for (int t3 = threadIdx.x; t3 < T; t3 += BlockSize) {
            if(t3 <= t) {
                float acc = (float) __ldcs(att_bth + t3) * ((float) __ldcs(datt_bth + t3) - local_sum);
                __stcs(dpreatt_bth + t3, (floatX) (scale * acc));
            } else {
                __stcs(dpreatt_bth + t3, (floatX)0.f);
            }
        }
    }
}


// KERNEL LAUNCHERS
/*
full forward pass of standard (non-Flash) attention in CUDA — operating in FP8/16 but computing softmax in FP32
    Extracts Q, K, V from a packed input.
    Computes QKᵀ scaled dot product.
    Applies softmax.
    Computes attention output by multiplying with V.
    Reassembles final tensor.
*/
void attention_forward(floatX* out, floatX* qkvr, floatX* att, floatX* inp, int B, int T, int C, int NH, cudaStream_t stream) {
    /* ARGUMENTS
    -----------------
    inp → shape: (B, T, 3C) — holds [Q, K, V] interleaved (packed)
    qkvr → buffer to store unpacked q, k, v
    att → output of softmax(QKᵀ)
    out → final result of attention: shape (B, T, C)
    B = batch size, T = sequence length, C = embedding dim, NH = num heads
    HS = C / NH = head size (per attention head)
    */
    
    NVTX_RANGE_FN();
    const int block_size = 256;
    const int HS = C / NH;

    floatX *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    int total_threads = B * NH * T * HS;
    int num_blocks = CEIL_DIV(total_threads, block_size);
    // inp is packed as (B, T, 3, NH, HS), kernel extracts: q → (B, NH, T, HS), k → (B, NH, T, HS), v → (B, NH, T, HS)
    // Total threads: B * NH * T * HS (1 thread per Q/K/V element)
    permute_kernel<<<num_blocks, block_size, 0, stream>>>(q, k, v, inp, B, T, NH, HS);

    floatX* preatt = inp;
    // preatt ← Q × Kᵀ, shape: (B, NH, T, T), matmul_cublaslt() is the batched matmul wrapper
    matmul_cublaslt(preatt, k, q, nullptr, T, T, HS, stream, true, false, B * NH, T * HS, T * HS, T * T);

    // apply softmax row-wise over T entries, scale = 1 / sqrt(HS), att becomes the normalized weights: shape (B, NH, T, T)
    float scale = 1.f / sqrtf(HS);
    int grid_size = CEIL_DIV(B * NH * T * WARP_SIZE, block_size);
    softmax_forward_kernel5<<<grid_size, block_size, 0, stream>>>(att, scale, preatt, B * NH, T);

    // Output is stored in vaccum, shape: (B, NH, T, HS), y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    floatX* vaccum = inp;
    matmul_cublaslt(vaccum, v, att, nullptr, HS, T, T, stream, false, false, B * NH, T * HS, T * T, T * HS);

    // Unpermute Output → Final Result (B, T, C), Reassembles heads by interleaving head dimension
    num_blocks = CEIL_DIV(B * T * C, block_size);
    unpermute_kernel<<<num_blocks, block_size, 0, stream>>>(vaccum, out, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
}


/*
backpropagates gradients through every subcomponent of the attention forward 
pass — including softmax, matmuls, and permutations.
*/
void attention_backward(floatX* dinp, floatX* dqkvr, floatX* datt, floatX* scratch,const floatX* dout,const floatX* qkvr, const floatX* att,int B, int T, int C, int NH, cudaStream_t stream) {
    NVTX_RANGE_FN();
    
    
    const int block_size = 256;
    const int HS = C / NH;
    // Splits the packed qkvr tensor into separate Q, K, V pointers and their corresponding gradient tensors.
    const floatX *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    floatX *dq, *dk, *dv;
    dq = dqkvr + 0 * B * T * C;
    dk = dqkvr + 1 * B * T * C;
    dv = dqkvr + 2 * B * T * C;

    // dout is shape (B, T, C), converts it into (B, NH, T, HS) to match V's layout, Output goes into scratch, which will be reused later
    int num_blocks = CEIL_DIV(B * T * C, block_size);
    unpermute_kernel_backward<<<num_blocks, block_size, 0, stream>>>(scratch, dout, B, T, NH, HS);
    // Result stored in datt (same buffer as preatt in forward pass)
    matmul_cublaslt(datt, v, scratch, nullptr, T, T, HS, stream, true, false, B * NH, T * HS, T * HS, T * T);
    // backward into V
    matmul_cublaslt(dv, scratch, att, nullptr, HS, T, T, stream, false, true, B * NH, T * HS, T * T, T * HS);
    const float scale = 1.0f / sqrtf((float)HS);
    // backprop through softmax
    softmax_autoregressive_backward_inplace_kernel<<<dim3(T / 4, B * NH), 256>>>(datt, att, B, T, C, scale);
    const floatX* dpreatt = datt;
    // backward into q
    matmul_cublaslt(dq, k, dpreatt, nullptr, HS, T, T, stream, false, false, B * NH, T * HS, T * T, T * HS);
    // backward into k
    matmul_cublaslt(dk, q, dpreatt, nullptr, HS, T, T, stream, false, true, B * NH, T * HS, T * T, T * HS);
    // backward into inp
    num_blocks = CEIL_DIV(B * NH * T * HS, block_size);
    permute_kernel_backward<<<num_blocks, block_size, 0, stream>>>(dinp, dq, dk, dv, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
}