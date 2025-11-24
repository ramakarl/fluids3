/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation and 
 * any modifications thereto.  Any use, reproduction, disclosure, or distribution 
 * of this software and related documentation without an express license 
 * agreement from NVIDIA Corporation is strictly prohibited.
 * 
 */

#define NUM_BANKS		16
#define LOG_NUM_BANKS	 4

#ifndef _SCAN_BEST_KERNEL_CU_
#define _SCAN_BEST_KERNEL_CU_

// Define this to more rigorously avoid bank conflicts, 
// even at the lower (root) levels of the tree
// Note that due to the higher addressing overhead, performance 
// is lower with ZERO_BANK_CONFLICTS enabled.  It is provided
// as an example.
//#define ZERO_BANK_CONFLICTS 


#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS + (index) >> (2*LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)
#endif

///////////////////////////////////////////////////////////////////////////////
// Work-efficient compute implementation of scan, one thread per 2 elements
// Work-efficient: O(log(n)) steps, and O(n) adds.
// Also shared storage efficient: Uses n + n/NUM_BANKS shared memory -- no ping-ponging
// Also avoids most bank conflicts using single-element offsets every NUM_BANKS elements.
//
// In addition, If ZERO_BANK_CONFLICTS is defined, uses 
//     n + n/NUM_BANKS + n/(NUM_BANKS*NUM_BANKS) 
// shared memory. If ZERO_BANK_CONFLICTS is defined, avoids ALL bank conflicts using 
// single-element offsets every NUM_BANKS elements, plus additional single-element offsets 
// after every NUM_BANKS^2 elements.
//
// Uses a balanced tree type algorithm.  See Blelloch, 1990 "Prefix Sums 
// and Their Applications", or Prins and Chatterjee PRAM course notes:
// https://www.cs.unc.edu/~prins/Classes/633/Handouts/pram.pdf
// 
// This work-efficient version is based on the algorithm presented in Guy Blelloch's
// excellent paper "Prefix sums and their applications".
// http://www.cs.cmu.edu/~blelloch/papers/Ble93.pdf
//
// Pro: Work Efficient, very few bank conflicts (or zero if ZERO_BANK_CONFLICTS is defined)
// Con: More instructions to compute bank-conflict-free shared memory addressing,
// and slightly more shared memory storage used.
//

template <bool isNP2> __device__ void loadSharedChunkFromMem (float *s_data, const float *g_idata, int n, int baseIndex, int& ai, int& bi, int& mem_ai, int& mem_bi, int& bankOffsetA, int& bankOffsetB )
{
    int thid = threadIdx.x;
    mem_ai = baseIndex + threadIdx.x;
    mem_bi = mem_ai + blockDim.x;

    ai = thid;
    bi = thid + blockDim.x;
    bankOffsetA = CONFLICT_FREE_OFFSET(ai);		    // compute spacing to avoid bank conflicts
    bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    
	s_data[ai + bankOffsetA] = g_idata[mem_ai];		// Cache the computational window in shared memory pad values beyond n with zeros
    
    if (isNP2) { // compile-time decision
        s_data[bi + bankOffsetB] = (bi < n) ? g_idata[mem_bi] : 0; 
    } else {
        s_data[bi + bankOffsetB] = g_idata[mem_bi]; 
    }
}


template <bool isNP2> __device__ void loadSharedChunkFromMemInt (int *s_data, const int *g_idata, int n, int baseIndex, int& ai, int& bi, int& mem_ai, int& mem_bi, int& bankOffsetA, int& bankOffsetB )
{
    int thid = threadIdx.x;
    mem_ai = baseIndex + threadIdx.x;
    mem_bi = mem_ai + blockDim.x;

    ai = thid;
    bi = thid + blockDim.x;
    bankOffsetA = CONFLICT_FREE_OFFSET(ai);		    // compute spacing to avoid bank conflicts
    bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    
	s_data[ai + bankOffsetA] = g_idata[mem_ai];		// Cache the computational window in shared memory pad values beyond n with zeros
    
    if (isNP2) { // compile-time decision
        s_data[bi + bankOffsetB] = (bi < n) ? g_idata[mem_bi] : 0; 
    } else {
        s_data[bi + bankOffsetB] = g_idata[mem_bi]; 
    }
}

template <bool isNP2> __device__ void storeSharedChunkToMem(float* g_odata, const float* s_data, int n, int ai, int bi, int mem_ai, int mem_bi,int bankOffsetA, int bankOffsetB)
{
    __syncthreads();

    g_odata[mem_ai] = s_data[ai + bankOffsetA];			// write results to global memory
    if (isNP2) { // compile-time decision
        if (bi < n) g_odata[mem_bi] = s_data[bi + bankOffsetB]; 
    } else {
        g_odata[mem_bi] = s_data[bi + bankOffsetB]; 
    }
}
template <bool isNP2> __device__ void storeSharedChunkToMemInt (int* g_odata, const int* s_data, int n, int ai, int bi, int mem_ai, int mem_bi,int bankOffsetA, int bankOffsetB)
{
    __syncthreads();

    g_odata[mem_ai] = s_data[ai + bankOffsetA];			// write results to global memory
    if (isNP2) { // compile-time decision
        if (bi < n) g_odata[mem_bi] = s_data[bi + bankOffsetB]; 
    } else {
        g_odata[mem_bi] = s_data[bi + bankOffsetB]; 
    }
}


template <bool storeSum> __device__ void clearLastElement( float* s_data, float *g_blockSums, int blockIndex)
{
    if (threadIdx.x == 0) {
        int index = (blockDim.x << 1) - 1;
        index += CONFLICT_FREE_OFFSET(index);        
        if (storeSum) { // compile-time decision
            // write this block's total sum to the corresponding index in the blockSums array
            g_blockSums[blockIndex] = s_data[index];
        }
        s_data[index] = 0;		// zero the last element in the scan so it will propagate back to the front
    }
}

template <bool storeSum> __device__ void clearLastElementInt ( int* s_data, int *g_blockSums, int blockIndex)
{
    if (threadIdx.x == 0) {
        int index = (blockDim.x << 1) - 1;
        index += CONFLICT_FREE_OFFSET(index);        
        if (storeSum) { // compile-time decision
            // write this block's total sum to the corresponding index in the blockSums array
            g_blockSums[blockIndex] = s_data[index];
        }
        s_data[index] = 0;		// zero the last element in the scan so it will propagate back to the front
    }
}


__device__ unsigned int buildSum(float *s_data)
{
    unsigned int thid = threadIdx.x;
    unsigned int stride = 1;
    
    // build the sum in place up the tree
    for (int d = blockDim.x; d > 0; d >>= 1) {
        __syncthreads();

        if (thid < d) {
            int i  = __mul24(__mul24(2, stride), thid);
            int ai = i + stride - 1;
            int bi = ai + stride;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            s_data[bi] += s_data[ai];
        }
        stride *= 2;
    }
    return stride;
}
__device__ unsigned int buildSumInt (int *s_data)
{
    unsigned int thid = threadIdx.x;
    unsigned int stride = 1;
    
    // build the sum in place up the tree
    for (int d = blockDim.x; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int i  = __mul24(__mul24(2, stride), thid);
            int ai = i + stride - 1;
            int bi = ai + stride;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            s_data[bi] += s_data[ai];
        }
        stride *= 2;
    }
    return stride;
}

__device__ void scanRootToLeaves(float *s_data, unsigned int stride)
{
     unsigned int thid = threadIdx.x;

    // traverse down the tree building the scan in place
    for (int d = 1; d <= blockDim.x; d *= 2) {
        stride >>= 1;
        __syncthreads();

        if (thid < d) {
            int i  = __mul24(__mul24(2, stride), thid);
            int ai = i + stride - 1;
            int bi = ai + stride;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            float t = s_data[ai];
            s_data[ai] = s_data[bi];
            s_data[bi] += t;
        }
    }
}

__device__ void scanRootToLeavesInt (int *s_data, unsigned int stride)
{
     unsigned int thid = threadIdx.x;

    // traverse down the tree building the scan in place
    for (int d = 1; d <= blockDim.x; d *= 2) {
        stride >>= 1;
        __syncthreads();

        if (thid < d) {
            int i  = __mul24(__mul24(2, stride), thid);
            int ai = i + stride - 1;
            int bi = ai + stride;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            int t = s_data[ai];
            s_data[ai] = s_data[bi];
            s_data[bi] += t;
        }
    }
}

template <bool storeSum> __device__ void prescanBlock(float *data, int blockIndex, float *blockSums)
{
    int stride = buildSum (data);               // build the sum in place up the tree
    clearLastElement<storeSum> (data, blockSums, (blockIndex == 0) ? blockIdx.x : blockIndex);
    scanRootToLeaves (data, stride);            // traverse down tree to build the scan 
}
template <bool storeSum> __device__ void prescanBlockInt (int *data, int blockIndex, int *blockSums)
{
    int stride = buildSumInt (data);               // build the sum in place up the tree
    clearLastElementInt <storeSum>(data, blockSums, (blockIndex == 0) ? blockIdx.x : blockIndex);
    scanRootToLeavesInt (data, stride);            // traverse down tree to build the scan 
}

__global__ void uniformAdd (float *g_data, float *uniforms, int n, int blockOffset, int baseIndex)
{
    __shared__ float uni;
    if (threadIdx.x == 0) uni = uniforms[blockIdx.x + blockOffset];    
    unsigned int address = __mul24(blockIdx.x, (blockDim.x << 1)) + baseIndex + threadIdx.x; 

    __syncthreads();    
    // note two adds per thread
    g_data[address]              += uni;
    g_data[address + blockDim.x] += (threadIdx.x + blockDim.x < n) * uni;
}
__global__ void uniformAddInt (int *g_data, int *uniforms, int n, int blockOffset, int baseIndex)
{
    __shared__ int uni;
    if (threadIdx.x == 0) uni = uniforms[blockIdx.x + blockOffset];    
    unsigned int address = __mul24(blockIdx.x, (blockDim.x << 1)) + baseIndex + threadIdx.x; 

    __syncthreads();    
    // note two adds per thread
    g_data[address]              += uni;
    g_data[address + blockDim.x] += (threadIdx.x + blockDim.x < n) * uni;
}

template <bool storeSum, bool isNP2> __global__ void prescan(float *g_odata, const float *g_idata, float *g_blockSums, int n, int blockIndex, int baseIndex) {
	int ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB;
	extern __shared__ float s_data[];
	loadSharedChunkFromMem<isNP2>(s_data, g_idata, n, (baseIndex == 0) ? __mul24(blockIdx.x, (blockDim.x << 1)):baseIndex, ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB); 
	prescanBlock<storeSum>(s_data, blockIndex, g_blockSums); 
	storeSharedChunkToMem<isNP2>(g_odata, s_data, n, ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB); 
}
		
template <bool storeSum, bool isNP2> __global__ void prescanInt (int *g_odata, const int *g_idata, int *g_blockSums, int n, int blockIndex, int baseIndex) {
	int ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB;
	extern __shared__ int s_dataInt [];
	loadSharedChunkFromMemInt <isNP2>(s_dataInt, g_idata, n, (baseIndex == 0) ? __mul24(blockIdx.x, (blockDim.x << 1)):baseIndex, ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB); 
	prescanBlockInt<storeSum>(s_dataInt, blockIndex, g_blockSums); 
	storeSharedChunkToMemInt <isNP2>(g_odata, s_dataInt, n, ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB); 
}

#define SCAN_BLOCKSIZE      512

__global__ void prefixFixup(unsigned int *input, unsigned int *aux, int len)
{
	unsigned int t = threadIdx.x;
	unsigned int start = t + 2 * blockIdx.x * SCAN_BLOCKSIZE;
	if (start < len)					input[start] += aux[blockIdx.x];
	if (start + SCAN_BLOCKSIZE < len)   input[start + SCAN_BLOCKSIZE] += aux[blockIdx.x];
}

__global__ void prefixSum(unsigned int* input, unsigned int* output, unsigned int* aux, int len, int zeroff)
{
	__shared__ unsigned int scan_array[SCAN_BLOCKSIZE << 1];
	unsigned int t1 = threadIdx.x + 2 * blockIdx.x * SCAN_BLOCKSIZE;
	unsigned int t2 = t1 + SCAN_BLOCKSIZE;

	// Pre-load into shared memory
	scan_array[threadIdx.x] = (t1<len) ? input[t1] : 0.0f;
	scan_array[threadIdx.x + SCAN_BLOCKSIZE] = (t2<len) ? input[t2] : 0.0f;
	__syncthreads();

	// Reduction
	int stride;
	for (stride = 1; stride <= SCAN_BLOCKSIZE; stride <<= 1) {
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index < 2 * SCAN_BLOCKSIZE)
			scan_array[index] += scan_array[index - stride];
		__syncthreads();
	}

	// Post reduction
	for (stride = SCAN_BLOCKSIZE >> 1; stride > 0; stride >>= 1) {
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index + stride < 2 * SCAN_BLOCKSIZE)
			scan_array[index + stride] += scan_array[index];
		__syncthreads();
	}
	__syncthreads();

	// Output values & aux
	if (t1 + zeroff < len)	output[t1 + zeroff] = scan_array[threadIdx.x];
	if (t2 + zeroff < len)	output[t2 + zeroff] = (threadIdx.x == SCAN_BLOCKSIZE - 1 && zeroff) ? 0 : scan_array[threadIdx.x + SCAN_BLOCKSIZE];
	if (threadIdx.x == 0) {
		if (zeroff) output[0] = 0;
		if (aux) aux[blockIdx.x] = scan_array[2 * SCAN_BLOCKSIZE - 1];
	}
}


template __global__ void prescan<true,true> (float*, const float*, float*, int, int, int);
template __global__ void prescan<true,false> (float*, const float*, float*, int, int, int); 
template __global__ void prescan<false,true> (float*, const float*, float*, int, int, int);
template __global__ void prescan<false,false> (float*, const float*, float*, int, int, int);

template __global__ void prescanInt<true,true> (int*, const int*, int*, int, int, int);
template __global__ void prescanInt<true,false> (int*, const int*, int*, int, int, int);
template __global__ void prescanInt<false,true> (int*, const int*, int*, int, int, int);
template __global__ void prescanInt<false,false> (int*, const int*, int*, int, int, int);




#endif // #ifndef _SCAN_BEST_KERNEL_CU_


