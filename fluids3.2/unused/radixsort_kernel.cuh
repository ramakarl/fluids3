

#ifndef _RADIXSORT_KERNEL_CUH_
#define _RADIXSORT_KERNEL_CUH_

#include <stdio.h>
#include "radixsort.cuh"

extern const int NUM_SMS;
extern const int NUM_THREADS_PER_SM ;
extern const int NUM_THREADS_PER_BLOCK;
extern const int NUM_BLOCKS;
extern const int RADIX;
extern const int RADICES;
extern const int RADIXMASK;
#if SIXTEEN
extern const int RADIXBITS;
#else
extern const int RADIXBITS;
#endif
extern const int RADIXTHREADS;
extern const int RADIXGROUPS;
extern const int TOTALRADIXGROUPS;
extern const int SORTRADIXGROUPS;
extern const int GRFELEMENTS;
extern const int GRFSIZE;

// Prefix sum variables
extern  const int PREFIX_NUM_THREADS_PER_SM;
extern  const int PREFIX_NUM_THREADS_PER_BLOCK;
extern  const int PREFIX_NUM_BLOCKS;
extern  const int PREFIX_BLOCKSIZE;
extern  const int PREFIX_GRFELEMENTS ;
extern  const int PREFIX_GRFSIZE;

// Shuffle variables
extern  const int SHUFFLE_GRFOFFSET;
extern  const int SHUFFLE_GRFELEMENTS ;
extern  const int SHUFFLE_GRFSIZE;

extern __global__ void RadixSum(KeyValuePair *pData, uint elements, uint elements_rounded_to_3072, uint shift);
extern __global__ void RadixPrefixSum();
extern __global__ void RadixAddOffsetsAndShuffle(KeyValuePair* pSrc, KeyValuePair* pDst, uint elements, uint elements_rounded_to_3072, int shift);


#endif // #ifndef _RADIXSORT_KERNEL_CUH_
