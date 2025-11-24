
#include <cutil.h>
#include <cstdlib>
#include <cstdio>
#include <string.h>

#if defined(__APPLE__) || defined(MACOSX)
	#include <GLUT/glut.h>
#else
	#include <GL/glut.h>
#endif
#include <cuda_gl_interop.h>

#include "fluid_system_kern.cu"

extern "C"
{

// Compute number of blocks to create
int iDivUp (int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}
void computeNumBlocks (int numPnts, int minThreads, int &numBlocks, int &numThreads)
{
    numThreads = min( minThreads, numPnts );
    numBlocks = iDivUp ( numPnts, numThreads );
}


void Grid_InsertParticlesCUDA ( uchar* data, uint stride, uint numPoints )
{
    int numThreads, numBlocks;
    computeNumBlocks (numPoints, 256, numBlocks, numThreads);

	// transfer point data to device
    char* pntData;
	size = numPoints * stride;
	cudaMalloc( (void**) &pntData, size);
	cudaMemcpy( pntData, data, size, cudaMemcpyHostToDevice);    

    // execute the kernel
    insertParticles<<< numBlocks, numThreads >>> ( pntData, stride );
    
    // transfer data back to host
    cudaMemcpy( data, pntData, cudaMemcpyDeviceToHost);
    
    // check if kernel invocation generated an error
    CUT_CHECK_ERROR("Kernel execution failed");
    CUDA_SAFE_CALL(cudaGLUnmapBufferObject(vboPos));
}