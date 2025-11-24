//-----------------------------------------------------------------------------
// FLUIDS v5.0 - SPH Fluid Simulator for CPU and GPU
// Copyright (C) 2012-2013, 2021. Rama Hoetzlein, http://fluids3.com
//-----------------------------------------------------------------------------

#define CUDA_KERNEL
#include "particles.cuh"

#include "cutil_math.h"			// cutil32.lib
#include <string.h>
#include <assert.h>
#include <curand.h>
#include <curand_kernel.h>

#include "datax.h"

__constant__ FParams_t	FParams;		
__constant__ cuDataX	FPnts;
__constant__ cuDataX	FPntTmp;
__constant__ cuDataX	FAccel;

#define SCAN_BLOCKSIZE		512

extern "C" __global__ void insertParticles ( int pnum )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= pnum ) return;
	
	register float3		p;
	register float3		gcf;
	register int3		gc;	
	register int		gs;	

	p = FPnts.bufF3(FPOS)[i];
	gcf = (p - FParams.gridMin) * FParams.gridDelta; 
	gc = make_int3( int(gcf.x), int(gcf.y), int(gcf.z) );
	gs = (gc.y * FParams.gridRes.z + gc.z) * FParams.gridRes.x + gc.x;

	if ( gc.x >= 1 && gc.x <= FParams.gridScanMax.x && gc.y >= 1 && gc.y <= FParams.gridScanMax.y && gc.z >= 1 && gc.z <= FParams.gridScanMax.z ) {
		FPnts.bufI(FGCELL)[i] = gs;													// Grid cell insert.
		FPnts.bufI(FGNDX)[i] = atomicAdd ( &FAccel.bufI(AGRIDCNT)[ gs ], 1 );		// Grid counts.
	} else {
		FPnts.bufI(FGCELL)[i] = GRID_UNDEF;		
	}

	//--- debugging
	/*if ( i==0 ) {
		printf ( "FPnts:FPOS: %012llx\n", FPnts.bufF3(FPOS) );
		printf ( "gridRes: %d, %d, %d\n", FParams.gridRes.x, FParams.gridRes.y, FParams.gridRes.z );
		printf ( "gridScanMax: %d, %d, %d\n", FParams.gridScanMax.x, FParams.gridScanMax.y, FParams.gridScanMax.z );
		printf ( "gridDelta: %f, %f, %f\n", FParams.gridDelta.x, FParams.gridDelta.y, FParams.gridDelta.z );
		printf ( "pos: %f,%f,%f  gc: %d,%d,%d   gs: %d\n", p.x, p.y, p.z, gc.x, gc.y, gc.z, gs);
	}*/
}

// debugAccess - very useful function to show all GPU pointers
__device__ void debugAccess ()
{
	printf ( "--- gpu bufs\n" );
	for (int i=0; i < 8; i++)
		printf ( "%d: %012llx   %012llx\n", i, FPnts.bufI(i), FPntTmp.bufI(i) );	
}

extern "C" __global__ void countingSortFull ( int pnum )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;		// particle index				
	if ( i >= pnum ) return;

	// This algorithm is O(2NK) in space, O(N/P) time, where K=sizeof(Fluid)
	// Copy particle from original, unsorted buffer (msortbuf),
	// into sorted memory location on device (mpos/mvel). 
	
	// **NOTE** We cannot use shared memory for temporary storage since this is a 
	// global reordering and there is no synchronization across blocks. 

	int icell = FPntTmp.bufI(FGCELL) [ i ];	

	if ( icell != GRID_UNDEF ) {	  
		// Determine the sort_ndx; location of the particle after sort		
		int indx =  FPntTmp.bufI(FGNDX)  [ i ];		
	    int sort_ndx = FAccel.bufI(AGRIDOFF) [ icell ] + indx ;	// global_ndx = grid_cell_offet + particle_offset	
		
		// Transfer data to sort location		
		FPnts.bufF3(FPOS) [sort_ndx] =		FPntTmp.bufF3(FPOS) [i];				
		FPnts.bufF3(FVEL) [sort_ndx] =		FPntTmp.bufF3(FVEL) [i];		
		FPnts.bufI (FCLR) [sort_ndx] =		FPntTmp.bufI(FCLR) [i];
		FPnts.bufF3(FVEVAL)[sort_ndx] =		FPntTmp.bufF3(FVEVAL) [i];		
		FPnts.bufF (FPRESS)[sort_ndx] =		FPntTmp.bufF(FPRESS) [i];
		FPnts.bufF3(FFORCE)[sort_ndx] =		FPntTmp.bufF3(FFORCE) [i];	
		FPnts.bufI (FGCELL) [sort_ndx] =	icell;
		FPnts.bufI (FGNDX) [sort_ndx] =		indx; 		

		FAccel.bufI (AGRID) [ sort_ndx ] =	sort_ndx;			// full sort, grid indexing becomes identity				
	}
} 

extern "C" __device__ float contributePressure ( int i, float3 p, int cell )
{			
	if ( FAccel.bufI(AGRIDCNT)[cell] == 0 ) return 0.0;

	float3 dist;
	float dsq, sum = 0.0;
	register float d2 = FParams.sim_scale * FParams.sim_scale;
	register float r2 = FParams.r2 / d2;
	
	int clast = FAccel.bufI(AGRIDOFF)[cell] + FAccel.bufI(AGRIDCNT)[cell];

	for ( int cndx = FAccel.bufI(AGRIDOFF)[cell]; cndx < clast; cndx++ ) {
		int pndx = FAccel.bufI(AGRID) [cndx];
		dist = p - FPnts.bufF3(FPOS) [pndx];
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
		if ( dsq < r2 && dsq > 0.0) {
			dsq = (r2 - dsq)*d2;
			sum += dsq * dsq * dsq;				
		} 
	}	
	return sum;
}
			
extern "C" __global__ void computePressure ( int pnum )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= pnum ) return;

	// Get search cell	
	uint gc = FPnts.bufI(FGCELL) [i];
	if ( gc == GRID_UNDEF ) return;						// particle out-of-range
	gc -= (1*FParams.gridRes.z + 1)*FParams.gridRes.x + 1;

	float3 dist;
	float dsq, sum = 0.0;
	register int cell;

	// Sum Pressures
	float3 pos = FPnts.bufF3(FPOS) [i];

	for (int c=0; c < FParams.gridAdjCnt; c++) {
		cell = gc + FParams.gridAdj[c];
		int clast = FAccel.bufI(AGRIDOFF)[cell] + FAccel.bufI(AGRIDCNT)[cell];
		for ( int cndx = FAccel.bufI(AGRIDOFF)[cell]; cndx < clast; cndx++ ) {			
			dist = pos - FPnts.bufF3(FPOS) [ FAccel.bufI(AGRID)[cndx] ];
			dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
			if ( dsq < FParams.rd2 && dsq > 0.0) {
				dsq = (FParams.rd2 - dsq) * FParams.d2;
				sum += dsq * dsq * dsq;				
			} 
		}			
		//--- not used (function call is slow, uses too many registers)
		// sum += contributePressure ( i, pos, gc + FParams.gridAdj[c] );
	}
	__syncthreads();
		
	// Compute Density & Pressure
	sum = sum * FParams.pmass * FParams.poly6kern;
	if ( sum == 0.0 ) sum = 1.0;	
	FPnts.bufF(FPRESS) [ i ] = sum;
}

extern "C" __device__ float3 contributeForce ( int i, float3 ipos, float3 iveleval, float di, float pi, int cell)
{			
	if ( FAccel.bufI(AGRIDCNT)[cell] == 0 ) return make_float3(0,0,0);	

	float dsq, c, pterm;	
	float3 dist, force = make_float3(0,0,0);
	float pj;
	int j;

	int clast = FAccel.bufI(AGRIDOFF)[cell] + FAccel.bufI(AGRIDCNT)[cell];

	for ( int cndx = FAccel.bufI(AGRIDOFF)[cell]; cndx < clast; cndx++ ) {
		
		j = FAccel.bufI(AGRID)[ cndx ];				
		dist = ( ipos - FPnts.bufF3(FPOS)[ j ] );		// dist in cm
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);

		if ( dsq < FParams.rd2 && dsq > 0) {			
			dsq = sqrt(dsq * FParams.d2);
			c = ( FParams.psmoothradius - dsq ); 
			pj = (FPnts.bufF(FPRESS)[j] - FParams.prest_dens ) * FParams.pintstiff;
			pterm = FParams.sim_scale * -0.5f * c * FParams.spikykern * ( pi + pj ) / dsq;			
			force += ( pterm * dist + FParams.vterm * ( FPnts.bufF3(FVEVAL)[ j ] - iveleval )) * c / (di * FPnts.bufF(FPRESS)[j]);
		}	
	}
	return force;
}


extern "C" __global__ void computeForce ( int pnum)
{			
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= pnum ) return;

	// Get search cell	
	uint gc = FPnts.bufI(FGCELL)[ i ];
	if ( gc == GRID_UNDEF ) return;						// particle out-of-range
	gc -= (1*FParams.gridRes.z + 1)*FParams.gridRes.x + 1;

	// Sum Pressures	
	register int cell, c, j, cndx;
	register float3 force, dist;	
	register float pterm, dsq;
	float pi, pj;

	force = make_float3(0,0,0);			

	for ( c=0; c < FParams.gridAdjCnt; c++) {
		cell = gc + FParams.gridAdj[c];		

		for ( cndx = FAccel.bufI(AGRIDOFF)[cell]; cndx < FAccel.bufI(AGRIDOFF)[cell] + FAccel.bufI(AGRIDCNT)[cell]; cndx++ ) {
			j = FAccel.bufI(AGRID)[ cndx ];				
			dist = ( FPnts.bufF3(FPOS)[i] - FPnts.bufF3(FPOS)[ j ] );		// dist in cm
			dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
			if ( dsq < FParams.rd2 && dsq > 0) {			
				dsq = sqrt(dsq * FParams.d2);			
				pi = (FPnts.bufF(FPRESS)[i] - FParams.prest_dens ) * FParams.pintstiff;
				pj = (FPnts.bufF(FPRESS)[j] - FParams.prest_dens ) * FParams.pintstiff;
				pterm = FParams.sim_scale * -0.5f * (FParams.psmoothradius-dsq) * FParams.spikykern * ( pi + pj ) / dsq;			
				force += ( pterm * dist + FParams.vterm * ( FPnts.bufF3(FVEVAL)[ j ] - FPnts.bufF3(FVEVAL)[i] )) * (FParams.psmoothradius-dsq) / (FPnts.bufF(FPRESS)[i] * FPnts.bufF(FPRESS)[ j ] );
			}	
		}
		//--- not used (function call is slow, uses too many registers)
		// force += contributeForce ( i, FPnts.bufF3(FPOS)[ i ], FPnts.bufF3(FVEVAL)[ i ], 1/FPnts.bufF(FPRESS)[ i ], (FPnts.bufF(FPRESS)[i] - FParams.prest_dens ) * FParams.pintstiff, gc + FParams.gridAdj[c] );
	}
	FPnts.bufF3(FFORCE)[ i ] = force;
}

__device__ uint getGridCell ( float3 pos, uint3& gc )
{	
	gc.x = (int)( (pos.x - FParams.gridMin.x) * FParams.gridDelta.x);			// Cell in which particle is located
	gc.y = (int)( (pos.y - FParams.gridMin.y) * FParams.gridDelta.y);
	gc.z = (int)( (pos.z - FParams.gridMin.z) * FParams.gridDelta.z);		
	return (int) ( (gc.y*FParams.gridRes.z + gc.z)*FParams.gridRes.x + gc.x);	
}

#define maxf(a,b)  (a>b ? a : b)
		
extern "C" __global__ void advanceParticles ( float time, float dt, float ss, int numPnts )
{		
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= numPnts ) return;
	
	if ( FPnts.bufI(FGCELL)[i] == GRID_UNDEF ) {
		FPnts.bufF3(FPOS)[i] = make_float3(-1000,-1000,-1000);
		FPnts.bufF3(FVEL)[i] = make_float3(0,0,0);
		return;
	}
		
	// Get particle vars
	register float3 accel, norm;	
	register float3 pos = FPnts.bufF3(FPOS)[i];
	register float3 vel = FPnts.bufF3(FVEL)[i];
	register float3 veval = FPnts.bufF3(FVEVAL)[i];	
	register float diff, adj, speed;

	// Leapfrog integration						
	accel = FPnts.bufF3(FFORCE)[i] * FParams.pmass;
		
	// Boundaries
	// Y-axis	
	diff = FParams.pradius - (pos.y - (FParams.bound_min.y + (pos.x-FParams.bound_min.x)*FParams.bound_slope )) * ss;
	if ( diff > EPSILON ) {
		norm = make_float3( -FParams.bound_slope, 1.0 - FParams.bound_slope, 0);
		adj = FParams.bound_stiff * diff - FParams.bound_damp * dot(norm, veval );		
		norm *= adj; accel += norm - veval * FParams.bound_friction;
	}

	diff = FParams.pradius - ( FParams.bound_max.y - pos.y )*ss;
	if ( diff > EPSILON ) {
		norm = make_float3(0, -1, 0);
		adj = FParams.bound_stiff * diff - FParams.bound_damp * dot(norm, veval );
		norm *= adj; accel += norm - veval * FParams.bound_friction;
	}

	// X-axis
	float wall = (sin(time*FParams.bound_wall_freq + pos.z/400.f)*0.5+0.5) * FParams.bound_wall_force;
	diff = FParams.pradius - (pos.x - (FParams.bound_min.x + wall) )*ss;
	if ( diff > EPSILON ) {
		norm = make_float3( 1, 0, 0);
		adj = wall * FParams.bound_stiff * diff - FParams.bound_damp * dot(norm, veval );
		norm *= adj; accel += norm;
	}
	diff = FParams.pradius - ( FParams.bound_max.x - pos.x)*ss;
	if ( diff > EPSILON ) {
		norm = make_float3(-1, 0, 0);
		adj = FParams.bound_stiff * diff - FParams.bound_damp * dot(norm, veval );
		norm *= adj; accel += norm;
	}

	// Z-axis
	diff = FParams.pradius - (pos.z - FParams.bound_min.z ) * ss;
	if ( diff > EPSILON ) {
		norm = make_float3( 0, 0, 1 );
		adj = FParams.bound_stiff * diff - FParams.bound_damp * dot(norm, veval );
		norm *= adj; accel += norm;
	}
	diff = FParams.pradius - ( FParams.bound_max.z - pos.z )*ss;
	if ( diff > EPSILON ) {
		norm = make_float3( 0, 0, -1 );
		adj = FParams.bound_stiff * diff - FParams.bound_damp * dot(norm, veval );
		norm *= adj; accel += norm;
	}
		
	// Gravity
	accel += FParams.gravity;

	// Accel Limit
	speed = accel.x*accel.x + accel.y*accel.y + accel.z*accel.z;
	if ( speed > FParams.AL2 ) {
		accel *= FParams.AL / sqrt(speed);
	}
	// Velocity Limit	
	speed = vel.x*vel.x + vel.y*vel.y + vel.z*vel.z;
	if ( speed > FParams.VL2 ) {		
		vel *= FParams.VL / sqrt(speed);
	}

	// Leap-frog Integration
	float3 vnext = accel*dt + vel;					// v(t+1/2) = v(t-1/2) + a(t) dt		
	FPnts.bufF3(FVEVAL)[i] = (vel + vnext) * 0.5;	// v(t+1) = [v(t-1/2) + v(t+1/2)] * 0.5			
	FPnts.bufF3(FVEL)[i] = vnext;
	FPnts.bufF3(FPOS)[i] += vnext * (dt/ss);			// p(t+1) = p(t) + v(t+1/2) dt		
}


extern "C" __global__ void prefixFixup(uint *input, uint *aux, int len)
{
	unsigned int t = threadIdx.x;
	unsigned int start = t + 2 * blockIdx.x * SCAN_BLOCKSIZE;
	if (start < len)					input[start] += aux[blockIdx.x];
	if (start + SCAN_BLOCKSIZE < len)   input[start + SCAN_BLOCKSIZE] += aux[blockIdx.x];
}

extern "C" __global__ void prefixSum(uint* input, uint* output, uint* aux, int len, int zeroff)
{
	__shared__ uint scan_array[SCAN_BLOCKSIZE << 1];
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

