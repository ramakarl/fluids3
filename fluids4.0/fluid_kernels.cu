//-----------------------------------------------------------------------------
// FLUIDS v.3.1 - SPH Fluid Simulator for CPU and GPU
// Copyright (C) 2012-2013, 2021. Rama Hoetzlein, http://fluids3.com
//-----------------------------------------------------------------------------

#define CUDA_KERNEL
#include "fluid_kernels.cuh"

#include "cutil_math.h"			// cutil32.lib
#include <string.h>
#include <assert.h>
#include <curand.h>
#include <curand_kernel.h>

__constant__ FParams		fparam;			// CPU Fluid params
__constant__ FBufs			fbuf;			// GPU Particle buffers (unsorted)
__constant__ FBufs			ftemp;			// GPU Particle buffers (sorted)
__constant__ uint			gridActive;

#define SCAN_BLOCKSIZE		512

extern "C" __global__ void insertParticles ( int pnum )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= pnum ) return;

	//-- debugging (pointers should match CUdeviceptrs on host side)
	// printf ( " pos: %012llx, gcell: %012llx, gndx: %012llx, gridcnt: %012llx\n", fbuf.bufC(FPOS), fbuf.bufC(FGCELL), fbuf.bufC(FGNDX), fbuf.bufC(FGRIDCNT) );

	register float3 gridMin =	fparam.gridMin;
	register float3 gridDelta = fparam.gridDelta;
	register int3 gridRes =		fparam.gridRes;
	register int3 gridScan =	fparam.gridScanMax;

	register int		gs;
	register float3		gcf;
	register int3		gc;	

	gcf = (fbuf.bufF3(FPOS)[i] - gridMin) * gridDelta; 
	gc = make_int3( int(gcf.x), int(gcf.y), int(gcf.z) );
	gs = (gc.y * gridRes.z + gc.z)*gridRes.x + gc.x;

	if ( gc.x >= 1 && gc.x <= gridScan.x && gc.y >= 1 && gc.y <= gridScan.y && gc.z >= 1 && gc.z <= gridScan.z ) {
		fbuf.bufI(FGCELL)[i] = gs;											// Grid cell insert.
		fbuf.bufI(FGNDX)[i] = atomicAdd ( &fbuf.bufI(FGRIDCNT)[ gs ], 1 );		// Grid counts.

		//gcf = (-make_float3(poff,poff,poff) + fbuf.bufF3(FPOS)[i] - gridMin) * gridDelta;
		//gc = make_int3( int(gcf.x), int(gcf.y), int(gcf.z) );
		//gs = ( gc.y * gridRes.z + gc.z)*gridRes.x + gc.x;
	} else {
		fbuf.bufI(FGCELL)[i] = GRID_UNDEF;		
	}
}


extern "C" __global__ void countingSortFull ( int pnum )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;		// particle index				
	if ( i >= pnum ) return;

	// This algorithm is O(2NK) in space, O(N/P) time, where K=sizeof(Fluid)
	// Copy particle from original, unsorted buffer (msortbuf),
	// into sorted memory location on device (mpos/mvel). 
	
	// **NOTE** We cannot use shared memory for temporary storage since there is
	// no synchronization across blocks. 

	uint icell = ftemp.bufI(FGCELL) [ i ];	

	if ( icell != GRID_UNDEF ) {	  
		// Determine the sort_ndx; location of the particle after sort		
		uint indx =  ftemp.bufI(FGNDX)  [ i ];		
	    int sort_ndx = fbuf.bufI(FGRIDOFF) [ icell ] + indx ;	// global_ndx = grid_cell_offet + particle_offset	
		
		// Transfer data to sort location		
		fbuf.bufF3(FPOS) [sort_ndx] =		ftemp.bufF3(FPOS) [i];
		fbuf.bufF3(FVEL) [sort_ndx] =		ftemp.bufF3(FVEL) [i];
		fbuf.bufF3(FVEVAL)[sort_ndx] =		ftemp.bufF3(FVEVAL) [i];
		fbuf.bufF3(FFORCE)[sort_ndx] =		ftemp.bufF3(FFORCE) [i];
		fbuf.bufF (FPRESS)[sort_ndx] =		ftemp.bufF(FPRESS) [i];
		fbuf.bufI (FCLR) [sort_ndx] =		ftemp.bufI(FCLR) [i];

		fbuf.bufI (FGRID) [ sort_ndx ] =	sort_ndx;			// full sort, grid indexing becomes identity		
		fbuf.bufI (FGCELL) [sort_ndx] =		icell;
		fbuf.bufI (FGNDX) [sort_ndx] =		indx;		
	}
} 

extern "C" __device__ float contributePressure ( int i, float3 p, int cell )
{			
	if ( fbuf.bufI(FGRIDCNT)[cell] == 0 ) return 0.0;

	float3 dist;
	float dsq, sum = 0.0;
	register float d2 = fparam.sim_scale * fparam.sim_scale;
	register float r2 = fparam.r2 / d2;
	
	int clast = fbuf.bufI(FGRIDOFF)[cell] + fbuf.bufI(FGRIDCNT)[cell];

	for ( int cndx = fbuf.bufI(FGRIDOFF)[cell]; cndx < clast; cndx++ ) {
		int pndx = fbuf.bufI(FGRID) [cndx];
		dist = p - fbuf.bufF3(FPOS) [pndx];
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
	uint gc = fbuf.bufI(FGCELL) [i];
	if ( gc == GRID_UNDEF ) return;						// particle out-of-range
	gc -= (1*fparam.gridRes.z + 1)*fparam.gridRes.x + 1;

	float3 dist;
	float dsq, sum = 0.0;
	register int cell;

	// Sum Pressures
	float3 pos = fbuf.bufF3(FPOS) [i];

	for (int c=0; c < fparam.gridAdjCnt; c++) {
		cell = gc + fparam.gridAdj[c];
		int clast = fbuf.bufI(FGRIDOFF)[cell] + fbuf.bufI(FGRIDCNT)[cell];
		for ( int cndx = fbuf.bufI(FGRIDOFF)[cell]; cndx < clast; cndx++ ) {			
			dist = pos - fbuf.bufF3(FPOS) [ fbuf.bufI(FGRID)[cndx] ];
			dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
			if ( dsq < fparam.rd2 && dsq > 0.0) {
				dsq = (fparam.rd2 - dsq) * fparam.d2;
				sum += dsq * dsq * dsq;				
			} 
		}			
		//--- not used (function call is slow, uses too many registers)
		// sum += contributePressure ( i, pos, gc + fparam.gridAdj[c] );
	}
	__syncthreads();
		
	// Compute Density & Pressure
	sum = sum * fparam.pmass * fparam.poly6kern;
	if ( sum == 0.0 ) sum = 1.0;	
	fbuf.bufF(FPRESS) [ i ] = sum;
}

extern "C" __device__ float3 contributeForce ( int i, float3 ipos, float3 iveleval, float di, float pi, int cell)
{			
	if ( fbuf.bufI(FGRIDCNT)[cell] == 0 ) return make_float3(0,0,0);	

	float dsq, c, pterm;	
	float3 dist, force = make_float3(0,0,0);
	float pj;
	int j;

	int clast = fbuf.bufI(FGRIDOFF)[cell] + fbuf.bufI(FGRIDCNT)[cell];

	for ( int cndx = fbuf.bufI(FGRIDOFF)[cell]; cndx < clast; cndx++ ) {
		
		j = fbuf.bufI(FGRID)[ cndx ];				
		dist = ( ipos - fbuf.bufF3(FPOS)[ j ] );		// dist in cm
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);

		if ( dsq < fparam.rd2 && dsq > 0) {			
			dsq = sqrt(dsq * fparam.d2);
			c = ( fparam.psmoothradius - dsq ); 
			pj = (fbuf.bufF(FPRESS)[j] - fparam.prest_dens ) * fparam.pintstiff;
			pterm = fparam.sim_scale * -0.5f * c * fparam.spikykern * ( pi + pj ) / dsq;			
			force += ( pterm * dist + fparam.vterm * ( fbuf.bufF3(FVEVAL)[ j ] - iveleval )) * c / (di * fbuf.bufF(FPRESS)[j]);
		}	
	}
	return force;
}


extern "C" __global__ void computeForce ( int pnum)
{			
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= pnum ) return;

	// Get search cell	
	uint gc = fbuf.bufI(FGCELL)[ i ];
	if ( gc == GRID_UNDEF ) return;						// particle out-of-range
	gc -= (1*fparam.gridRes.z + 1)*fparam.gridRes.x + 1;

	// Sum Pressures	
	register int cell, c, j, cndx;
	register float3 force, dist;	
	register float pterm, dsq;
	float pi, pj;

	force = make_float3(0,0,0);			

	for ( c=0; c < fparam.gridAdjCnt; c++) {
		cell = gc + fparam.gridAdj[c];		

		for ( cndx = fbuf.bufI(FGRIDOFF)[cell]; cndx < fbuf.bufI(FGRIDOFF)[cell] + fbuf.bufI(FGRIDCNT)[cell]; cndx++ ) {
			j = fbuf.bufI(FGRID)[ cndx ];				
			dist = ( fbuf.bufF3(FPOS)[i] - fbuf.bufF3(FPOS)[ j ] );		// dist in cm
			dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
			if ( dsq < fparam.rd2 && dsq > 0) {			
				dsq = sqrt(dsq * fparam.d2);			
				pi = (fbuf.bufF(FPRESS)[i] - fparam.prest_dens ) * fparam.pintstiff;
				pj = (fbuf.bufF(FPRESS)[j] - fparam.prest_dens ) * fparam.pintstiff;
				pterm = fparam.sim_scale * -0.5f * (fparam.psmoothradius-dsq) * fparam.spikykern * ( pi + pj ) / dsq;			
				force += ( pterm * dist + fparam.vterm * ( fbuf.bufF3(FVEVAL)[ j ] - fbuf.bufF3(FVEVAL)[i] )) * (fparam.psmoothradius-dsq) / (fbuf.bufF(FPRESS)[i] * fbuf.bufF(FPRESS)[ j ] );
			}	
		}
		//--- not used (function call is slow, uses too many registers)
		// force += contributeForce ( i, fbuf.bufF3(FPOS)[ i ], fbuf.bufF3(FVEVAL)[ i ], 1/fbuf.bufF(FPRESS)[ i ], (fbuf.bufF(FPRESS)[i] - fparam.prest_dens ) * fparam.pintstiff, gc + fparam.gridAdj[c] );
	}
	fbuf.bufF3(FFORCE)[ i ] = force;
}

extern "C" __global__ void randomInit ( int seed, int numPnts )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= numPnts ) return;

	// Initialize particle random generator	
	curandState_t* st = (curandState_t*) (fbuf.bufC(FSTATE) + i*sizeof(curandState_t));
	curand_init ( seed + i, 0, 0, st );		
}

#define CURANDMAX		2147483647

extern "C" __global__ void emitParticles ( float frame, int emit, int numPnts )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= emit ) return;

	curandState_t* st = (curandState_t*) (fbuf.bufC(FSTATE) + i*sizeof(curandState_t));
	uint v = curand( st);
	uint j = v & (numPnts-1);
	float3 bmin = make_float3(-170,10,-20);
	float3 bmax = make_float3(-190,60, 20);

	float3 pos = make_float3(0,0,0);	
	pos.x = float( v & 0xFF ) / 256.0;
	pos.y = float((v>>8) & 0xFF ) / 256.0;
	pos.z = float((v>>16) & 0xFF ) / 256.0;
	pos = bmin + pos*(bmax-bmin);	
	
	fbuf.bufF3(FPOS)[j] = pos;
	fbuf.bufF3(FVEVAL)[j] = make_float3(0,0,0);
	fbuf.bufF3(FVEL)[j] = make_float3(5,-2,0);
	fbuf.bufF3(FFORCE)[j] = make_float3(0,0,0);	
	
}

__device__ uint getGridCell ( float3 pos, uint3& gc )
{	
	gc.x = (int)( (pos.x - fparam.gridMin.x) * fparam.gridDelta.x);			// Cell in which particle is located
	gc.y = (int)( (pos.y - fparam.gridMin.y) * fparam.gridDelta.y);
	gc.z = (int)( (pos.z - fparam.gridMin.z) * fparam.gridDelta.z);		
	return (int) ( (gc.y*fparam.gridRes.z + gc.z)*fparam.gridRes.x + gc.x);	
}

extern "C" __global__ void sampleParticles ( float* brick, uint3 res, float3 bmin, float3 bmax, int numPnts, float scalar )
{
	float3 dist;
	float dsq;
	int j, cell;	
	register float r2 = fparam.r2;
	register float h2 = 2.0*r2 / 8.0;		// 8.0=smoothing. higher values are sharper

	uint3 i = blockIdx * make_uint3(blockDim.x, blockDim.y, blockDim.z) + threadIdx;
	if ( i.x >= res.x || i.y >= res.y || i.z >= res.z ) return;
	
	float3 p = bmin + make_float3(float(i.x)/res.x, float(i.y)/res.y, float(i.z)/res.z) * (bmax-bmin);
	//float3 v = make_float3(0,0,0);
	float v = 0.0;

	// Get search cell
	int nadj = (1*fparam.gridRes.z + 1)*fparam.gridRes.x + 1;
	uint3 gc;
	uint gs = getGridCell ( p, gc );
	if ( gc.x < 1 || gc.x > fparam.gridRes.x-fparam.gridSrch || gc.y < 1 || gc.y > fparam.gridRes.y-fparam.gridSrch || gc.z < 1 || gc.z > fparam.gridRes.z-fparam.gridSrch ) {
		brick[ (i.y*int(res.z) + i.z)*int(res.x) + i.x ] = 0.0;
		return;
	}

	gs -= nadj;	

	for (int c=0; c < fparam.gridAdjCnt; c++) {
		cell = gs + fparam.gridAdj[c];		
		if ( fbuf.bufI(FGRIDCNT)[cell] != 0 ) {				
			for ( int cndx = fbuf.bufI(FGRIDOFF)[cell]; cndx < fbuf.bufI(FGRIDOFF)[cell] + fbuf.bufI(FGRIDCNT)[cell]; cndx++ ) {
				j = fbuf.bufI(FGRID)[cndx];
				dist = p - fbuf.bufF3(FPOS)[ j ];
				dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
				if ( dsq < fparam.rd2 && dsq > 0 ) {
					dsq = sqrt(dsq * fparam.d2);					
					//v += fbuf.mvel[j] * (fparam.gausskern * exp ( -(dsq*dsq)/h2 ) / fbuf.mdensity[ j ]);
					v += fparam.gausskern * exp ( -(dsq*dsq)/h2 );
				}
			}
		}
	}
	__syncthreads();

	brick[ (i.z*int(res.y) + i.y)*int(res.x) + i.x ] = v * scalar;
	//brick[ (i.z*int(res.y) + i.y)*int(res.x) + i.x ] = length(v) * scalar;
}

extern "C" __global__ void computeQuery ( int pnum )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= pnum ) return;

	// Get search cell
	int nadj = (1*fparam.gridRes.z + 1)*fparam.gridRes.x + 1;
	uint gc = fbuf.bufI(FGCELL) [i];
	if ( gc == GRID_UNDEF ) return;						// particle out-of-range
	gc -= nadj;

	// Sum Pressures
	float sum = 0.0;
	for (int c=0; c < fparam.gridAdjCnt; c++) {
		sum += 1.0;
	}
	__syncthreads();
	
}

#define maxf(a,b)  (a>b ? a : b)
		
extern "C" __global__ void advanceParticles ( float time, float dt, float ss, int numPnts )
{		
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= numPnts ) return;
	
	if ( fbuf.bufI(FGCELL)[i] == GRID_UNDEF ) {
		fbuf.bufF3(FPOS)[i] = make_float3(-1000,-1000,-1000);
		fbuf.bufF3(FVEL)[i] = make_float3(0,0,0);
		return;
	}
			
	// Get particle vars
	register float3 accel, norm;
	register float diff, adj, speed;
	register float3 pos = fbuf.bufF3(FPOS)[i];
	register float3 veval = fbuf.bufF3(FVEVAL)[i];
	float3 fric;

	// Leapfrog integration						
	accel = fbuf.bufF3(FFORCE)[i];
	accel *= fparam.pmass;	
		
	// Boundaries
	// Y-axis	
	diff = fparam.pradius - (pos.y - (fparam.bound_min.y + (pos.x-fparam.bound_min.x)*fparam.bound_slope )) * ss;
	if ( diff > EPSILON ) {
		norm = make_float3( -fparam.bound_slope, 1.0 - fparam.bound_slope, 0);
		adj = fparam.bound_stiff * diff - fparam.bound_damp * dot(norm, veval );		
		norm *= adj; accel += norm - veval * fparam.bound_friction;
	}

	diff = fparam.pradius - ( fparam.bound_max.y - pos.y )*ss;
	if ( diff > EPSILON ) {
		norm = make_float3(0, -1, 0);
		adj = fparam.bound_stiff * diff - fparam.bound_damp * dot(norm, veval );
		norm *= adj; accel += norm - veval * fparam.bound_friction;
	}

	// X-axis
	float wall = (sin(time*fparam.bound_wall_freq + pos.z/400.f)*0.5+0.5) * fparam.bound_wall_force;
	diff = fparam.pradius - (pos.x - (fparam.bound_min.x + wall) )*ss;
	if ( diff > EPSILON ) {
		norm = make_float3( 1, 0, 0);
		adj = wall * fparam.bound_stiff * diff - fparam.bound_damp * dot(norm, veval );
		norm *= adj; accel += norm;
	}
	diff = fparam.pradius - ( fparam.bound_max.x - pos.x)*ss;
	if ( diff > EPSILON ) {
		norm = make_float3(-1, 0, 0);
		adj = fparam.bound_stiff * diff - fparam.bound_damp * dot(norm, veval );
		norm *= adj; accel += norm;
	}

	// Z-axis
	diff = fparam.pradius - (pos.z - fparam.bound_min.z ) * ss;
	if ( diff > EPSILON ) {
		norm = make_float3( 0, 0, 1 );
		adj = fparam.bound_stiff * diff - fparam.bound_damp * dot(norm, veval );
		norm *= adj; accel += norm;
	}
	diff = fparam.pradius - ( fparam.bound_max.z - pos.z )*ss;
	if ( diff > EPSILON ) {
		norm = make_float3( 0, 0, -1 );
		adj = fparam.bound_stiff * diff - fparam.bound_damp * dot(norm, veval );
		norm *= adj; accel += norm;
	}
		
	// Gravity
	accel += fparam.gravity;

	// Accel Limit
	speed = accel.x*accel.x + accel.y*accel.y + accel.z*accel.z;
	if ( speed > fparam.AL2 ) {
		accel *= fparam.AL / sqrt(speed);
	}

	// Velocity Limit
	float3 vel = fbuf.bufF3(FVEL)[i];
	speed = vel.x*vel.x + vel.y*vel.y + vel.z*vel.z;
	if ( speed > fparam.VL2 ) {		
		vel *= fparam.VL / sqrt(speed);
	}

	// Leap-frog Integration
	float3 vnext = accel*dt + vel;					// v(t+1/2) = v(t-1/2) + a(t) dt		
	fbuf.bufF3(FVEVAL)[i] = (vel + vnext) * 0.5;	// v(t+1) = [v(t-1/2) + v(t+1/2)] * 0.5			
	fbuf.bufF3(FVEL)[i] = vnext;
	fbuf.bufF3(FPOS)[i] += vnext * (dt/ss);			// p(t+1) = p(t) + v(t+1/2) dt		
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

