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

	Fluid* p = fbuf.pnt(i);

	gcf = (p->pos - gridMin) * gridDelta; 
	gc = make_int3( int(gcf.x), int(gcf.y), int(gcf.z) );
	gs = (gc.y * gridRes.z + gc.z)*gridRes.x + gc.x;

	if ( gc.x >= 1 && gc.x <= gridScan.x && gc.y >= 1 && gc.y <= gridScan.y && gc.z >= 1 && gc.z <= gridScan.z ) {
		p->gcell = gs;												// Grid cell insert.
		p->gndx = atomicAdd ( &fbuf.bufI(FGRIDCNT)[ gs ], 1 );		// Grid counts.
	} else {
		p->gcell = GRID_UNDEF;		
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

	Fluid* t = fbuf.ptemp(i);

	if ( t->gcell != GRID_UNDEF ) {	  
		// Determine the sort_ndx; location of the particle after sort	
	    int sort_ndx = fbuf.bufI(FGRIDOFF)[ t->gcell ] + t->gndx ;	// global_ndx = grid_cell_offet + particle_offset	
		
		// Transfer data to sort location		
		Fluid* p = fbuf.pnt( sort_ndx );		
		p->pos = t->pos;
		p->vel = t->vel;
		p->veval = t->veval;
		p->force = t->force;
		p->press = t->press;
		p->clr = t->clr;		
		p->gndx = t->gndx;
		p->gcell = t->gcell;

		fbuf.bufI(FGRID)[ sort_ndx ] = sort_ndx;				// full sort, grid indexing becomes identity			
	} 
} 

extern "C" __device__ float contributePressure ( Fluid* pi, int cell )
{			
	if ( fbuf.bufI(FGRIDCNT)[cell] == 0 ) return 0.0;

	float dsq, sum = 0.0;
	float3 dist;	
	Fluid* pj;

	for ( int cndx = fbuf.bufI(FGRIDOFF)[cell]; cndx < fbuf.bufI(FGRIDOFF)[cell] + fbuf.bufI(FGRIDCNT)[cell]; cndx++ ) {
		pj = fbuf.pnt( fbuf.bufI(FGRID)[cndx] );
		dist = pi->pos - pj->pos;
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
			if ( dsq < fparam.rd2 && dsq > 0.0) {
				dsq = (fparam.rd2 - dsq) * fparam.d2;
				sum += dsq * dsq * dsq;				
			} 
	}	
	return sum;
}
			
extern "C" __global__ void computePressure ( int pnum )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= pnum ) return;

	float3 dist;
	float dsq;
	register int cell;

	// Get search cell	
	uint gc = fbuf.pnt(i)->gcell;
	if ( gc == GRID_UNDEF ) return;						// particle out-of-range
	gc -= (1*fparam.gridRes.z + 1)*fparam.gridRes.x + 1;

	// Sum pressures
	float sum = 0.0;
	float3 pos = fbuf.pnt(i)->pos;

	for (int c=0; c < fparam.gridAdjCnt; c++) {
		cell = gc + fparam.gridAdj[c];
		
		for ( int cndx = fbuf.bufI(FGRIDOFF)[cell]; cndx < fbuf.bufI(FGRIDOFF)[cell] + fbuf.bufI(FGRIDCNT)[cell]; cndx++ ) {			
			dist = pos - fbuf.pnt( fbuf.bufI(FGRID)[cndx]  )->pos;
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
	fbuf.pnt(i)->press = sum;
}

extern "C" __device__ float3 contributeForce ( Fluid* pi, float pressi, int cell)
{			
	if ( fbuf.bufI(FGRIDCNT)[cell] == 0 ) return make_float3(0,0,0);	

	float dsq, c, pterm;	
	float3 dist, force = make_float3(0,0,0);
	Fluid* pj;
	float pressj;

	for ( int cndx = fbuf.bufI(FGRIDOFF)[cell]; cndx < fbuf.bufI(FGRIDOFF)[cell] + fbuf.bufI(FGRIDCNT)[cell]; cndx++ ) {
		pj = fbuf.pnt( fbuf.bufI(FGRID)[cndx] );
		dist = pi->pos - pj->pos;
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);

		if ( dsq < fparam.rd2 && dsq > 0) {			
			dsq = sqrt(dsq * fparam.d2);
			c = ( fparam.psmoothradius - dsq ); 
			pressj = (pj->press - fparam.prest_dens ) * fparam.pintstiff;
			pterm = fparam.sim_scale * -0.5f * c * fparam.spikykern * ( pressi + pressj ) / dsq;			
			force += (pterm * dist + fparam.vterm * ( pj->veval - pi->veval)) * c / (pi->press * pj->press);
		}	
	}
	return force;
}


extern "C" __global__ void computeForce ( int pnum)
{			
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= pnum ) return;

	register int cell, j, cndx;
	register float3 force, dist;	
	register float pterm, dsq;	
	float pressi, pressj;
			
	// Get search cell	
	uint gc = fbuf.pnt(i)->gcell;
	if ( gc == GRID_UNDEF ) return;						// particle out-of-range
	gc -= (1*fparam.gridRes.z + 1)*fparam.gridRes.x + 1;
	
	pressi = (fbuf.pnt(i)->press - fparam.prest_dens ) * fparam.pintstiff;
	
	// Sum forces
	force = make_float3(0,0,0);

	for ( int c=0; c < fparam.gridAdjCnt; c++) {
		cell = gc + fparam.gridAdj[c];				

		for ( int cndx = fbuf.bufI(FGRIDOFF)[cell]; cndx < fbuf.bufI(FGRIDOFF)[cell] + fbuf.bufI(FGRIDCNT)[cell]; cndx++ ) {			
			j = fbuf.bufI(FGRID)[cndx];			
			dist = fbuf.pnt(i)->pos - fbuf.pnt(j)->pos;
			dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
			if ( dsq < fparam.rd2 && dsq > 0) {			
				dsq = sqrt(dsq * fparam.d2);
				pressj = (fbuf.pnt(j)->press - fparam.prest_dens ) * fparam.pintstiff;
				pterm = fparam.sim_scale * -0.5f * (fparam.psmoothradius - dsq) * fparam.spikykern * ( pressi + pressj ) / dsq;			
				force += ( pterm * dist + fparam.vterm * ( fbuf.pnt(j)->veval - fbuf.pnt(i)->veval)) * (fparam.psmoothradius - dsq) / (fbuf.pnt(i)->press * fbuf.pnt(j)->press);
			}	
		}
		//--- not used (function call is slow, uses too many registers)
		// force += contributeForce ( i, fbuf.bufF3(FPOS)[ i ], fbuf.bufF3(FVEVAL)[ i ], 1/fbuf.bufF(FPRESS)[ i ], (fbuf.bufF(FPRESS)[i] - fparam.prest_dens ) * fparam.pintstiff, gc + fparam.gridAdj[c] );
	}
	fbuf.pnt(i)->force = force;
}

__device__ uint getGridCell ( float3 pos, uint3& gc )
{	
	gc.x = (int)( (pos.x - fparam.gridMin.x) * fparam.gridDelta.x);			// Cell in which particle is located
	gc.y = (int)( (pos.y - fparam.gridMin.y) * fparam.gridDelta.y);
	gc.z = (int)( (pos.z - fparam.gridMin.z) * fparam.gridDelta.z);		
	return (int) ( (gc.y*fparam.gridRes.z + gc.z)*fparam.gridRes.x + gc.x);	
}

		
extern "C" __global__ void advanceParticles ( float time, float dt, float ss, int numPnts )
{		
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= numPnts ) return;

	Fluid* p = fbuf.pnt(i);

	if ( p->gcell == GRID_UNDEF ) {
		p->pos = make_float3(-1000,-1000,-1000);
		p->vel = make_float3(0,0,0);
		return;
	}			
	// Get particle vars
	register float3 accel, norm;
	register float diff, adj, speed;	
	float3 fric;

	// Leapfrog integration						
	accel = p->force;
	accel *= fparam.pmass;	
		
	// Boundaries
	// Y-axis	
	diff = fparam.pradius - (p->pos.y - (fparam.bound_min.y + (p->pos.x-fparam.bound_min.x)*fparam.bound_slope )) * ss;
	if ( diff > EPSILON ) {
		norm = make_float3( -fparam.bound_slope, 1.0 - fparam.bound_slope, 0);
		adj = fparam.bound_stiff * diff - fparam.bound_damp * dot(norm, p->veval );		
		norm *= adj; accel += norm - p->veval * fparam.bound_friction;
	}

	diff = fparam.pradius - ( fparam.bound_max.y - p->pos.y )*ss;
	if ( diff > EPSILON ) {
		norm = make_float3(0, -1, 0);
		adj = fparam.bound_stiff * diff - fparam.bound_damp * dot(norm, p->veval );
		norm *= adj; accel += norm - p->veval * fparam.bound_friction;
	}

	// X-axis
	float wall = (sin(time*fparam.bound_wall_freq + p->pos.z/200.f)+1) * 0.5 * fparam.bound_wall_force;
	diff = fparam.pradius - (p->pos.x - (fparam.bound_min.x + wall) )*ss;
	if ( diff > EPSILON ) {
		norm = make_float3( 1, 0, 0);
		adj = wall * fparam.bound_stiff * diff - fparam.bound_damp * dot(norm, p->veval );
		norm *= adj; accel += norm;
	}
	diff = fparam.pradius - ( fparam.bound_max.x - p->pos.x)*ss;
	if ( diff > EPSILON ) {
		norm = make_float3(-1, 0, 0);
		adj = fparam.bound_stiff * diff - fparam.bound_damp * dot(norm, p->veval );
		norm *= adj; accel += norm;
	}

	// Z-axis
	diff = fparam.pradius - (p->pos.z - fparam.bound_min.z ) * ss;
	if ( diff > EPSILON ) {
		norm = make_float3( 0, 0, 1 );
		adj = fparam.bound_stiff * diff - fparam.bound_damp * dot(norm, p->veval );
		norm *= adj; accel += norm;
	}
	diff = fparam.pradius - ( fparam.bound_max.z - p->pos.z )*ss;
	if ( diff > EPSILON ) {
		norm = make_float3( 0, 0, -1 );
		adj = fparam.bound_stiff * diff - fparam.bound_damp * dot(norm, p->veval );
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
	speed = p->vel.x * p->vel.x + p->vel.y * p->vel.y + p->vel.z * p->vel.z;
	if ( speed > fparam.VL2 ) {		
		p->vel *= fparam.VL / sqrt(speed);
	}

	// Leap-frog Integration
	float3 vnext = accel*dt + p->vel;					// v(t+1/2) = v(t-1/2) + a(t) dt		
	p->veval = (p->vel + vnext) * 0.5;	// v(t+1) = [v(t-1/2) + v(t+1/2)] * 0.5			
	p->vel = vnext;
	p->pos += vnext * (dt/ss);			// p(t+1) = p(t) + v(t+1/2) dt		
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

