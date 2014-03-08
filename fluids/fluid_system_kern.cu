/*
  FLUIDS v.3 - SPH Fluid Simulator for CPU and GPU
  Copyright (C) 2012-2013. Rama Hoetzlein, http://fluids3.com

  Attribute-ZLib license (* See additional part 4)

  This software is provided 'as-is', without any express or implied
  warranty. In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
  4. Any published work based on this code must include public acknowledgement
     of the origin. This includes following when applicable:
	   - Journal/Paper publications. Credited by reference to work in text & citation.
	   - Public presentations. Credited in at least one slide.
	   - Distributed Games/Apps. Credited as single line in game or app credit page.	 
	 Retaining this additional license term is required in derivative works.
	 Acknowledgement may be provided as:
	   Publication version:  
	      2012-2013, Hoetzlein, Rama C. Fluids v.3 - A Large-Scale, Open Source
	 	  Fluid Simulator. Published online at: http://fluids3.com
	   Single line (slides or app credits):
	      GPU Fluids: Rama C. Hoetzlein (Fluids v3 2013)

 Notes on Clause 4:
  The intent of this clause is public attribution for this contribution, not code use restriction. 
  Both commerical and open source projects may redistribute and reuse without code release.
  However, clause #1 of ZLib indicates that "you must not claim that you wrote the original software". 
  Clause #4 makes this more specific by requiring public acknowledgement to be extended to 
  derivative licenses. 

*/

#define CUDA_KERNEL
#include "fluid_system_kern.cuh"

#include "cutil_math.h"

#include "radixsort.cu"						// Build in RadixSort

__constant__ FluidParams		simData;
__constant__ uint				gridActive;

__global__ void insertParticles ( bufList buf, int pnum )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= pnum ) return;

	register float3 gridMin = simData.gridMin;
	register float3 gridDelta = simData.gridDelta;
	register int3 gridRes = simData.gridRes;
	register int3 gridScan = simData.gridScanMax;
	register float poff = simData.psmoothradius / simData.psimscale;

	register int		gs;
	register float3		gcf;
	register int3		gc;

	gcf = (buf.mpos[i] - gridMin) * gridDelta; 
	gc = make_int3( int(gcf.x), int(gcf.y), int(gcf.z) );
	gs = (gc.y * gridRes.z + gc.z)*gridRes.x + gc.x;
	if ( gc.x >= 1 && gc.x <= gridScan.x && gc.y >= 1 && gc.y <= gridScan.y && gc.z >= 1 && gc.z <= gridScan.z ) {
		buf.mgcell[i] = gs;											// Grid cell insert.
		buf.mgndx[i] = atomicAdd ( &buf.mgridcnt[ gs ], 1 );		// Grid counts.

		gcf = (-make_float3(poff,poff,poff) + buf.mpos[i] - gridMin) * gridDelta;
		gc = make_int3( int(gcf.x), int(gcf.y), int(gcf.z) );
		gs = ( gc.y * gridRes.z + gc.z)*gridRes.x + gc.x;		
	} else {
		buf.mgcell[i] = GRID_UNDEF;		
	}
}

// the mutex variable
__device__ int g_mutex = 0;

// GPU simple synchronization function
__device__ void __gpu_sync(int goalVal)
{

	__threadfence ();

	// only thread 0 is used for synchronization
	if (threadIdx.x == 0) 
		atomicAdd(&g_mutex, 1);
	
	// only when all blocks add 1 to g_mutex will
	// g_mutex equal to goalVal
	while(g_mutex < goalVal) {			// infinite loop until g_mutx = goalVal
	}

	if ( blockIdx.x == 0 && threadIdx.x == 0 ) g_mutex = 0;
	
	__syncthreads();
}

// countingSortInPlace -- GPU_SYNC DOES NOT WORK
/*uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;		// particle index				
	if ( i >= pnum ) { __gpu_sync ( 2 ); return; }

	register float3	ipos, ivel, iveleval, iforce;
	register float	ipress, idens;
	register int	icell, indx, iclr;

	icell = buf.mgcell [ i ];
	indx = buf.mgndx [ i ];
	int sort_ndx = buf.mgridoff[ icell ] + indx;				// global_ndx = grid_cell_offet + particle_offset
	if ( icell == GRID_UNDEF ) { __gpu_sync ( 2 ); return; }

	ipos = buf.mpos [ i ];
	ivel = buf.mvel [ i ];
	iveleval = buf.mveleval [ i ];
	iforce = buf.mforce [ i ];
	ipress = buf.mpress [ i ];
	idens = buf.mdensity [ i ];
	iclr = buf.mclr [ i ];

	__gpu_sync ( 2 ) ; //threadfence();			// make sure every thread in all blocks has their data

	
	buf.mpos [ sort_ndx ] = ipos;
	buf.mvel [ sort_ndx ] = ivel;
	buf.mveleval [ sort_ndx ] = iveleval;
	buf.mforce [ sort_ndx ] = iforce;
	buf.mpress [ sort_ndx ] = ipress;
	buf.mdensity [ sort_ndx ] = idens;
	buf.mclr [ sort_ndx ] = iclr;*/



// Counting Sort - Index
__global__ void countingSortIndex ( bufList buf, int pnum )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;		// particle index				
	if ( i >= pnum ) return;

	uint icell = buf.mgcell[i];
	uint indx =  buf.mgndx[i];
	int sort_ndx = buf.mgridoff[ icell ] + indx;				// global_ndx = grid_cell_offet + particle_offset
	if ( icell != GRID_UNDEF ) {
		buf.mgrid[ sort_ndx ] = i;					// index sort, grid refers to original particle order
	}
}

// Counting Sort - Full (deep copy)
__global__ void countingSortFull ( bufList buf, int pnum )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;		// particle index				
	if ( i >= pnum ) return;

	// Copy particle from original, unsorted buffer (msortbuf),
	// into sorted memory location on device (mpos/mvel)
	uint icell = *(uint*) (buf.msortbuf + pnum*BUF_GCELL + i*sizeof(uint) );
	uint indx =  *(uint*) (buf.msortbuf + pnum*BUF_GNDX + i*sizeof(uint) );		

	if ( icell != GRID_UNDEF ) {	  
		// Determine the sort_ndx, location of the particle after sort
	    int sort_ndx = buf.mgridoff[ icell ] + indx;				// global_ndx = grid_cell_offet + particle_offset	
		
		// Find the original particle data, offset into unsorted buffer (msortbuf)
		char* bpos = buf.msortbuf + i*sizeof(float3);

		// Transfer data to sort location
		buf.mgrid[ sort_ndx ] = sort_ndx;			// full sort, grid indexing becomes identity		
		buf.mpos[ sort_ndx ] =		*(float3*) (bpos);
		buf.mvel[ sort_ndx ] =		*(float3*) (bpos + pnum*BUF_VEL );
		buf.mveleval[ sort_ndx ] =	*(float3*) (bpos + pnum*BUF_VELEVAL );
		buf.mforce[ sort_ndx ] =	*(float3*) (bpos + pnum*BUF_FORCE );
		buf.mpress[ sort_ndx ] =	*(float*) (buf.msortbuf + pnum*BUF_PRESS + i*sizeof(float) );
		buf.mdensity[ sort_ndx ] =	*(float*) (buf.msortbuf + pnum*BUF_DENS + i*sizeof(float) );
		buf.mclr[ sort_ndx ] =		*(uint*) (buf.msortbuf + pnum*BUF_CLR+ i*sizeof(uint) );		// ((uint) 255)<<24; -- dark matter
		buf.mgcell[ sort_ndx ] =	icell;
		buf.mgndx[ sort_ndx ] =		indx;		
	}
}

// ***** UNUSED CODE (not working) ******
__global__ void countActiveCells ( bufList buf, int pnum )
{	
	if ( threadIdx.x == 0 ) {		
		// use only one processor
		
		//gridActive = -1;

		int last_ndx = buf.mgridoff [ simData.gridTotal-1 ] + buf.mgridcnt[ simData.gridTotal-1 ] - 1;
		int last_p = buf.mgrid[ last_ndx ];
		int last_cell = buf.mgcell[ last_p ];
		int first_p = buf.mgrid[ 0 ];
		int first_cell = buf.mgcell[ first_p ] ;

		int cell, cnt = 0, curr = 0;
		cell = first_cell;
		while ( cell < last_cell ) {			
			buf.mgridactive[ cnt ] = cell;			// add cell to active list
			cnt++;
			curr += buf.mgridcnt[cell];				// advance to next active cell
			// id = buf.mgrid[curr];				// get particle id -- when unsorted only
			cell = buf.mgcell [ curr ];				// get cell we are in -- use id when unsorted
		}
		// gridActive = cnt;
	}
	__syncthreads();
}


__device__ float contributePressure ( int i, float3 p, int cell, bufList buf )
{			
	float3 dist;
	float dsq, c, sum;
	register float d2 = simData.psimscale * simData.psimscale;
	register float r2 = simData.r2 / d2;
	
	sum = 0.0;

	if ( buf.mgridcnt[cell] == 0 ) return 0.0;
	
	int cfirst = buf.mgridoff[ cell ];
	int clast = cfirst + buf.mgridcnt[ cell ];
	
	for ( int cndx = cfirst; cndx < clast; cndx++ ) {
		dist = p - buf.mpos[ buf.mgrid[cndx] ];
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
		if ( dsq < r2 && dsq > 0.0) {
			c = (r2 - dsq)*d2;
			sum += c * c * c;				
		} 
	}
	
	return sum;
}
			
__global__ void computePressure ( bufList buf, int pnum )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= pnum ) return;

	// Get search cell
	int nadj = (1*simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[ i ];
	if ( gc == GRID_UNDEF ) return;						// particle out-of-range
	gc -= nadj;

	// Sum Pressures
	float3 pos = buf.mpos[ i ];
	float sum = 0.0;
	for (int c=0; c < simData.gridAdjCnt; c++) {
		sum += contributePressure ( i, pos, gc + simData.gridAdj[c], buf );
	}
	__syncthreads();
		
	// Compute Density & Pressure
	sum = sum * simData.pmass * simData.poly6kern;
	if ( sum == 0.0 ) sum = 1.0;
	buf.mpress[ i ] = ( sum - simData.prest_dens ) * simData.pintstiff;
	buf.mdensity[ i ] = 1.0f / sum;
}

		
__global__ void computeQuery ( bufList buf, int pnum )
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= pnum ) return;

	// Get search cell
	int nadj = (1*simData.gridRes.z + 1)*simData.gridRes.x + 1;
	uint gc = buf.mgcell[ i ];
	if ( gc == GRID_UNDEF ) return;						// particle out-of-range
	gc -= nadj;

	// Sum Pressures
	float3 pos = buf.mpos[ i ];
	float sum = 0.0;
	for (int c=0; c < simData.gridAdjCnt; c++) {
		sum += 1.0;
	}
	__syncthreads();
	
}

/*FindNeighbors
int cid = blockIdx.x * blockSize.x + blockIdx.y;   // cluster id	
int pid = threadIdx.x;		           // 0 to 85 (max particles per cell)	
__shared__ Particle  clist[ 85 ];	
__shared__ Particle  plist[ 85*8 ];
if ( pid < clusterCnt[cid] )  
	clist [ pid ] = particles [ clusterNdx[cid] + pid ];

for ( gid = 0;  gid < 8;  gid++ ) {
	if ( pid < gridCnt[  cid + group[gid] ] )  
		plist [ cid*CELL_CNT + pid ] = particles [ sortNdx[ cid + group[gid] ]  + pid ]; 	}

__syncthreads();	
	
for ( int j = 0; j < cellcnt;  j++ ) {
	dst = plist[ pid ] - plist[ j ];
	if ( dst < R2 ) {
     		  ...
	}
}*/

/*grid		    block
<gx, gy, gz>    <1, 32, 64>
256, 256, 256  
total:  */


#define LOCAL_PMAX		896
#define NUM_CELL		27
#define LAST_CELL		26
#define CENTER_CELL		13

__global__ void computePressureGroup ( bufList buf, int pnum )
{
	__shared__ float3	cpos[ LOCAL_PMAX ];

	__shared__ int		ncnt[ NUM_CELL ];
	__shared__ int		ngridoff[ NUM_CELL ];
	__shared__ int		noff[ NUM_CELL ];
	
	int bid = __mul24( blockIdx.y, gridDim.x ) + blockIdx.x;
	if ( bid > gridActive ) return;				// block must be in a valid grid
	uint cell = buf.mgridactive [ bid ];		// get grid cell (from blockID 1:1)
	register int i = -1;
	register float3 ipos;

	uint ndx = threadIdx.x;							
	if ( ndx < buf.mgridcnt[cell] ) {
		i = buf.mgridoff[cell] + ndx;		// particle id to process
		ipos = buf.mpos[ i ];
	}
	int gid = threadIdx.x;

	register float d2 = simData.psimscale * simData.psimscale;
	register float r2 = simData.r2 / d2;
	register float3 dist;
	register float c, dsq, sum;
	int neighbor;

	// copy neighbor cell counts to shared mem
	if ( gid < NUM_CELL ) {
		int nadj = (1*simData.gridRes.z + 1)*simData.gridRes.x + 1;
		neighbor = cell - nadj + simData.gridAdj[gid];					// neighbor cell id
		ncnt[gid] = buf.mgridcnt [ neighbor ];	
		ngridoff[gid] = buf.mgridoff [ neighbor ];
	}
	__syncthreads ();

	if ( gid == 0 ) {									// compute neighbor local ndx (as prefix sum)
		int nsum = 0;
		for (int z=0; z < NUM_CELL; z++) {				// 27-step prefix sum
			noff[z] = nsum;
			nsum += ncnt[z];
		}
	}
	__syncthreads ();

	// copy particles into shared memory
	if ( gid < NUM_CELL ) {
		for (int j=0; j < ncnt[gid]; j++ ) {
			neighbor = buf.mgrid [ ngridoff[gid] + j ];		// neighbor particle id
			ndx = noff[ gid ] + j;
			cpos[ ndx ] = buf.mpos [ neighbor ];
		}
	}
	__syncthreads ();

	
	// compute pressure for current particle
	if ( i == -1 ) return;
	
	int jnum = noff[LAST_CELL] + ncnt[LAST_CELL];
	sum = 0.0;
	for (int j = 0; j < jnum; j++) {
		dist = ipos - cpos[ j ];
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);			
		if ( dsq > 0.0 && dsq < r2 ) {
			c = (r2 - dsq)*d2;
			sum += c * c * c;
		}
	}	
	__syncthreads ();

	// put result into global mem
	sum = sum * simData.pmass * simData.poly6kern;
	if ( sum == 0.0 ) sum = 1.0;
	buf.mpress[ i ] = ( sum - simData.prest_dens ) * simData.pintstiff;
	buf.mdensity[ i ] = 1.0f / sum; 	
}


__device__ float3 contributeForce ( int i, float3 ipos, float3 iveleval, float ipress, float idens, int cell, bufList buf )
{			
	float dsq, c;	
	float pterm;
	float3 dist, force;	
	int j;					

	if ( buf.mgridcnt[cell] == 0 ) return make_float3(0,0,0);	

	force = make_float3(0,0,0);

	for ( int cndx = buf.mgridoff[ cell ]; cndx < buf.mgridoff[ cell ] + buf.mgridcnt[ cell ]; cndx++ ) {										
		j = buf.mgrid[ cndx ];				
		dist = ( ipos - buf.mpos[ j ] );		// dist in cm
		dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
		if ( dsq < simData.rd2 && dsq > 0) {			
			dsq = sqrt(dsq * simData.d2);
			c = ( simData.psmoothradius - dsq ); 
			pterm = simData.psimscale * -0.5f * c * simData.spikykern * ( ipress + buf.mpress[ j ] ) / dsq;			
			force += ( pterm * dist + simData.vterm * ( buf.mveleval[ j ] - iveleval )) * c * idens * (buf.mdensity[ j ] );
		}	
	}
	return force;
}


__global__ void computeForce ( bufList buf, int pnum)
{			
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= pnum ) return;

	// Get search cell	
	uint gc = buf.mgcell[ i ];
	if ( gc == GRID_UNDEF ) return;						// particle out-of-range
	gc -= (1*simData.gridRes.z + 1)*simData.gridRes.x + 1;

	// Sum Pressures	
	register float3 force;
	force = make_float3(0,0,0);		

	for (int c=0; c < simData.gridAdjCnt; c++) {
		force += contributeForce ( i, buf.mpos[ i ], buf.mveleval[ i ], buf.mpress[ i ], buf.mdensity[ i ], gc + simData.gridAdj[c], buf );
	}
	buf.mforce[ i ] = force;
}
	

/*__global__ void computeForceNbr ( char* bufPnts, int* bufGrid, int numPnt )
{		
	uint ndx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index		
	if ( ndx >= numPnt ) return;
				
	char* ioffs = bufPnts + __mul24(ndx, simData.stride );
	float3 ipos = *(float3*)	(ioffs + OFFSET_POS);
	float3 ivelval = *(float3*)	(ioffs + OFFSET_VELEVAL);
	float press = *(float*)		(ioffs + OFFSET_PRESS);
	float dens =  *(float*)		(ioffs + OFFSET_DENS);
	int icnt =  *(int*)			(ioffs + OFFSET_NBRCNT);

	char* joffs;
	float3 jpos, jveleval;

	float3 dist, force;		
	float c, ndistj, pterm, dterm, vterm;
		
	vterm = simData.lapkern * simData.visc;
		
	force = make_float3(0,0,0);
	for (int nbr=0; nbr < icnt; nbr++) {		// base 1, n[0] = count
		ndistj = bufNdist[ndx][nbr];
		joffs = bufPnts + __mul24(bufNeighbor[ndx][nbr], simData.stride);
		jpos = *(float3*)		(joffs + OFFSET_POS);
		jveleval = *(float3*)	(joffs + OFFSET_VELEVAL);
		c = ( simData.smooth_rad - ndistj ); 
		dist.x = ( ipos.x - jpos.x );		// dist in cm
		dist.y = ( ipos.y - jpos.y );
		dist.z = ( ipos.z - jpos.z );			
		pterm = simData.sim_scale * -0.5f * c * simData.spikykern * ( press + *(float*)(joffs+OFFSET_PRESS) ) / ndistj;
		dterm = c * dens * *(float*)(joffs+OFFSET_DENS);	
		force.x += ( pterm * dist.x + vterm * ( jveleval.x - ivelval.x )) * dterm;
		force.y += ( pterm * dist.y + vterm * ( jveleval.y - ivelval.y )) * dterm;
		force.z += ( pterm * dist.z + vterm * ( jveleval.z - ivelval.z )) * dterm;			
	}
	*(float3*) ( ioffs + OFFSET_FORCE ) = force;		
}*/

		
__global__ void advanceParticles ( float time, float dt, float ss, bufList buf, int numPnts )
{		
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if ( i >= numPnts ) return;
	
	if ( buf.mgcell[i] == GRID_UNDEF ) {
		buf.mpos[i] = make_float3(-1000,-1000,-1000);
		buf.mvel[i] = make_float3(0,0,0);
		return;
	}
			
	// Get particle vars
	register float3 accel, norm;
	register float diff, adj, speed;
	register float3 pos = buf.mpos[i];
	register float3 veval = buf.mveleval[i];

	// Leapfrog integration						
	accel = buf.mforce[i];
	accel *= simData.pmass;
		
	// Boundaries
	// Y-axis
	
	diff = simData.pradius - (pos.y - (simData.pboundmin.y + (pos.x-simData.pboundmin.x)*simData.pground_slope )) * ss;
	if ( diff > EPSILON ) {
		norm = make_float3( -simData.pground_slope, 1.0 - simData.pground_slope, 0);
		adj = simData.pextstiff * diff - simData.pdamp * dot(norm, veval );
		norm *= adj; accel += norm;
	}

	diff = simData.pradius - ( simData.pboundmax.y - pos.y )*ss;
	if ( diff > EPSILON ) {
		norm = make_float3(0, -1, 0);
		adj = simData.pextstiff * diff - simData.pdamp * dot(norm, veval );
		norm *= adj; accel += norm;
	}

	// X-axis
	diff = simData.pradius - (pos.x - (simData.pboundmin.x + (sin(time*simData.pforce_freq)+1)*0.5 * simData.pforce_min))*ss;
	if ( diff > EPSILON ) {
		norm = make_float3( 1, 0, 0);
		adj = (simData.pforce_min+1) * simData.pextstiff * diff - simData.pdamp * dot(norm, veval );
		norm *= adj; accel += norm;
	}
	diff = simData.pradius - ( (simData.pboundmax.x - (sin(time*simData.pforce_freq)+1)*0.5*simData.pforce_max) - pos.x)*ss;
	if ( diff > EPSILON ) {
		norm = make_float3(-1, 0, 0);
		adj = (simData.pforce_max+1) * simData.pextstiff * diff - simData.pdamp * dot(norm, veval );
		norm *= adj; accel += norm;
	}

	// Z-axis
	diff = simData.pradius - (pos.z - simData.pboundmin.z ) * ss;
	if ( diff > EPSILON ) {
		norm = make_float3( 0, 0, 1 );
		adj = simData.pextstiff * diff - simData.pdamp * dot(norm, veval );
		norm *= adj; accel += norm;
	}
	diff = simData.pradius - ( simData.pboundmax.z - pos.z )*ss;
	if ( diff > EPSILON ) {
		norm = make_float3( 0, 0, -1 );
		adj = simData.pextstiff * diff - simData.pdamp * dot(norm, veval );
		norm *= adj; accel += norm;
	}
		
	// Gravity
	accel += simData.pgravity;

	// Accel Limit
	speed = accel.x*accel.x + accel.y*accel.y + accel.z*accel.z;
	if ( speed > simData.AL2 ) {
		accel *= simData.AL / sqrt(speed);
	}

	// Velocity Limit
	float3 vel = buf.mvel[i];
	speed = vel.x*vel.x + vel.y*vel.y + vel.z*vel.z;
	if ( speed > simData.VL2 ) {
		speed = simData.VL2;
		vel *= simData.VL / sqrt(speed);
	}

	// Ocean colors
	if ( speed > simData.VL2*0.2) {
		adj = simData.VL2*0.2;
		buf.mclr[i] += ((  buf.mclr[i] & 0xFF) < 0xFD ) ? +0x00000002 : 0;		// decrement R by one
		buf.mclr[i] += (( (buf.mclr[i]>>8) & 0xFF) < 0xFD ) ? +0x00000200 : 0;	// decrement G by one
		buf.mclr[i] += (( (buf.mclr[i]>>16) & 0xFF) < 0xFD ) ? +0x00020000 : 0;	// decrement G by one
	}
	if ( speed < 0.03 ) {		
		int v = int(speed/.01)+1;
		buf.mclr[i] += ((  buf.mclr[i] & 0xFF) > 0x80 ) ? -0x00000001 * v : 0;		// decrement R by one
		buf.mclr[i] += (( (buf.mclr[i]>>8) & 0xFF) > 0x80 ) ? -0x00000100 * v : 0;	// decrement G by one
	}
	
	//-- surface particle density 
	//buf.mclr[i] = buf.mclr[i] & 0x00FFFFFF;
	//if ( buf.mdensity[i] > 0.0014 ) buf.mclr[i] += 0xAA000000;

	// Leap-frog Integration
	float3 vnext = accel*dt + vel;				// v(t+1/2) = v(t-1/2) + a(t) dt		
	buf.mveleval[i] = (vel + vnext) * 0.5;		// v(t+1) = [v(t-1/2) + v(t+1/2)] * 0.5			
	buf.mvel[i] = vnext;
	buf.mpos[i] += vnext * (dt/ss);						// p(t+1) = p(t) + v(t+1/2) dt		
}


void updateSimParams ( FluidParams* cpufp )
{
	cudaError_t status;
	#ifdef CUDA_42
		// Only for CUDA 4.x or earlier. Depricated in CUDA 5.0+
		// Original worked even if symbol was declared __device__
		status = cudaMemcpyToSymbol ( "simData", cpufp, sizeof(FluidParams) );
	#else
		// CUDA 5.x+. Only works if symbol is declared __constant__
		status = cudaMemcpyToSymbol ( simData, cpufp, sizeof(FluidParams) );
	#endif

	/*app_printf ( "SIM PARAMETERS:\n" );
	app_printf ( "  CPU: %p\n", cpufp );	
	app_printf ( "  GPU: %p\n", &simData );	 */
}

