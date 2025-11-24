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

#ifndef DEF_KERN_CUDA
	#define DEF_KERN_CUDA

	#include <stdio.h>
	#include <math.h>

	typedef unsigned int		uint;
	typedef unsigned short int	ushort;

	// Particle & Grid Buffers
	struct bufList {
		float3*			mpos;
		float3*			mvel;
		float3*			mveleval;
		float3*			mforce;
		float*			mpress;
		float*			mdensity;		
		uint*			mgcell;
		uint*			mgndx;
		uint*			mclr;			// 4 byte color

		uint*			mcluster;

		char*			msortbuf;

		uint*			mgrid;	
		int*			mgridcnt;
		int*			mgridoff;
		int*			mgridactive;
	};

	// Temporary sort buffer offsets
	#define BUF_POS			0
	#define BUF_VEL			(sizeof(float3))
	#define BUF_VELEVAL		(BUF_VEL + sizeof(float3))
	#define BUF_FORCE		(BUF_VELEVAL + sizeof(float3))
	#define BUF_PRESS		(BUF_FORCE + sizeof(float3))
	#define BUF_DENS		(BUF_PRESS + sizeof(float))
	#define BUF_GCELL		(BUF_DENS + sizeof(float))
	#define BUF_GNDX		(BUF_GCELL + sizeof(uint))
	#define BUF_CLR			(BUF_GNDX + sizeof(uint))

	// Fluid Parameters (stored on both host and device)
	struct FluidParams {
		int				numThreads, numBlocks;
		int				gridThreads, gridBlocks;	

		int				szPnts, szHash, szGrid;
		int				stride, pnum;
		int				chk;
		float			pdist, pmass, prest_dens;
		float			pextstiff, pintstiff;
		float			pradius, psmoothradius, r2, psimscale, pvisc;
		float			pforce_min, pforce_max, pforce_freq, pground_slope;
		float			pvel_limit, paccel_limit, pdamp;
		float3			pboundmin, pboundmax, pgravity;
		float			AL, AL2, VL, VL2;

		float			d2, rd2, vterm;		// used in force calculation		 
		
		float			poly6kern, spikykern, lapkern;

		float3			gridSize, gridDelta, gridMin, gridMax;
		int3			gridRes, gridScanMax;
		int				gridSrch, gridTotal, gridAdjCnt, gridActive;

		int				gridAdj[64];
	};

	// Prefix Sum defines - 16 banks on G80
	#define NUM_BANKS		16
	#define LOG_NUM_BANKS	 4


	#ifndef CUDA_KERNEL		
		// Host functions
		// Declare kernel functions that are available to the host.
		// These are defined in kern.cu, but declared here so host.cu can call them.

		
		__global__ void insertParticles ( bufList buf, int pnum );
		__global__ void countingSortIndex ( bufList buf, int pnum );		
		__global__ void countingSortFull ( bufList buf, int pnum );		
		__global__ void computeQuery ( bufList buf, int pnum );	
		__global__ void computePressure ( bufList buf, int pnum );		
		__global__ void computeForce ( bufList buf, int pnum );
		__global__ void computePressureGroup ( bufList buf, int pnum );
		__global__ void advanceParticles ( float time, float dt, float ss, bufList buf, int numPnts );
		__global__ void countActiveCells ( bufList buf, int pnum );		
		

		void updateSimParams ( FluidParams* cpufp );
	#endif

	// Prefix scan
	#define SCAN_BLOCKSIZE		512				// must match value in fluid_system_cuda.cu

	#ifdef CUDA_KERNEL
		// Kernel functions
		
	
	#else
		// Host headers		
		// NOTE: Template functions must be defined in the header
		__global__ template<bool,bool> void prescan (float*, const float*, float*, int, int, int );		
		__global__ template<bool,bool> void prescanInt (int*, const int*, int*, int, int, int );	
		
		__global__ void uniformAddInt (int*  g_data, int *uniforms, int n, int blockOffset, int baseIndex);	
		__global__ void uniformAdd    (float*g_data, float *uniforms, int n, int blockOffset, int baseIndex);	

		__global__ void prefixFixup ( unsigned int *input, unsigned int *aux, int len);
		__global__ void prefixSum ( unsigned int* input, unsigned int* output, unsigned int* aux, int len, int zeroff );
	#endif	

	#define EPSILON				0.00001f
	#define GRID_UCHAR			0xFF
	#define GRID_UNDEF			4294967295

	
#endif
