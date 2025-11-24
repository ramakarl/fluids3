/*
  FLUIDS v.3 - SPH Fluid Simulator for CPU and GPU
  Copyright (C) 2008, 2021. Rama Hoetzlein, http://ramakarl.com
*/

#ifndef DEF_FLUID_GPU
	#define DEF_FLUID_GPU
	
	#include <cuda.h>
	#include <curand.h>

	#define GRID_UNDEF		2147483647			// max int

	// Particle data
	#define FPOS		0		
	#define FVEL		1
	#define FCLR		2
	#define FVEVAL		3
	#define FFORCE		4
	#define FPRESS		5	
	#define FGCELL		6
	#define FGNDX		7	

	// Acceleration grid data
	#define AGRID		0	
	#define AGRIDCNT	1
	#define	AGRIDOFF	2
	#define AAUXARRAY1	3
	#define AAUXSCAN1	4
	#define AAUXARRAY2	5
	#define AAUXSCAN2	6

	// Fluid Parameters (stored on both host and device)
	// **NOTE** In future this should also be stored in DataX buffers

	#ifdef CUDA_KERNEL
		typedef float3		f3;
		typedef int3		i3;
	#else
		#include "vec.h"
		typedef Vector3DF	f3;		
		typedef Vector3DI	i3;
	#endif
	
	struct FParams_t {		
		int				mode, example;
		float			time, dt, sim_scale;
		bool			capture;

		int				numThreads, numBlocks;
		int				gridThreads, gridBlocks;	
		int				szPnts, szHash, szGrid;
		int				stride, pnum, chk;

		float			pdist, pmass, prest_dens, pintstiff;
		float			pradius, psmoothradius, pspacing, r2, pvisc;
		
		float			AL, AL2, VL, VL2;
		float			d2, rd2, vterm;		// used in force calculation		 
		float			poly6kern, spikykern, lapkern, gausskern;

		float			grid_size, grid_density;
		f3				gridSize, gridDelta, gridMin, gridMax;
		i3				gridRes, gridScanMax;
		int				gridSrch, gridTotal, gridAdjCnt, gridActive;
		int				gridAdj[64];	
		
		float			bound_slope, bound_stiff, bound_friction, bound_damp;
		float			bound_wall_force, bound_wall_freq;
		f3				bound_min, bound_max;
		f3				init_min, init_max;

		f3				grav_dir, grav_pos, gravity;
		float			grav_amt;

		f3				emit_rate, emit_pos, emit_ang, emit_dang, emit_spread;

		i3				brickRes;

		int				pemit;
		int				pmem;
	};

#endif	
