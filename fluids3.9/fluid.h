//-----------------------------------------------------------------------------
// FLUIDS v.4.0 - SPH Fluid Simulator for CPU and GPU
// Copyright (C) 2012-2013, 2021. Rama Hoetzlein, http://fluids3.com
//-----------------------------------------------------------------------------


#ifndef DEF_FLUID
	#define DEF_FLUID
	
	#include <cuda.h>
	#include <curand.h>

	#define ALIGN(x)	__align__(x)

	typedef	unsigned int		uint;	
	typedef	unsigned short int	ushort;	

	#ifdef CUDA_KERNEL
		typedef float3		f3;
		typedef int3		i3;
	#else
		#include "vec.h"
		typedef Vector3DF	f3;		
		typedef Vector3DI	i3;
	#endif

	struct ALIGN(16) Fluid {
		f3		pos;
		f3		vel;
		uint	clr;
		f3		veval;
		f3		force;
		float	press;		

		int		gcell;
		int		gndx;
	};

	#define FFLUID		0		// fluid buffer
	#define FFLUIDTEMP	1
	#define FGRID		2		// uniform acceleration grid
	#define FGRIDCNT	3
	#define	FGRIDOFF	4
	#define FGRIDACT	5
	#define FPARAMS		6		// fluid parameters
	#define FAUXARRAY1	7		// auxiliary arrays (prefix sums)
	#define FAUXSCAN1   8
	#define FAUXARRAY2	9
	#define FAUXSCAN2	10
	#define MAX_BUF		12

	#ifdef CUDA_KERNEL
		#define	CALLFUNC	__device__
	#else
		#define CALLFUNC
	#endif		

	// Grid Buffers
	struct ALIGN(16) FBufs {
		
		size_t sz[MAX_BUF] = { 0 };
		char* mcpu[MAX_BUF] = { nullptr };

		inline CALLFUNC void setBuf(int n, char* buf) {
			if (mcpu[n] != nullptr) {
				free(mcpu[n]);
			}
			mcpu[n] = buf;
		}		
		inline CALLFUNC int size(int n) { return sz[n]; }

		#ifdef CUDA_KERNEL
			// Kernel code
			char* mgpu[MAX_BUF] = { nullptr };			// on device, pointer is local 	
			char* mgrsc[MAX_BUF] = { nullptr };
			int	  mglid[MAX_BUF] = { 0 };				

			inline CALLFUNC Fluid*  pnt(int n)			{ return ((Fluid*) mgpu[FFLUID]) + n; }
			inline CALLFUNC Fluid*  ptemp(int n)		{ return ((Fluid*) mgpu[FFLUIDTEMP]) + n; }
			inline CALLFUNC f3*		bufF3(int n)		{ return (f3*)	   mgpu[n]; }		
			inline CALLFUNC float*  bufF (int n)		{ return (float*)  mgpu[n]; }
			inline CALLFUNC uint*   bufI (int n)		{ return (uint*)   mgpu[n]; }
			inline CALLFUNC char*   bufC (int n)		{ return (char*)   mgpu[n]; }			

		#else
			// Host code
			CUdeviceptr		mgpu[MAX_BUF] = { 0 };			// on host, gpu is a device pointer			
			CUgraphicsResource  mgrsc[MAX_BUF] = { 0 };		// for CUDA-GL interop (only some buffers)
			int				mglid[MAX_BUF] = { 0 };			// for GL buffers (only some buffers)

			CUdeviceptr		gpu (int n )	{ return mgpu[n]; }
			CUdeviceptr*	gpuptr (int n )	{ return &mgpu[n]; }
			int*			glid_ptr ( int n )  { return &mglid[n]; }
			int				glid ( int n )		{ return mglid[n]; }
			CUgraphicsResource* grsc(int n) { return &mgrsc[n]; }

			inline ~FBufs() {
				for (int i = 0; i < MAX_BUF; i++) {
					if (mcpu[i] != nullptr) {
						free(mcpu[i]);
					}
					if (mgpu[i] != 0) {
						cuMemFree(mgpu[i]);
					}
				}
			}
			inline CALLFUNC Fluid*  pnt(int n)			{ return ((Fluid*) mcpu[FFLUID]) + n; }
			inline CALLFUNC Fluid*  ptemp(int n)		{ return ((Fluid*) mcpu[FFLUIDTEMP]) + n; }			
			inline CALLFUNC f3*		bufF3(int n)		{ return (f3*)	   mcpu[n]; }		
			inline CALLFUNC float*  bufF (int n)		{ return (float*)  mcpu[n]; }
			inline CALLFUNC uint*   bufI (int n)		{ return (uint*)   mcpu[n]; }
			inline CALLFUNC char*   bufC (int n)		{ return (char*)   mcpu[n]; }		

		#endif	

	};

	// Fluid Parameters (stored on both host and device)
	struct ALIGN(16) FParams {
		int				mode, example;
		float			sim_scale;
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