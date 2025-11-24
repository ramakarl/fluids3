/*
  FLUIDS v.1 - SPH Fluid Simulator for CPU and GPU
  Copyright (C) 2008. Rama Hoetzlein, http://www.rchoetzlein.com

  ZLib license
  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

#ifndef DEF_FLUID
	#define DEF_FLUID
	
	#include <cuda.h>
	#include <curand.h>

	typedef	unsigned int		uint;	
	typedef	unsigned short int	ushort;	

	struct NList {
		int num;
		int first;
	};
	#ifdef CUDA_KERNEL
		typedef float3		f3;
		typedef int3		i3;
	#else
		#include "vec.h"
		typedef Vector3DF	f3;		
		typedef Vector3DI	i3;
	#endif

	#define FPOS		0		// particle buffers
	#define FVEL		1
	#define FVEVAL		2
	#define FFORCE		3
	#define FPRESS		4		
	#define FCLR		5
	#define FGCELL		6
	#define FGNDX		7
	#define FGNEXT		8

	#define FNBRNDX		9		// particle neighbors (optional)
	#define FNBRCNT		10
	#define FCLUSTER	11	

	#define FGRID		12		// uniform acceleration grid
	#define FGRIDCNT	13
	#define	FGRIDOFF	14
	#define FGRIDACT	15
	#define FSTATE		16
	#define FBRICK		17
	#define FPARAMS		18		// fluid parameters
	#define FAUXARRAY1	19		// auxiliary arrays (prefix sums)
	#define FAUXSCAN1   20
	#define FAUXARRAY2	21
	#define FAUXSCAN2	22
	#define MAX_BUF		23

	#ifdef CUDA_KERNEL
		#define	CALLFUNC	__device__
	#else
		#define CALLFUNC
	#endif		

	// Particle & Grid Buffers
	struct FBufs {
		
		char* mcpu[MAX_BUF] = { nullptr };

		inline CALLFUNC void setBuf(int n, char* buf) {
			if (mcpu[n] != nullptr) {
				free(mcpu[n]);
			}
			mcpu[n] = buf;
		}		

		#ifdef CUDA_KERNEL
			// Kernel code
			char* mgpu[MAX_BUF] = { nullptr };			// on device, pointer is local 	
			char* mgrsc[MAX_BUF] = { nullptr };
			int	  mglid[MAX_BUF] = { 0 };				

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
			inline CALLFUNC f3*		bufF3(int n)		{ return (f3*)	   mcpu[n]; }		
			inline CALLFUNC float*  bufF (int n)		{ return (float*)  mcpu[n]; }
			inline CALLFUNC uint*   bufI (int n)		{ return (uint*)   mcpu[n]; }
			inline CALLFUNC char*   bufC (int n)		{ return (char*)   mcpu[n]; }		

		#endif	

		
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

	#define OFFSET_POS		0
	#define OFFSET_VEL		12
	#define OFFSET_VELEVAL	24
	#define OFFSET_FORCE	36
	#define OFFSET_PRESS	48
	#define OFFSET_DENS		52
	#define OFFSET_CELL		56
	#define OFFSET_GCONT	60
	#define OFFSET_CLR		64	

	// Fluid Parameters (stored on both host and device)
	struct FParams {
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

#endif /*PARTICLE_H_*/
