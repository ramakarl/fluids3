//-----------------------------------------------------------------------------
// FLUIDS v.3.1 - SPH Fluid Simulator for CPU and GPU
// Copyright (C) 2012-2013, 2021. Rama Hoetzlein, http://fluids3.com
//-----------------------------------------------------------------------------

#ifndef DEF_FLUID_SYS
	#define DEF_FLUID_SYS

	#include <iostream>
	#include <vector>
	#include <stdio.h>
	#include <stdlib.h>
	#include <math.h>	

	#include "fluid.h"
	#include "vec.h"
	#include "camera.h"

	#define MAX_PARAM			50
	#define GRID_UCHAR			0xFF
	#define GRID_UNDEF			4294967295	

	#define FUNC_INSERT			0
	#define	FUNC_COUNTING_SORT	1
	#define FUNC_QUERY			2
	#define FUNC_COMPUTE_PRESS	3
	#define FUNC_COMPUTE_FORCE	4
	#define FUNC_ADVANCE		5
	#define FUNC_EMIT			6
	#define FUNC_RANDOMIZE		7
	#define FUNC_SAMPLE			8
	#define FUNC_FPREFIXSUM		9
	#define FUNC_FPREFIXFIXUP	10
	#define FUNC_MAX			12

	#define COLORA(r,g,b,a)	( (uint((a)*255.0f)<<24) | (uint((b)*255.0f)<<16) | (uint((g)*255.0f)<<8) | uint((r)*255.0f) )

	#define GPU_OFF				0
	#define GPU_SINGLE			1
	#define GPU_TEMP			2
	#define GPU_DUAL			4
	#define	GPU_GL				8

	#define CPU_OFF				4
	#define CPU_YES				5

	class FluidSystem {
	public:
		FluidSystem ();
		~FluidSystem();
		
		// Setup
		void Initialize ( bool bGPU );
		void Start ( int num );
		void Run ();
		void SetupKernels ();
		void SetupDefaultParams ();
		void SetupExampleParams ();
		void SetupSpacing ();
		void SetupAddVolume ( Vector3DF min, Vector3DF max, float spacing, float offs, int total );
		void SetupGrid ();		
		void Reallocate ( int cnt, bool bCPU );				
		void IntegrityCheck();
		void LoadKernel ( int id, std::string kname );
	
		// Rendering
		void Draw ( int frame, Camera3D* cam, float rad );

		// CPU Simulation
		
		void InsertParticles ();
		void PrefixSumCells ();
		void CountingSortFull ();
		void ComputePressures ();
		void ComputeForces ();
		void AdvanceTime ();		
		void Advance ();
		void EmitParticles ();
		void Exit ();
			
		// GPU Simulation		
		void FluidSetupCUDA (  int num, int gsrch, Vector3DI res, Vector3DF size, Vector3DF delta, Vector3DF gmin, Vector3DF gmax, int total, int chk );
		void UpdateParamsCUDA ();
		void TransferToCUDA (int bufid);
		void TransferFromCUDA (int bufid);
		void TransferCopyCUDA (int src, int dest);
		void InsertParticlesCUDA ( uint* gcell, uint* ccell, uint* gcnt );	
		void PrefixSumCellsCUDA ( uint* goff, int zero_offsets );		
		void CountingSortFullCUDA ( Vector3DF* gpos );
		void ComputePressureCUDA ();
		void ComputeQueryCUDA ();
		void ComputeForceCUDA ();	
		void SampleParticlesCUDA ( float* outbuf, Vector3DI res, Vector3DF bmin, Vector3DF bmax, float scalar );		
		void AdvanceCUDA ( float time, float dt, float ss );
		void EmitParticlesCUDA ( float time, int cnt );

		// Buffers
		void AllocateBuffer ( std::string name, int buf_id, int stride, int cnt, int gpumode, int cpumode);	
		void MapBuffer ( int bufid, bool map ) ;
		Fluid* AddParticle ();
		void AddEmit ( float spacing );
		int NumPoints ()					{ return mNumPoints; }		
		CUdeviceptr getBufferGPU ( int id )	{ return m_Fluid.gpu(id); }

		// Acceleration Grid
		int getGridCell ( int p, Vector3DI& gc );
		int getGridCell ( Vector3DF& p, Vector3DI& gc );
		int getGridTotal ()		{ return m_GridTotal; }
		int getSearchCnt ()		{ return m_GridAdjCnt; }
		Vector3DI getCell ( int gc );
		Vector3DF GetGridRes ()		{ return m_GridRes; }
		Vector3DF GetGridMin ()		{ return m_GridMin; }
		Vector3DF GetGridMax ()		{ return m_GridMax; }
		Vector3DF GetGridDelta ()	{ return m_GridDelta; }

		double GetDT()		{ return m_DT; }
		void SetDebug(bool b) { mbDebug = b; }
		void DebugPrintMemory ();
	
	private:
		bool						m_bCPU;
		bool						m_Cmds[10];
		Vector3DI					m_FrameRange;
		Vector3DI					m_VolRes;
		int							m_BrkRes;
		std::string					m_InFile;
		std::string					m_OutFile;
		std::string					m_WorkPath;
		float						m_Thresh;
		bool						mbDebug;

		std::string					mSceneName;

		// Time
		int							m_Frame;		
		float						m_DT;
		float						m_Time;	

		// CUDA Kernels
		CUmodule					m_Module;
		CUfunction					m_Func[ FUNC_MAX ];

		// SPH Kernel functions
		float						m_R2, m_Poly6Kern, m_LapKern, m_SpikyKern;		

		// Particle Buffers
		int						mNumPoints;
		int						mMaxPoints;
		int						mGoodPoints;
		FBufs					m_Fluid;				// Fluid buffers		
		FParams					m_Params;				// Fluid parameters

		CUdeviceptr				cuFBuf;					// GPU pointer containers		
		CUdeviceptr				cuFParams;

		// Acceleration Grid		
		int						m_GridTotal;			// total # cells
		Vector3DI				m_GridRes;				// resolution in each axis
		Vector3DF				m_GridMin;				// volume of grid (may not match domain volume exactly)
		Vector3DF				m_GridMax;		
		Vector3DF				m_GridSize;				// physical size in each axis
		Vector3DF				m_GridDelta;
		int						m_GridSrch;
		int						m_GridAdjCnt;
		int						m_GridAdj[216];

		// Acceleration Neighbor Table
		int						m_NeighborNum;
		int						m_NeighborMax;
		int*					m_NeighborTable;
		float*					m_NeighborDist;

		int						mVBO[3];

		// Record/Playback
		bool					mbRecord;		
		bool					mbRecordBricks;
		int						mSpherePnts;
		int						mTex[1];		

		// Selected particle
		int						mSelected;


		// Saved results (for algorithm validation)
		uint*					mSaveNdx;
		uint*					mSaveCnt;
		uint*					mSaveNeighbors;		
	};	

	

#endif
