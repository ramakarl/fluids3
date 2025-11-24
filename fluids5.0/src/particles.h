//-----------------------------------------------------------------------------
// FLUIDS v5.0 - SPH Fluid Simulator for CPU and GPU
// Copyright (C) 2021. Rama Hoetzlein, http://fluids3.com
//-----------------------------------------------------------------------------

#ifndef DEF_PARTICLES
	#define DEF_PARTICLES

	#include <iostream>
	#include <vector>
	#include <stdio.h>
	#include <stdlib.h>
	#include <math.h>	
	#include "vec.h"
	#include "camera.h"

	#include "datax.h"
	#include "fluid.h"			// SPH fluid particles	

	#define FUNC_INSERT			0
	#define	FUNC_COUNTING_SORT	1
	#define FUNC_COMPUTE_PRESS	2
	#define FUNC_COMPUTE_FORCE	3
	#define FUNC_ADVANCE		4
	#define FUNC_FPREFIXSUM		5
	#define FUNC_FPREFIXFIXUP	6
	#define FUNC_MAX			8

	class Particles {
	public:
		Particles ();
		~Particles ();
		
		// Setup
		void Initialize ();
		void LoadKernel ( int id, std::string kname );
		void Restart ( int num );		
		void SetupDefaultParams ();
		void SetupExampleParams ();				
		int  AddParticle ();		
		void AddPointsInVolume ( Vector3DF min, Vector3DF max );		
		void ReallocateParticles ( int cnt );
		void RebuildAccelGrid ();
		void UpdateParams ();
		void UpdateGPUAccess();
		void Clear ();
		void DebugPrintMemory ();			
		void Draw ( int frame, Camera3D* cam, float rad );		// OpenGL points

		// SPH Simulation (CPU & GPU)
		void Run ();		
		void InsertParticles ();		
		void PrefixScanParticles ();
		void CountingSort ();	
		void ComputePressure ();
		void ComputeForce ();	
		void Advance ();
		void AdvanceTime ();		
		
		// Data transfers
		void CopyAllToTemp ();
		void CommitAll ();
		void Retrieve ( int buf );
		void DebugPrint ( DataX& dat, int buf, int start=0, int disp=20 );

		// Acceleration Queries
		int getGridCell ( int p, Vector3DI& gc );
		int getGridCell ( Vector3DF& p, Vector3DI& gc );		
		Vector3DI getCell ( int gc );		
	
		// Query functions
		int NumPoints ()						{ return mNumPoints; }
		Vector3DF* getPos ( int n )				{ return m_Points.bufF3(FPOS,n); }
		Vector3DF* getVel ( int n )				{ return m_Points.bufF3(FVEL,n); }
		uint* getClr ( int n )					{ return m_Points.bufUI(FCLR,n); }				
		void SetDebug(bool b)	{ m_bDebug = b; }
	
	private:
		bool					m_bGPU;					// CPU or GPU execution
		bool					m_bDebug;			
		int						m_Frame;		

		// CUDA Kernels
		CUmodule				m_Module;
		CUfunction				m_Func[ FUNC_MAX ];

		// Particle Buffers
		int						mNumPoints;
		int						mMaxPoints;
		DataX					m_Points;				// Particle buffers
		DataX					m_PointsTemp;
		DataX					m_Accel;				// Acceleration buffers
		FParams_t				m_Params;				// Fluid parameters (should be made DataX in future)
		CUdeviceptr				m_cuParams;
	};	

	

#endif
