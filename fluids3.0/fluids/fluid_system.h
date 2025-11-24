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


#ifndef DEF_FLUID_SYS
	#define DEF_FLUID_SYS

	#include <iostream>
	#include <vector>
	#include <stdio.h>
	#include <stdlib.h>
	#include <math.h>

	#include "xml_settings.h"

	#define MAX_PARAM			50
	#define GRID_UCHAR			0xFF
	#define GRID_UNDEF			4294967295

	#define RUN_PAUSE			0
	#define RUN_SEARCH			1
	#define RUN_VALIDATE		2
	#define RUN_CPU_SLOW		3
	#define RUN_CPU_GRID		4
	#define RUN_CUDA_RADIX		5
	#define RUN_CUDA_INDEX		6
	#define RUN_CUDA_FULL		7
	#define RUN_CUDA_CLUSTER	8
	#define RUN_PLAYBACK		9

	// Scalar params
	#define PMODE				0
	#define PNUM				1
	#define PEXAMPLE			2
	#define PSIMSIZE			3
	#define PSIMSCALE			4
	#define PGRID_DENSITY		5
	#define PGRIDSIZE			6
	#define PVISC				7
	#define PRESTDENSITY		8
	#define PMASS				9
	#define PRADIUS				10
	#define PDIST				11
	#define PSMOOTHRADIUS		12
	#define PINTSTIFF			13
	#define PEXTSTIFF			14
	#define PEXTDAMP			15
	#define PACCEL_LIMIT		16
	#define PVEL_LIMIT			17
	#define PSPACING			18
	#define PGROUND_SLOPE		19
	#define PFORCE_MIN			20
	#define PFORCE_MAX			21
	#define PMAX_FRAC			22
	#define PDRAWMODE			23
	#define PDRAWSIZE			24
	#define PDRAWGRID			25	
	#define PDRAWTEXT			26	
	#define PCLR_MODE			27
	#define PPOINT_GRAV_AMT		28
	#define PSTAT_OCCUPY		29
	#define PSTAT_GRIDCNT		30
	#define PSTAT_NBR			31
	#define PSTAT_NBRMAX		32
	#define PSTAT_SRCH			33
	#define PSTAT_SRCHMAX		34
	#define PSTAT_PMEM			35
	#define PSTAT_GMEM			36
	#define PTIME_INSERT		37
	#define PTIME_SORT			38
	#define PTIME_COUNT			39
	#define PTIME_PRESS			40
	#define PTIME_FORCE			41
	#define PTIME_ADVANCE		42
	#define PTIME_RECORD		43
	#define PTIME_RENDER		44
	#define PTIME_TOGPU			45
	#define PTIME_FROMGPU		46
	#define PFORCE_FREQ			47
	

	// Vector params
	#define PVOLMIN				0
	#define PVOLMAX				1
	#define PBOUNDMIN			2
	#define PBOUNDMAX			3
	#define PINITMIN			4
	#define PINITMAX			5
	#define PEMIT_POS			6
	#define PEMIT_ANG			7
	#define PEMIT_DANG			8
	#define PEMIT_SPREAD		9
	#define PEMIT_RATE			10
	#define PPOINT_GRAV_POS		11	
	#define PPLANE_GRAV_DIR		12	

	// Booleans
	#define PRUN				0
	#define PDEBUG				1	
	#define PUSE_CUDA			2	
	#define	PUSE_GRID			3
	#define PWRAP_X				4
	#define PWALL_BARRIER		5
	#define PLEVY_BARRIER		6
	#define PDRAIN_BARRIER		7		
	#define PPLANE_GRAV_ON		11	
	#define PPROFILE			12
	#define PCAPTURE			13

	#define BFLUID				2

	struct NList {
		int num;
		int first;
	};
	struct Fluid {						// offset - TOTAL: 72 (must be multiple of 12)
		Vector3DF		pos;			// 0
		Vector3DF		vel;			// 12
		Vector3DF		veleval;		// 24
		Vector3DF		force;			// 36
		float			pressure;		// 48
		float			density;		// 52
		int				grid_cell;		// 56
		int				grid_next;		// 60
		DWORD			clr;			// 64
		int				padding;		// 68
	};

	class FluidSystem {
	public:
		FluidSystem ();
		
		// Rendering
		void Draw ( Camera3D& cam, float rad );
		void DrawDomain ();
		void DrawGrid ();
		void DrawText ();
		void DrawCell ( int gx, int gy, int gz );
		void DrawParticle ( int p, int r1, int r2, Vector3DF clr2 );
		void DrawParticleInfo ()		{ DrawParticleInfo ( mSelected ); }
		void DrawParticleInfo ( int p );
		void DrawNeighbors ( int p );
		void DrawCircle ( Vector3DF pos, float r, Vector3DF clr, Camera3D& cam );

		// Particle Utilities
		void AllocateParticles ( int cnt );
		int AddParticle ();
		void AddEmit ( float spacing );
		int NumPoints ()		{ return mNumPoints; }
		
		// Setup
		void Setup ( bool bStart );
		void SetupRender ();
		void SetupKernels ();
		void SetupDefaultParams ();
		void SetupExampleParams ( bool bStart );
		void SetupSpacing ();
		void SetupAddVolume ( Vector3DF min, Vector3DF max, float spacing, float offs, int total );
		void SetupGridAllocate ( Vector3DF min, Vector3DF max, float sim_scale, float cell_size, float border );
		int ParseXML ( std::string name, int id, bool bStart );

		// Neighbor Search
		void Search ();
		void InsertParticles ();
		void BasicSortParticles ();
		void BinSortParticles ();
		void FindNbrsSlow ();
		void FindNbrsGrid ();

		// Simulation
		void Run (int w, int h);
		void RunSearchCPU ();
		void RunValidate ();
		void RunSimulateCPUSlow ();
		void RunSimulateCPUGrid ();
		void RunSimulateCUDARadix ();
		void RunSimulateCUDAIndex ();
		void RunSimulateCUDAFull ();
		void RunSimulateCUDACluster ();
		void RunPlayback ();
		
		void Advance ();
		void EmitParticles ();
		void Exit ();
		void TransferToCUDA ();
		void TransferFromCUDA ();
		void ValidateSortCUDA ();
		double GetDT()		{ return m_DT; }

		// Debugging
		void SaveResults ();
		void CaptureVideo (int width, int height);
		void ValidateResults ();
		void TestPrefixSum ( int num );
		void DebugPrintMemory ();
		void record ( int param, std::string, Time& start );
		int SelectParticle ( int x, int y, int wx, int wy, Camera3D& cam );
		int GetSelected ()		{ return mSelected; }

		
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

		// Acceleration Neighbor Tables
		void AllocateNeighborTable ();
		void ClearNeighborTable ();
		void ResetNeighbors ();
		int GetNeighborTableSize ()	{ return m_NeighborNum; }
		void ClearNeighbors ( int i );
		int AddNeighbor();
		int AddNeighbor( int i, int j, float d );
		
		// Smoothed Particle Hydrodynamics		
		void ComputePressureGrid ();			// O(kn) - spatial grid
		void ComputeForceGrid ();				// O(kn) - spatial grid
		void ComputeForceGridNC ();				// O(cn) - neighbor table		
		

		// GPU Support functions
		void AllocatePackBuf ();
		void PackParticles ();
		void UnpackParticles ();

		//void SPH_ComputePressureSlow ();			// O(n^2)	
		//void SPH_ComputeForceSlow ();				// O(n^2)
		//void SPH_ComputeForceGrid ();				// O(kn) - spatial grid

		// Recording
		void StartRecord ();
		void StartPlayback ( int p );
		void Record ();
		std::string getFilename ( int n );
		int getLastRecording ();
		int getMode ()		{ return m_Param[PMODE]; }
		std::string getModeStr ();
		void getModeClr ();

		// Parameters			
		void SetParam (int p, float v );
		void SetParam (int p, int v )		{ m_Param[p] = (float) v; }
		float GetParam ( int p )			{ return (float) m_Param[p]; }
		float SetParam ( int p, float v, float mn, float mx )	{ m_Param[p] = v ; if ( m_Param[p] > mx ) m_Param[p] = mn; return m_Param[p];}
		float IncParam ( int p, float v, float mn, float mx )	{ 
			m_Param[p] += v; 
			if ( m_Param[p] < mn ) m_Param[p] = mn; 
			if ( m_Param[p] > mx ) m_Param[p] = mn; 
			return m_Param[p];
		}
		Vector3DF GetVec ( int p )			{ return m_Vec[p]; }
		void SetVec ( int p, Vector3DF v );
		void Toggle ( int p )				{ m_Toggle[p] = !m_Toggle[p]; }		
		bool GetToggle ( int p )			{ return m_Toggle[p]; }
		std::string		getSceneName ()		{ return mSceneName; }
		
	private:

		std::string				mSceneName;

		// Time
		int							m_Frame;		
		double						m_DT;
		double						m_Time;	

		// Simulation Parameters
		double						m_Param [ MAX_PARAM ];			// see defines above
		Vector3DF					m_Vec [ MAX_PARAM ];
		bool						m_Toggle [ MAX_PARAM ];

		// SPH Kernel functions
		double					m_R2, m_Poly6Kern, m_LapKern, m_SpikyKern;		

		// Particle Buffers
		int						mNumPoints;
		int						mMaxPoints;
		int						mGoodPoints;
		Vector3DF*				mPos;
		DWORD*					mClr;
		Vector3DF*				mVel;
		Vector3DF*				mVelEval;
		unsigned short*			mAge;
		float*					mPressure;
		float*					mDensity;
		Vector3DF*				mForce;
		uint*					mGridCell;
		uint*					mClusterCell;
		uint*					mGridNext;
		uint*					mNbrNdx;
		uint*					mNbrCnt;
		
		// Acceleration Grid
		uint*					m_Grid;
		uint*					m_GridCnt;
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

		char*					mPackBuf;
		int*					mPackGrid;

		int						mVBO[3];

		// Record/Playback
		int						mFileNum;
		std::string				mFileName;
		float					mFileSize;
		FILE*					mFP;
		int						mLastPoints;
		
		int						mSpherePnts;
		int						mTex[1];
		GLuint					instancingShader;

		// Selected particle
		int						mSelected;

		nvImg					mImg;


		// Saved results (for algorithm validation)
		uint*					mSaveNdx;
		uint*					mSaveCnt;
		uint*					mSaveNeighbors;

		// XML Settings file
		XmlSettings				xml;
	};	

#endif
