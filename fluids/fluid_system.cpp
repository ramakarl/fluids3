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

#include <assert.h>
#include <stdio.h>
#include <conio.h>

#include "app_perf.h"
#include "fluid_defs.h"
#include "fluid_system.h"

#ifdef BUILD_CUDA
	#include "fluid_system_host.cuh"
#endif

#define EPSILON			0.00001f			//for collision detection

void FluidSystem::TransferToCUDA ()
{ 
	CopyToCUDA ( (float*) mPos, (float*) mVel, (float*) mVelEval, (float*) mForce, mPressure, mDensity, mClusterCell, mGridNext, (char*) mClr ); 
}
void FluidSystem::TransferFromCUDA ()	
{
	CopyFromCUDA ( (float*) mPos, (float*) mVel, (float*) mVelEval, (float*) mForce, mPressure, mDensity, mClusterCell, mGridNext, (char*) mClr );
}

//------------------------------ Initialization
FluidSystem::FluidSystem ()
{
	mNumPoints = 0;
	mMaxPoints = 0;
	mPackBuf = 0x0;
	mPackGrid = 0x0;
	mFP = 0x0;

	mPos = 0x0;
	mClr = 0x0;
	mVel = 0x0;
	mVelEval = 0x0;
	mAge = 0x0;
	mPressure = 0x0;
	mDensity = 0x0;
	mForce = 0x0;
	mClusterCell = 0x0;
	mGridNext = 0x0;
	mNbrNdx = 0x0;
	mNbrCnt = 0x0;
	mSelected = -1;
	m_Grid = 0x0;
	m_GridCnt = 0x0;

	m_Frame = 0;
	
	m_NeighborTable = 0x0;
	m_NeighborDist = 0x0;
	
	m_Param [ PMODE ]		= RUN_CUDA_FULL;
	m_Param [ PEXAMPLE ]	= 1;
	m_Param [ PGRID_DENSITY ] = 2.0;
	m_Param [ PNUM ]		= 65536 * 128;


	m_Toggle [ PDEBUG ]		=	false;
	m_Toggle [ PUSE_GRID ]	=	false;
	m_Toggle [ PPROFILE ]	=	false;
	m_Toggle [ PCAPTURE ]   =	false;

	if ( !xml.Load ( "scene.xml" ) ) {
		app_printf ( "fluid", "ERROR: Problem loading scene.xml. Check formatting.\n" );
		exit(-1);
	}

}

void FluidSystem::Setup ( bool bStart )
{
	#ifdef TEST_PREFIXSUM
		TestPrefixSum ( 16*1024*1024 );		
		exit(-2);
	#endif

	m_Frame = 0;
	m_Time = 0;

	ClearNeighborTable ();
	mNumPoints = 0;
	
	SetupDefaultParams ();
	
	SetupExampleParams ( bStart );

	m_Param [PGRIDSIZE] = 2*m_Param[PSMOOTHRADIUS] / m_Param[PGRID_DENSITY];

	AllocateParticles ( m_Param[PNUM] );
	AllocatePackBuf ();
	
	SetupKernels ();
	
	SetupSpacing ();

	SetupAddVolume ( m_Vec[PINITMIN], m_Vec[PINITMAX], m_Param[PSPACING], 0.1, m_Param[PNUM] );													// Create the particles

	SetupGridAllocate ( m_Vec[PVOLMIN], m_Vec[PVOLMAX], m_Param[PSIMSCALE], m_Param[PGRIDSIZE], 1.0 );	// Setup grid

	#ifdef BUILD_CUDA

		FluidClearCUDA ();

		Sleep ( 500 );
		
		FluidSetupCUDA ( NumPoints(), m_GridSrch, *(int3*)& m_GridRes, *(float3*)& m_GridSize, *(float3*)& m_GridDelta, *(float3*)& m_GridMin, *(float3*)& m_GridMax, m_GridTotal, (int) m_Vec[PEMIT_RATE].x );

		Sleep ( 500 );

		Vector3DF grav = m_Vec[PPLANE_GRAV_DIR];
		FluidParamCUDA ( m_Param[PSIMSCALE], m_Param[PSMOOTHRADIUS], m_Param[PRADIUS], m_Param[PMASS], m_Param[PRESTDENSITY], *(float3*)& m_Vec[PBOUNDMIN], *(float3*)& m_Vec[PBOUNDMAX], m_Param[PEXTSTIFF], m_Param[PINTSTIFF], m_Param[PVISC], m_Param[PEXTDAMP], m_Param[PFORCE_MIN], m_Param[PFORCE_MAX], m_Param[PFORCE_FREQ], m_Param[PGROUND_SLOPE], grav.x, grav.y, grav.z, m_Param[PACCEL_LIMIT], m_Param[PVEL_LIMIT] );

		TransferToCUDA ();		// Initial transfer
	#endif
}

void FluidSystem::SetParam (int p, float v )
{
	// Update CPU
	m_Param[p] = v;
	// Update GPU
	Vector3DF grav = m_Vec[PPLANE_GRAV_DIR];
	FluidParamCUDA ( m_Param[PSIMSCALE], m_Param[PSMOOTHRADIUS], m_Param[PRADIUS], m_Param[PMASS], m_Param[PRESTDENSITY], *(float3*)& m_Vec[PBOUNDMIN], *(float3*)& m_Vec[PBOUNDMAX], m_Param[PEXTSTIFF], m_Param[PINTSTIFF], m_Param[PVISC], m_Param[PEXTDAMP], m_Param[PFORCE_MIN], m_Param[PFORCE_MAX], m_Param[PFORCE_FREQ], m_Param[PGROUND_SLOPE], grav.x, grav.y, grav.z, m_Param[PACCEL_LIMIT], m_Param[PVEL_LIMIT] );
}

void FluidSystem::SetVec ( int p, Vector3DF v )	
{ 
	// Update CPU
	m_Vec[p] = v; 
	// Update GPU
	Vector3DF grav = m_Vec[PPLANE_GRAV_DIR];
	FluidParamCUDA ( m_Param[PSIMSCALE], m_Param[PSMOOTHRADIUS], m_Param[PRADIUS], m_Param[PMASS], m_Param[PRESTDENSITY], *(float3*)& m_Vec[PBOUNDMIN], *(float3*)& m_Vec[PBOUNDMAX], m_Param[PEXTSTIFF], m_Param[PINTSTIFF], m_Param[PVISC], m_Param[PEXTDAMP], m_Param[PFORCE_MIN], m_Param[PFORCE_MAX], m_Param[PFORCE_FREQ], m_Param[PGROUND_SLOPE], grav.x, grav.y, grav.z, m_Param[PACCEL_LIMIT], m_Param[PVEL_LIMIT] );
}

void FluidSystem::Exit ()
{
	free ( mPos );
	free ( mClr );
	free ( mVel );
	free ( mVelEval );
	free ( mAge );
	free ( mPressure );
	free ( mDensity );
	free ( mForce );
	free ( mClusterCell );
	free ( mGridCell );
	free ( mGridNext );
	free ( mNbrNdx );
	free ( mNbrCnt );

	FluidClearCUDA();

	cudaExit ();


}


// Allocate particle memory
void FluidSystem::AllocateParticles ( int cnt )
{
	int nump = 0;		// number to copy from previous data

	Vector3DF* srcPos = mPos;
	mPos = (Vector3DF*)		malloc ( cnt*sizeof(Vector3DF) );
	if ( srcPos != 0x0 )	{ memcpy ( mPos, srcPos, nump *sizeof(Vector3DF)); free ( srcPos ); }

	DWORD* srcClr = mClr;	
	mClr = (DWORD*)			malloc ( cnt*sizeof(DWORD) );
	if ( srcClr != 0x0 )	{ memcpy ( mClr, srcClr, nump *sizeof(DWORD)); free ( srcClr ); }
	
	Vector3DF* srcVel = mVel;
	mVel = (Vector3DF*)		malloc ( cnt*sizeof(Vector3DF) );	
	if ( srcVel != 0x0 )	{ memcpy ( mVel, srcVel, nump *sizeof(Vector3DF)); free ( srcVel ); }

	Vector3DF* srcVelEval = mVelEval;
	mVelEval = (Vector3DF*)	malloc ( cnt*sizeof(Vector3DF) );	
	if ( srcVelEval != 0x0 ) { memcpy ( mVelEval, srcVelEval, nump *sizeof(Vector3DF)); free ( srcVelEval ); }

	unsigned short* srcAge = mAge;
	mAge = (unsigned short*) malloc ( cnt*sizeof(unsigned short) );
	if ( srcAge != 0x0 )	{ memcpy ( mAge, srcAge, nump *sizeof(unsigned short)); free ( srcAge ); }

	float* srcPress = mPressure;
	mPressure = (float*) malloc ( cnt*sizeof(float) );
	if ( srcPress != 0x0 ) { memcpy ( mPressure, srcPress, nump *sizeof(float)); free ( srcPress ); }	

	float* srcDensity = mDensity;
	mDensity = (float*) malloc ( cnt*sizeof(float) );
	if ( srcDensity != 0x0 ) { memcpy ( mDensity, srcDensity, nump *sizeof(float)); free ( srcDensity ); }	

	Vector3DF* srcForce = mForce;
	mForce = (Vector3DF*)	malloc ( cnt*sizeof(Vector3DF) );
	if ( srcForce != 0x0 )	{ memcpy ( mForce, srcForce, nump *sizeof(Vector3DF)); free ( srcForce ); }

	uint* srcCell = mClusterCell;
	mClusterCell = (uint*)	malloc ( cnt*sizeof(uint) );
	if ( srcCell != 0x0 )	{ memcpy ( mClusterCell, srcCell, nump *sizeof(uint)); free ( srcCell ); }

	uint* srcGCell = mGridCell;
	mGridCell = (uint*)	malloc ( cnt*sizeof(uint) );
	if ( srcGCell != 0x0 )	{ memcpy ( mGridCell, srcGCell, nump *sizeof(uint)); free ( srcGCell ); }

	uint* srcNext = mGridNext;
	mGridNext = (uint*)	malloc ( cnt*sizeof(uint) );
	if ( srcNext != 0x0 )	{ memcpy ( mGridNext, srcNext, nump *sizeof(uint)); free ( srcNext ); }
	
	uint* srcNbrNdx = mNbrNdx;
	mNbrNdx = (uint*)		malloc ( cnt*sizeof(uint) );
	if ( srcNbrNdx != 0x0 )	{ memcpy ( mNbrNdx, srcNbrNdx, nump *sizeof(uint)); free ( srcNbrNdx ); }
	
	uint* srcNbrCnt = mNbrCnt;
	mNbrCnt = (uint*)		malloc ( cnt*sizeof(uint) );
	if ( srcNbrCnt != 0x0 )	{ memcpy ( mNbrCnt, srcNbrCnt, nump *sizeof(uint)); free ( srcNbrCnt ); }

	m_Param[PSTAT_PMEM] = 68 * 2 * cnt;

	mMaxPoints = cnt;
}

int FluidSystem::AddParticle ()
{
	if ( mNumPoints >= mMaxPoints ) return -1;
	int n = mNumPoints;
	(mPos + n)->Set ( 0,0,0 );
	(mVel + n)->Set ( 0,0,0 );
	(mVelEval + n)->Set ( 0,0,0 );
	(mForce + n)->Set ( 0,0,0 );
	*(mPressure + n) = 0;
	*(mDensity + n) = 0;
	*(mGridNext + n) = -1;
	*(mClusterCell + n) = -1;
	
	mNumPoints++;
	return n;
}

void FluidSystem::SetupAddVolume ( Vector3DF min, Vector3DF max, float spacing, float offs, int total )
{
	Vector3DF pos;
	int n, p;
	float dx, dy, dz, x, y, z;
	int cntx, cnty, cntz;
	cntx = ceil( (max.x-min.x-offs) / spacing );
	cntz = ceil( (max.z-min.z-offs) / spacing );
	int cnt = cntx * cntz;
	int xp, yp, zp, c2;
	float odd;
	
	min += offs;
	max -= offs;

	dx = max.x-min.x;
	dy = max.y-min.y;
	dz = max.z-min.z;
		
	c2 = cnt/2;
	for (float y = min.y; y <= max.y; y += spacing ) {	
		for (int xz=0; xz < cnt; xz++ ) {
			
			x = min.x + (xz % int(cntx))*spacing;
			z = min.z + (xz / int(cntx))*spacing;
			p = AddParticle ();
			if ( p != -1 ) {
				(mPos+p)->Set ( x,y,z);
				Vector3DF clr ( (x-min.x)/dx, (y-min.y)/dy, (z-min.z)/dz );
				clr *= 0.8;
				clr += 0.2;				
				*(mClr+p) = COLORA( clr.x, clr.y, clr.z, 1); 
				//*(mClr+p) = COLORA( 0.25, +0.25 + (y-min.y)*.75/dy, 0.25 + (z-min.z)*.75/dz, 1);  // (x-min.x)/dx
			}
		}
	}	
	
	//--- Random positions
	/*
	for (int n=0; n < total; n++ ) {
		
		pos.Random ( min.x, max.x, min.y, max.y, min.z, max.z );

		p = AddParticle ();
		if ( p != -1 ) {
			*(mPos+p) = pos;
			Vector3DF clr ( (pos.x-min.x)/dx, (pos.y-min.y)/dy, (pos.z-min.z)/dz );
			clr *= 0.8;
			clr += 0.2;				
			*(mClr+p) = COLORA( clr.x, clr.y, clr.z, 1); 
			//*(mClr+p) = COLORA( 0.25, +0.25 + (y-min.y)*.75/dy, 0.25 + (z-min.z)*.75/dz, 1);  // (x-min.x)/dx
		}
	}
	*/
	
}

void FluidSystem::AddEmit ( float spacing )
{
	int p;
	Vector3DF dir;
	Vector3DF pos;
	float ang_rand, tilt_rand;
	float rnd = m_Vec[PEMIT_RATE].y * 0.15;	
	int x = (int) sqrt(m_Vec[PEMIT_RATE].y);

	for ( int n = 0; n < m_Vec[PEMIT_RATE].y; n++ ) {
		ang_rand = (float(rand()*2.0/RAND_MAX) - 1.0) * m_Vec[PEMIT_SPREAD].x;
		tilt_rand = (float(rand()*2.0/RAND_MAX) - 1.0) * m_Vec[PEMIT_SPREAD].y;
		dir.x = cos ( ( m_Vec[PEMIT_ANG].x + ang_rand) * DEGtoRAD ) * sin( ( m_Vec[PEMIT_ANG].y + tilt_rand) * DEGtoRAD ) * m_Vec[PEMIT_ANG].z;
		dir.y = sin ( ( m_Vec[PEMIT_ANG].x + ang_rand) * DEGtoRAD ) * sin( ( m_Vec[PEMIT_ANG].y + tilt_rand) * DEGtoRAD ) * m_Vec[PEMIT_ANG].z;
		dir.z = cos ( ( m_Vec[PEMIT_ANG].y + tilt_rand) * DEGtoRAD ) * m_Vec[PEMIT_ANG].z;
		pos = m_Vec[PEMIT_POS];
		pos.x += spacing * (n/x);
		pos.y += spacing * (n%x);
		
		p = AddParticle ();
		*(mPos+n) = pos;
		*(mVel+n) = dir;
		*(mVelEval+n) = dir;
		*(mAge+n) = 0;
		*(mClr+n) = COLORA ( m_Time/10.0, m_Time/5.0, m_Time /4.0, 1 );
	}
}

void FluidSystem::record ( int param, std::string name, Time& start )
{
	Time stop;
	stop.SetSystemTime ();
	stop = stop - start;
	m_Param [ param ] = stop.GetMSec();
//	if ( m_Toggle[PPROFILE] ) printf ("%s:  %s\n", name.c_str(), stop.GetReadableTime().c_str() );

}


void FluidSystem::RunSearchCPU ()
{
	Time start;
	// -- Insert particles on CPU 
	InsertParticles ();
	record ( PTIME_INSERT, "Insert CPU", start );
	// --- Neighbor Search
	start.SetSystemTime ();
	FindNbrsGrid ();
	record ( PTIME_SORT, "Search CPU", start );

}

void FluidSystem::RunValidate ()
{
	int valid = 0, bad = 0;
	// CPU results
	uint* cpu_gridcnt = m_GridCnt;
	int* cpu_gridoff = (int*) malloc ( m_GridTotal*sizeof(int) );	
	uint* cpu_grid = (uint*) malloc ( NumPoints()*sizeof(uint) );	
	// GPU results
	uint* gpu_gcell = (uint*) malloc ( NumPoints() * sizeof(uint) );
	uint* gpu_ccell = (uint*) malloc ( NumPoints() * sizeof(uint) );	
	int* gpu_gridcnt = (int*) malloc ( m_GridTotal*sizeof(int) );
	int* gpu_gridoff = (int*) malloc ( m_GridTotal*sizeof(int) );
	uint* gpu_grid = (uint*) malloc ( NumPoints() * sizeof(uint) );	
	
	int n=0, c=0;

	// Insert Particles. Determines grid cells, and cpu grid counts (m_GridCnt)
	app_printf ( "\nVALIDATE SIM\n" );
	app_printf ( "Insert particles:\n" );
	InsertParticles ();					
	TransferToCUDA ();
	InsertParticlesCUDA ( gpu_gcell, gpu_ccell, gpu_gridcnt );
	app_printf ( "CPU:\n"); for (n=0, c=0; n < NumPoints() && c < 20; n++) {app_printf ( "p: %d, cell: %d, cluster: %d\n", n, mGridCell[n], mClusterCell[n] ); c++;} 
	app_printf ( "GPU:\n"); for (n=0, c=0; n < NumPoints() && c < 20; n++) {app_printf ( "p: %d, cell: %d, cluster: %d\n", n, gpu_gcell[n], gpu_ccell[n] ); c++;} 
	for (n=0, valid=0, bad=0; n < NumPoints(); n++) if ( mGridCell[n]==gpu_gcell[n] ) valid++; else bad++; 
	app_printf ( "Insert particles. VALID %d, BAD %d.  \n", valid, bad );

	// Compare grid counts 
	app_printf ( "Grid Counts:\n" );	
	app_printf ( "CPU:\n"); for (n=0, c=0; n < m_GridTotal && c < 20; n++) if ( cpu_gridcnt[n]!=0 ) {app_printf ( "cell: %d, cnt: %d\n", n, (int) cpu_gridcnt[n] );c++;}
	app_printf ( "GPU:\n"); for (n=0, c=0; n < m_GridTotal && c < 20; n++) if ( gpu_gridcnt[n]!=0 ) {app_printf ( "cell: %d, cnt: %d\n", n, gpu_gridcnt[n] );c++;}	
	for (n=0, valid=0, bad=0; n < m_GridTotal; n++) if ( cpu_gridcnt[n]==gpu_gridcnt[n] ) valid++; else bad++; 
	app_printf ( "Grid Counts. VALID %d, BAD %d.  \n", valid, bad );	

	// Prefix Sum. Determine grid offsets.
	PrefixSumCellsCUDA ( gpu_gridoff );		// Prefix Sum on GPU
	app_printf ( "Prefix Sum:\n" );
	int sum = 0;
	for (n=0; n < m_GridTotal; n++) {		// Prefix Sum on CPU
		cpu_gridoff[n] = sum;
		sum += cpu_gridcnt[n];
	}
	app_printf ( "CPU:\n"); for (n=0, c=0; n < m_GridTotal && c < 20; n++) if ( cpu_gridcnt[n]!=0 ) {app_printf ( "cell: %d, cnt: %d, off: %d\n", n, (int) cpu_gridcnt[n], cpu_gridoff[n] );c++;}
	app_printf ( "GPU:\n"); for (n=0, c=0; n < m_GridTotal && c < 20; n++) if ( gpu_gridcnt[n]!=0 ) {app_printf ( "cell: %d, cnt: %d, off: %d\n", n, gpu_gridcnt[n], gpu_gridoff[n] );c++;}	
	for (n=0, valid=0, bad=0; n < m_GridTotal; n++) if ( cpu_gridoff[n]==gpu_gridoff[n] ) valid++; else bad++; 
	app_printf ( "Prefix Sum. VALID %d, BAD %d.  \n", valid, bad );	

	// Counting Sort. Reinsert particles to grid.
	CountingSortIndexCUDA ( gpu_grid );
	app_printf ( "Counting Sort:\n" );
	app_printf ( "GPU:\n"); 
	app_printf ( "CPU:\n"); 
	int gc = 0;
	for (n=0, c=0; n < NumPoints() && c < 20; n++) {
		while ( n >= cpu_gridoff[gc] && gc < m_GridTotal ) gc++; gc--;
		app_printf ( "ndx: %d, pnt: %d, cell:%d \n", n, cpu_grid[n], gc ); c++;
	}
	gc = 0;
	for (n=0, c=0; n < NumPoints() && c < 20; n++) {
		while ( n >= gpu_gridoff[gc] && gc < m_GridTotal ) gc++; gc--;
		app_printf ( "ndx: %d, pnt: %d, cell:%d \n", n, gpu_grid[n], gc ); c++;
	}
	int gs = 0;
	memset ( gpu_gridcnt, 0, m_GridTotal*sizeof(int) );		// temporary counts
	for (n=0; n < NumPoints(); n++) {		
		gs = mGridCell[n];
		c = cpu_gridoff[gs] + gpu_gridcnt[gs];	// global cell offsets
		cpu_grid [ c ] = n;						// global grid list
		gpu_gridcnt[ gs ]++;					// temporary counts.. should equal cpu_gridcnt again when done.
	}	
	for (n=0, valid=0, bad=0; n < NumPoints(); n++) if ( cpu_grid[n]==gpu_grid[n] ) valid++; else bad++; 
	app_printf ( "Counting Sort. DONE. ");	
	
	free ( gpu_grid );
	free ( gpu_gridoff );
	free ( gpu_gridcnt );
	free ( gpu_ccell );
	free ( gpu_gcell );
	
	free ( cpu_grid );
	free ( cpu_gridoff );	
}

void FluidSystem::RunSimulateCPUSlow ()
{
	Time start;
	start.SetSystemTime ();
	InsertParticles ();
	record ( PTIME_INSERT, "Insert CPU", start );			
	start.SetSystemTime ();
	//ComputePressureSlow ();
	record ( PTIME_PRESS, "Press CPU (Slow)", start );
	start.SetSystemTime ();
	//ComputeForceSlow ();
	record ( PTIME_FORCE, "Force CPU (Slow)", start );
	start.SetSystemTime ();
	Advance ();
	record ( PTIME_ADVANCE, "Advance CPU", start );
}

void FluidSystem::RunSimulateCPUGrid ()
{
	Time start;
	start.SetSystemTime ();
	PERF_PUSH ( "InsertCPU" );
	InsertParticles ();
	PERF_POP ();
	record ( PTIME_INSERT, "Insert CPU", start );			
	start.SetSystemTime ();
	PERF_PUSH ( "PressCPU" );
	ComputePressureGrid ();
	PERF_POP ();
	record ( PTIME_PRESS, "Press CPU", start );
	start.SetSystemTime ();
	PERF_PUSH ( "ForceCPU" );
	ComputeForceGrid ();
	PERF_POP ();
	record ( PTIME_FORCE, "Force CPU", start );
	start.SetSystemTime ();
	PERF_PUSH ( "AdvanceCPU" );
	Advance ();
	PERF_POP ();
	record ( PTIME_ADVANCE, "Advance CPU", start );
}

void FluidSystem::RunSimulateCUDARadix ()
{
	// Not completed yet
}

void FluidSystem::RunSimulateCUDAIndex ()
{
	Time start;
	start.SetSystemTime ();
	
	PERF_PUSH ( "InsertCUDA" );
	InsertParticlesCUDA ( 0x0, 0x0, 0x0 );
	record ( PTIME_INSERT, "Insert CUDA", start );			
	start.SetSystemTime ();
	PERF_POP ();
	
	PERF_PUSH ( "SortCUDA" );
	PrefixSumCellsCUDA ( 0x0 );
	CountingSortIndexCUDA ( 0x0 );
	record ( PTIME_SORT, "Index Sort CUDA", start );
	start.SetSystemTime ();
	PERF_POP ();
	
	PERF_PUSH ( "PressureCUDA" );
	ComputePressureCUDA();
	record ( PTIME_PRESS, "Press CUDA", start );		
	start.SetSystemTime ();
	PERF_POP ();
	
	PERF_PUSH ( "ForceCUDA" );
	ComputeForceCUDA ();	
	record ( PTIME_FORCE, "Force CUDA", start );
	start.SetSystemTime ();
	PERF_POP ();

	PERF_PUSH ( "AdvanceCUDA" );
	AdvanceCUDA ( m_Time, m_DT, m_Param[PSIMSCALE] );			
	record ( PTIME_ADVANCE, "Advance CUDA", start );
	PERF_POP ();

	TransferFromCUDA ();	// return for rendering			
}

void FluidSystem::RunSimulateCUDAFull ()
{
	Time start;
	start.SetSystemTime ();

	PERF_PUSH ( "InsertCUDA" );
	InsertParticlesCUDA ( 0x0, 0x0, 0x0 );
	record ( PTIME_INSERT, "Insert CUDA", start );			
	PERF_POP ();
	
	PERF_PUSH ( "SortCUDA" );
	start.SetSystemTime ();
	PrefixSumCellsCUDA ( 0x0 );
	CountingSortFullCUDA ( 0x0 );	
	record ( PTIME_SORT, "Full Sort CUDA", start );
	PERF_POP ();
	
	PERF_PUSH ( "PressureCUDA" );
	start.SetSystemTime ();
	ComputePressureCUDA();
	record ( PTIME_PRESS, "Press CUDA", start );		
	PERF_POP ();

	PERF_PUSH ( "ForceCUDA" );
	start.SetSystemTime ();
	ComputeForceCUDA ();	
	record ( PTIME_FORCE, "Force CUDA", start );
	PERF_POP ();

	PERF_PUSH ( "AdvanceCUDA" );
	start.SetSystemTime ();	
	AdvanceCUDA ( m_Time, m_DT, m_Param[PSIMSCALE] );			
	record ( PTIME_ADVANCE, "Advance CUDA", start );
	PERF_POP ();

	TransferFromCUDA ();	// return for rendering			
}

void FluidSystem::RunSimulateCUDACluster ()
{
	Time start;
	start.SetSystemTime ();
	InsertParticlesCUDA ( 0x0, 0x0, 0x0 );
	record ( PTIME_INSERT, "Insert CUDA", start );			
	start.SetSystemTime ();
	PrefixSumCellsCUDA ( 0x0 );
	CountingSortFullCUDA ( 0x0 );
	record ( PTIME_SORT, "Sort CUDA", start );			
	start.SetSystemTime ();
	CountActiveCUDA ();
	record ( PTIME_COUNT, "Count CUDA", start );			
	start.SetSystemTime ();
	ComputePressureGroupCUDA();
	record ( PTIME_PRESS, "Press CUDA (Cluster)", start );
	start.SetSystemTime ();
	ComputeForceCUDA ();	
	record ( PTIME_FORCE, "Force CUDA", start );
	start.SetSystemTime ();
	AdvanceCUDA ( m_Time, m_DT, m_Param[PSIMSCALE] );			
	record ( PTIME_ADVANCE, "Advance CUDA", start );
	TransferFromCUDA ();			
}


void FluidSystem::EmitParticles ()
{
	if ( m_Vec[PEMIT_RATE].x > 0 && (++m_Frame) % (int) m_Vec[PEMIT_RATE].x == 0 ) {
		float ss = m_Param [ PDIST ] / m_Param[ PSIMSCALE ];		// simulation scale (not Schutzstaffel)
		AddEmit ( ss ); 
	}
}


void FluidSystem::Run (int width, int height)
{
	// Clear sim timers
	m_Param[ PTIME_INSERT ] = 0.0;
	m_Param[ PTIME_SORT ] = 0.0;
	m_Param[ PTIME_COUNT ] = 0.0;
	m_Param[ PTIME_PRESS ] = 0.0;
	m_Param[ PTIME_FORCE ] = 0.0;
	m_Param[ PTIME_ADVANCE ] = 0.0;

	// Run	
	#ifdef TEST_VALIDATESIM
		m_Param[PMODE] = RUN_VALIDATE;
	#endif	

	switch ( (int) m_Param[PMODE] ) {
	case RUN_SEARCH:		RunSearchCPU();			break;
	case RUN_VALIDATE:		RunValidate();			break;
	case RUN_CPU_SLOW:		RunSimulateCPUSlow();	break;
	case RUN_CPU_GRID:		RunSimulateCPUGrid();	break;
	case RUN_CUDA_RADIX:	RunSimulateCUDARadix();	break;
	case RUN_CUDA_INDEX:	RunSimulateCUDAIndex();	break;
	case RUN_CUDA_FULL:	RunSimulateCUDAFull();	break;
	case RUN_CUDA_CLUSTER:	RunSimulateCUDACluster();	break;
	case RUN_PLAYBACK:		RunPlayback();			break;
	};

	/*if ( mMode == RUN_RECORD ) {
		start.SetSystemTime ();
		Record ();
		record ( PTIME_RECORD, "Record", start );
	}*/
	if ( m_Toggle[PCAPTURE] ) {
		CaptureVideo ( width, height );
	}

	m_Time += m_DT;
	m_Frame++;
}

void FluidSystem::AllocatePackBuf ()
{
	if ( mPackBuf != 0x0 ) free ( mPackBuf );	
	mPackBuf = (char*) malloc ( sizeof(Fluid) * mMaxPoints );
}

//------- NOT CURRENTLY USED
void FluidSystem::PackParticles ()
{
	// Bin particles in memory according to grid cells.
	// This is equivalent to a partial bucket sort, as a GPU radix sort is not necessary.

	int j;	
	char* dat = mPackBuf;
	int cnt = 0;

	for (int c=0; c < m_GridTotal; c++) {
		j = m_Grid[c];
		mPackGrid[c] = cnt;
		while ( j != -1 ) {
			*(Vector3DF*) dat = *(mPos+j);			dat += sizeof(Vector3DF);
			*(Vector3DF*) dat = *(mVel+j);			dat += sizeof(Vector3DF);
			*(Vector3DF*) dat = *(mVelEval+j);		dat += sizeof(Vector3DF);
			*(Vector3DF*) dat = *(mForce+j);		dat += sizeof(Vector3DF);
			*(float*) dat =		*(mPressure+j);		dat += sizeof(float);
			*(float*) dat =		*(mDensity+j);		dat += sizeof(float);
			*(int*) dat =		*(mClusterCell+j);	dat += sizeof(int);					// search cell
			*(int*) dat =		c;					dat += sizeof(int);					// container cell
			*(DWORD*) dat =		*(mClr+j);			dat += sizeof(DWORD);
			dat += sizeof(int);
			j = *(mGridNext+j);
			cnt++;
		}
	}
	mGoodPoints = cnt;

	//--- Debugging - Print packed particles
	/*printf ( "\nPACKED\n" );
	for (int n=cnt-30; n < cnt; n++ ) {
		dat = mPackBuf + n*sizeof(Fluid);
		printf ( " %d: %d, %d\n", n, *((int*) (dat+56)), *((int*) (dat+60)) );
	}*/	
}

//------- NOT CURRENTLY USED
void FluidSystem::UnpackParticles ()
{
	char* dat = mPackBuf;

	Vector3DF*  ppos =		mPos;
	Vector3DF*  pforce =	mForce;
	Vector3DF*  pvel =		mVel;
	Vector3DF*  pveleval =	mVelEval;
	float*		ppress =	mPressure;
	float*		pdens =		mDensity;
	DWORD*		pclr =		mClr;

	for (int n=0; n < mGoodPoints; n++ ) {
		*ppos++ =		*(Vector3DF*) dat;		dat += sizeof(Vector3DF);
		*pvel++ =		*(Vector3DF*) dat;		dat += sizeof(Vector3DF);
		*pveleval++ =	*(Vector3DF*) dat;		dat += sizeof(Vector3DF);
		*pforce++ =		*(Vector3DF*) dat;		dat += sizeof(Vector3DF);
		*ppress++ =		*(float*) dat;			dat += sizeof(float);
		*pdens++ =		*(float*) dat;			dat += sizeof(float);		
		dat += sizeof(int);
		dat += sizeof(int);
		*pclr++ =		*(DWORD*) dat;			dat += sizeof(DWORD);
		dat += sizeof(int);
	}
}

void FluidSystem::DebugPrintMemory ()
{
	int psize = 4*sizeof(Vector3DF) + sizeof(DWORD) + sizeof(unsigned short) + 2*sizeof(float) + sizeof(int) + sizeof(int)+sizeof(int);
	int gsize = 2*sizeof(int);
	int nsize = sizeof(int) + sizeof(float);
		
	app_printf ( "MEMORY:\n");
	app_printf ( "  Fluid (size):			%d bytes\n", sizeof(Fluid) );
	app_printf ( "  Particles:              %d, %f MB (%f)\n", mNumPoints, (psize*mNumPoints)/1048576.0, (psize*mMaxPoints)/1048576.0);
	app_printf ( "  Acceleration Grid:      %d, %f MB\n",	   m_GridTotal, (gsize*m_GridTotal)/1048576.0 );
	app_printf ( "  Acceleration Neighbors: %d, %f MB (%f)\n", m_NeighborNum, (nsize*m_NeighborNum)/1048576.0, (nsize*m_NeighborMax)/1048576.0 );
	
}

void FluidSystem::DrawDomain ()
{
	Vector3DF min, max;
	min = m_Vec[PVOLMIN];
	max = m_Vec[PVOLMAX];
	
	glColor3f ( 0.0, 0.0, 1.0 );
	glBegin ( GL_LINES );
	glVertex3f ( min.x, min.y, min.z );	glVertex3f ( max.x, min.y, min.z );
	glVertex3f ( min.x, max.y, min.z );	glVertex3f ( max.x, max.y, min.z );
	glVertex3f ( min.x, min.y, min.z );	glVertex3f ( min.x, max.y, min.z );
	glVertex3f ( max.x, min.y, min.z );	glVertex3f ( max.x, max.y, min.z );
	glEnd ();
}

void FluidSystem::Advance ()
{
	Vector3DF norm, z;
	Vector3DF dir, accel;
	Vector3DF vnext;
	Vector3DF bmin, bmax;
	Vector4DF clr;
	double adj;
	float AL, AL2, SL, SL2, ss, radius;
	float stiff, damp, speed, diff; 
	
	AL = m_Param[PACCEL_LIMIT];	AL2 = AL*AL;
	SL = m_Param[PVEL_LIMIT];	SL2 = SL*SL;
	
	stiff = m_Param[PEXTSTIFF];
	damp = m_Param[PEXTDAMP];
	radius = m_Param[PRADIUS];
	bmin = m_Vec[PBOUNDMIN];
	bmax = m_Vec[PBOUNDMAX];
	ss = m_Param[PSIMSCALE];

	// Get particle buffers
	Vector3DF*	ppos = mPos;
	Vector3DF*	pvel = mVel;
	Vector3DF*	pveleval = mVelEval;
	Vector3DF*	pforce = mForce;
	DWORD*		pclr = mClr;
	float*		ppress = mPressure;
	float*		pdensity = mDensity;

	// Advance each particle
	for ( int n=0; n < NumPoints(); n++ ) {

		if ( mGridCell[n] == GRID_UNDEF) continue;

		// Compute Acceleration		
		accel = *pforce;
		accel *= m_Param[PMASS];
	
		// Boundary Conditions
		// Y-axis walls
		diff = radius - ( ppos->y - (bmin.y+ (ppos->x-bmin.x)*m_Param[PGROUND_SLOPE] ) )*ss;
		if (diff > EPSILON ) {			
			norm.Set ( -m_Param[PGROUND_SLOPE], 1.0 - m_Param[PGROUND_SLOPE], 0 );
			adj = stiff * diff - damp * norm.Dot ( *pveleval );
			accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;
		}		
		diff = radius - ( bmax.y - ppos->y )*ss;
		if (diff > EPSILON) {
			norm.Set ( 0, -1, 0 );
			adj = stiff * diff - damp * norm.Dot ( *pveleval );
			accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;
		}		
		
		// X-axis walls
		if ( !m_Toggle[PWRAP_X] ) {
			diff = radius - ( ppos->x - (bmin.x + (sin(m_Time*m_Param[PFORCE_FREQ])+1)*0.5 * m_Param[PFORCE_MIN]) )*ss;	
			//diff = 2 * radius - ( p->pos.x - min.x + (sin(m_Time*10.0)-1) * m_Param[FORCE_XMIN_SIN] )*ss;	
			if (diff > EPSILON ) {
				norm.Set ( 1.0, 0, 0 );
				adj = (m_Param[ PFORCE_MIN ]+1) * stiff * diff - damp * norm.Dot ( *pveleval ) ;
				accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;					
			}

			diff = radius - ( (bmax.x - (sin(m_Time*m_Param[PFORCE_FREQ])+1)*0.5* m_Param[PFORCE_MAX]) - ppos->x )*ss;	
			if (diff > EPSILON) {
				norm.Set ( -1, 0, 0 );
				adj = (m_Param[ PFORCE_MAX ]+1) * stiff * diff - damp * norm.Dot ( *pveleval );
				accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;
			}
		}

		// Z-axis walls
		diff = radius - ( ppos->z - bmin.z )*ss;			
		if (diff > EPSILON) {
			norm.Set ( 0, 0, 1 );
			adj = stiff * diff - damp * norm.Dot ( *pveleval );
			accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;
		}
		diff = radius - ( bmax.z - ppos->z )*ss;
		if (diff > EPSILON) {
			norm.Set ( 0, 0, -1 );
			adj = stiff * diff - damp * norm.Dot ( *pveleval );
			accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;
		}
		

		// Wall barrier
		if ( m_Toggle[PWALL_BARRIER] ) {
			diff = 2 * radius - ( ppos->x - 0 )*ss;					
			if (diff < 2*radius && diff > EPSILON && fabs(ppos->y) < 3 && ppos->z < 10) {
				norm.Set ( 1.0, 0, 0 );
				adj = 2*stiff * diff - damp * norm.Dot ( *pveleval ) ;	
				accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;					
			}
		}
		
		// Levy barrier
		if ( m_Toggle[PLEVY_BARRIER] ) {
			diff = 2 * radius - ( ppos->x - 0 )*ss;					
			if (diff < 2*radius && diff > EPSILON && fabs(ppos->y) > 5 && ppos->z < 10) {
				norm.Set ( 1.0, 0, 0 );
				adj = 2*stiff * diff - damp * norm.Dot ( *pveleval ) ;	
				accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;					
			}
		}
		// Drain barrier
		if ( m_Toggle[PDRAIN_BARRIER] ) {
			diff = 2 * radius - ( ppos->z - bmin.z-15 )*ss;
			if (diff < 2*radius && diff > EPSILON && (fabs(ppos->x)>3 || fabs(ppos->y)>3) ) {
				norm.Set ( 0, 0, 1);
				adj = stiff * diff - damp * norm.Dot ( *pveleval );
				accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;
			}
		}

		// Plane gravity
		accel += m_Vec[PPLANE_GRAV_DIR];

		// Point gravity
		if ( m_Param[PPOINT_GRAV_AMT] > 0 ) {
			norm.x = ( ppos->x - m_Vec[PPOINT_GRAV_POS].x );
			norm.y = ( ppos->y - m_Vec[PPOINT_GRAV_POS].y );
			norm.z = ( ppos->z - m_Vec[PPOINT_GRAV_POS].z );
			norm.Normalize ();
			norm *= m_Param[PPOINT_GRAV_AMT];
			accel -= norm;
		}

		// Acceleration limiting 
		speed = accel.x*accel.x + accel.y*accel.y + accel.z*accel.z;
		if ( speed > AL2 ) {
			accel *= AL / sqrt(speed);
		}		

		// Velocity limiting 
		speed = pvel->x*pvel->x + pvel->y*pvel->y + pvel->z*pvel->z;
		if ( speed > SL2 ) {
			speed = SL2;
			(*pvel) *= SL / sqrt(speed);
		}		

		// Leapfrog Integration ----------------------------
		vnext = accel;							
		vnext *= m_DT;
		vnext += *pvel;						// v(t+1/2) = v(t-1/2) + a(t) dt

		*pveleval = *pvel;
		*pveleval += vnext;
		*pveleval *= 0.5;					// v(t+1) = [v(t-1/2) + v(t+1/2)] * 0.5		used to compute forces later
		*pvel = vnext;
		vnext *= m_DT/ss;
		*ppos += vnext;						// p(t+1) = p(t) + v(t+1/2) dt

		/*if ( m_Param[PCLR_MODE]==1.0 ) {
			adj = fabs(vnext.x)+fabs(vnext.y)+fabs(vnext.z) / 7000.0;
			adj = (adj > 1.0) ? 1.0 : adj;
			*pclr = COLORA( 0, adj, adj, 1 );
		}
		if ( m_Param[PCLR_MODE]==2.0 ) {
			float v = 0.5 + ( *ppress / 1500.0); 
			if ( v < 0.1 ) v = 0.1;
			if ( v > 1.0 ) v = 1.0;
			*pclr = COLORA ( v, 1-v, 0, 1 );
		}*/
		if ( speed > SL2*0.1) {
			adj = SL2*0.1;
			clr.fromClr ( *pclr );
			clr += float(2/255.0);
			clr.Clamp ( 1, 1, 1, 1);
			*pclr = clr.toClr();
		}
		if ( speed < 0.01 ) {
			clr.fromClr ( *pclr);
			clr.x -= float(1/255.0);		if ( clr.x < 0.2 ) clr.x = 0.2;
			clr.y -= float(1/255.0);		if ( clr.y < 0.2 ) clr.y = 0.2;
			*pclr = clr.toClr();
		}
		
		// Euler integration -------------------------------
		/* accel += m_Gravity;
		accel *= m_DT;
		p->vel += accel;				// v(t+1) = v(t) + a(t) dt
		p->vel_eval += accel;
		p->vel_eval *= m_DT/d;
		p->pos += p->vel_eval;
		p->vel_eval = p->vel;  */	


		if ( m_Toggle[PWRAP_X] ) {
			diff = ppos->x - (m_Vec[PBOUNDMIN].x + 2);			// -- Simulates object in center of flow
			if ( diff <= 0 ) {
				ppos->x = (m_Vec[PBOUNDMAX].x - 2) + diff*2;				
				ppos->z = 10;
			}
		}	

		ppos++;
		pvel++;
		pveleval++;
		pforce++;
		pclr++;
		ppress++;
		pdensity++;
	}

}

void FluidSystem::ClearNeighborTable ()
{
	if ( m_NeighborTable != 0x0 )	free (m_NeighborTable);
	if ( m_NeighborDist != 0x0)		free (m_NeighborDist );
	m_NeighborTable = 0x0;
	m_NeighborDist = 0x0;
	m_NeighborNum = 0;
	m_NeighborMax = 0;
}

void FluidSystem::ResetNeighbors ()
{
	m_NeighborNum = 0;
}

// Allocate new neighbor tables, saving previous data
int FluidSystem::AddNeighbor ()
{
	if ( m_NeighborNum >= m_NeighborMax ) {
		m_NeighborMax = 2*m_NeighborMax + 1;		
		int* saveTable = m_NeighborTable;
		m_NeighborTable = (int*) malloc ( m_NeighborMax * sizeof(int) );
		if ( saveTable != 0x0 ) {
			memcpy ( m_NeighborTable, saveTable, m_NeighborNum*sizeof(int) );
			free ( saveTable );
		}
		float* saveDist = m_NeighborDist;
		m_NeighborDist = (float*) malloc ( m_NeighborMax * sizeof(float) );
		if ( saveDist != 0x0 ) {
			memcpy ( m_NeighborDist, saveDist, m_NeighborNum*sizeof(int) );
			free ( saveDist );
		}
	};
	m_NeighborNum++;
	return m_NeighborNum-1;
}

void FluidSystem::ClearNeighbors ( int i )
{
	*(mNbrCnt+i) = 0;
}

int FluidSystem::AddNeighbor( int i, int j, float d )
{
	int k = AddNeighbor();
	m_NeighborTable[k] = j;
	m_NeighborDist[k] = d;
	if (*(mNbrCnt+i) == 0 ) *(mNbrNdx+i) = k;
	(*(mNbrCnt+i))++;
	return k;
}

// Ideal grid cell size (gs) = 2 * smoothing radius = 0.02*2 = 0.04
// Ideal domain size = k*gs/d = k*0.02*2/0.005 = k*8 = {8, 16, 24, 32, 40, 48, ..}
//    (k = number of cells, gs = cell size, d = simulation scale)
void FluidSystem::SetupGridAllocate ( Vector3DF min, Vector3DF max, float sim_scale, float cell_size, float border )
{
	float world_cellsize = cell_size / sim_scale;
	
	m_GridMin = min;
	m_GridMax = max;
	m_GridSize = m_GridMax;
	m_GridSize -= m_GridMin;
	m_GridRes.x = ceil ( m_GridSize.x / world_cellsize );		// Determine grid resolution
	m_GridRes.y = ceil ( m_GridSize.y / world_cellsize );
	m_GridRes.z = ceil ( m_GridSize.z / world_cellsize );
	m_GridSize.x = m_GridRes.x * cell_size / sim_scale;				// Adjust grid size to multiple of cell size
	m_GridSize.y = m_GridRes.y * cell_size / sim_scale;
	m_GridSize.z = m_GridRes.z * cell_size / sim_scale;
	m_GridDelta = m_GridRes;		// delta = translate from world space to cell #
	m_GridDelta /= m_GridSize;
	
	m_GridTotal = (int)(m_GridRes.x * m_GridRes.y * m_GridRes.z);

	// Allocate grid
	if ( m_Grid == 0x0 ) free (m_Grid);
	if ( m_GridCnt == 0x0 ) free (m_GridCnt);
	m_Grid = (uint*) malloc ( sizeof(uint*) * m_GridTotal );
	m_GridCnt = (uint*) malloc ( sizeof(uint*) * m_GridTotal );
	memset ( m_Grid, GRID_UCHAR, m_GridTotal*sizeof(uint) );
	memset ( m_GridCnt, GRID_UCHAR, m_GridTotal*sizeof(uint) );

	m_Param[PSTAT_GMEM] = 12 * m_GridTotal;		// Grid memory used

	// Number of cells to search:
	// n = (2r / w) +1,  where n = 1D cell search count, r = search radius, w = world cell width
	//
	m_GridSrch =  floor(2*(m_Param[PSMOOTHRADIUS]/sim_scale) / world_cellsize) + 1;
	if ( m_GridSrch < 2 ) m_GridSrch = 2;
	m_GridAdjCnt = m_GridSrch * m_GridSrch * m_GridSrch ;			// 3D search count = n^3, e.g. 2x2x2=8, 3x3x3=27, 4x4x4=64

	if ( m_GridSrch > 6 ) {
		app_printf ( "ERROR: Neighbor search is n > 6. \n " );
		exit(-1);
	}

	int cell = 0;
	for (int y=0; y < m_GridSrch; y++ ) 
		for (int z=0; z < m_GridSrch; z++ ) 
			for (int x=0; x < m_GridSrch; x++ ) 
				m_GridAdj[cell++] = ( y*m_GridRes.z + z )*m_GridRes.x +  x ;			// -1 compensates for ndx 0=empty
				

	app_printf ( "Adjacency table (CPU) \n");
	for (int n=0; n < m_GridAdjCnt; n++ ) {
		app_printf ( "  ADJ: %d, %d\n", n, m_GridAdj[n] );
	}

	if ( mPackGrid != 0x0 ) free ( mPackGrid );
	mPackGrid = (int*) malloc ( sizeof(int) * m_GridTotal );

	
}

int FluidSystem::getGridCell ( int p, Vector3DI& gc )
{
	return getGridCell ( *(mPos+p), gc );
}
int FluidSystem::getGridCell ( Vector3DF& pos, Vector3DI& gc )
{
	gc.x = (int)( (pos.x - m_GridMin.x) * m_GridDelta.x);			// Cell in which particle is located
	gc.y = (int)( (pos.y - m_GridMin.y) * m_GridDelta.y);
	gc.z = (int)( (pos.z - m_GridMin.z) * m_GridDelta.z);		
	return (int)( (gc.y*m_GridRes.z + gc.z)*m_GridRes.x + gc.x);		
}
Vector3DI FluidSystem::getCell ( int c )
{
	Vector3DI gc;
	int xz = m_GridRes.x*m_GridRes.z;
	gc.y = c / xz;				c -= gc.y*xz;
	gc.z = c / m_GridRes.x;		c -= gc.z*m_GridRes.x;
	gc.x = c;
	return gc;
}

void FluidSystem::InsertParticles ()
{
	int gs;
	int gx, gy, gz;
	
	// Reset all grid pointers and neighbor tables to empty
	memset ( mGridNext,		GRID_UCHAR, NumPoints()*sizeof(uint) );
	memset ( mGridCell,		GRID_UCHAR, NumPoints()*sizeof(uint) );
	memset ( mClusterCell,	GRID_UCHAR, NumPoints()*sizeof(uint) );

	// Reset all grid cells to empty
	memset ( m_Grid,		GRID_UCHAR, m_GridTotal*sizeof(uint) );
	memset ( m_GridCnt,				 0, m_GridTotal*sizeof(uint) );

	// Insert each particle into spatial grid
	Vector3DI gc;
	Vector3DF* ppos =	mPos;
	uint* pgrid =		mGridCell;
	uint* pcell =		mClusterCell;
	uint* pnext =		mGridNext;	

	float poff = m_Param[PSMOOTHRADIUS] / m_Param[PSIMSCALE];

	int ns = pow ( m_GridAdjCnt, 1/3.0 );
	register int xns, yns, zns;
	xns = m_GridRes.x - m_GridSrch;
	yns = m_GridRes.y - m_GridSrch;
	zns = m_GridRes.z - m_GridSrch;

	m_Param[ PSTAT_OCCUPY ] = 0.0;
	m_Param [ PSTAT_GRIDCNT ] = 0.0;

	for ( int n=0; n < NumPoints(); n++ ) {
		gs = getGridCell ( *ppos, gc );
		if ( gc.x >= 1 && gc.x <= xns && gc.y >= 1 && gc.y <= yns && gc.z >= 1 && gc.z <= zns ) {
			// put current particle at head of grid cell, pointing to next in list (previous head of cell)
			*pgrid = gs;
			*pnext = m_Grid[gs];				
			if ( *pnext == GRID_UNDEF ) m_Param[ PSTAT_OCCUPY ] += 1.0;
			m_Grid[gs] = n;
			m_GridCnt[gs]++;
			m_Param [ PSTAT_GRIDCNT ] += 1.0;
			/* -- 1/2 cell offset search method
			gx = (int)( (-poff + ppos->x - m_GridMin.x) * m_GridDelta.x);	
			if ( gx < 0 ) gx = 0;
			if ( gx > m_GridRes.x-2 ) gx = m_GridRes.x-2;
			gy = (int)( (-poff + ppos->y - m_GridMin.y) * m_GridDelta.y);
			if ( gy < 0 ) gy = 0;
			if ( gy > m_GridRes.y-2 ) gx = m_GridRes.y-2;
			gz = (int)( (-poff + ppos->z - m_GridMin.z) * m_GridDelta.z);
			if ( gz < 0 ) gz = 0;
			if ( gz > m_GridRes.z-2 ) gz = m_GridRes.z-2;
			*pcell = (int)( (gy*m_GridRes.z + gz)*m_GridRes.x + gx) ;	// Cell in which to start 2x2x2 search*/
		} else {			
			Vector3DF vel, ve;
			vel = *(mVel + n);
			ve = *(mVelEval + n);
			float pr, dn;
			pr = *(mPressure + n);
			dn = *(mDensity + n);
			//printf ( "WARNING: Out of Bounds: %d, P<%f %f %f>, V<%f %f %f>, prs:%f, dns:%f\n", n, ppos->x, ppos->y, ppos->z, vel.x, vel.y, vel.z, pr, dn );
			//ppos->x = -1; ppos->y = -1; ppos->z = -1;
		}
		pgrid++;
		ppos++;
		pnext++;
		pcell++;
	}

	// STATS
	/*m_Param[ PSTAT_OCCUPY ] = 0;
	m_Param[ PSTAT_GRIDCNT ] = 0;
	for (int n=0; n < m_GridTotal; n++) {
		if ( m_GridCnt[n] > 0 )  m_Param[ PSTAT_OCCUPY ] += 1.0;
		m_Param [ PSTAT_GRIDCNT ] += m_GridCnt[n];
	}*/
}

void FluidSystem::SaveResults ()
{
	if ( mSaveNdx != 0x0 ) free ( mSaveNdx );
	if ( mSaveCnt != 0x0 ) free ( mSaveCnt );
	if ( mSaveNeighbors != 0x0 )	free ( mSaveNeighbors );

	mSaveNdx = (uint*) malloc ( sizeof(uint) * NumPoints() );
	mSaveCnt = (uint*) malloc ( sizeof(uint) * NumPoints() );
	mSaveNeighbors = (uint*) malloc ( sizeof(uint) * m_NeighborNum );
	memcpy ( mSaveNdx, mNbrNdx, sizeof(uint) * NumPoints() );
	memcpy ( mSaveCnt, mNbrCnt, sizeof(uint) * NumPoints() );
	memcpy ( mSaveNeighbors, m_NeighborTable, sizeof(uint) * m_NeighborNum );
}

void FluidSystem::ValidateResults ()
{
//	Setup ();
	app_printf ( "VALIDATION:\n" );
	InsertParticles ();	app_printf ( "  Insert. OK\n" );	
	FindNbrsSlow ();	app_printf ( "  True Neighbors. OK\n" );	
	SaveResults ();		app_printf ( "  Save Results. OK\n" );	
	Run (0,0);			app_printf ( "  New Algorithm. OK\n" );
	
	// Quick validation
	app_printf ( "  Compare...\n" );
	int bad = 0;
	for (int n=0; n < NumPoints(); n++ ) {
		if ( *(mSaveCnt+n) != *(mNbrCnt+n) ) {
			*(mClr+n) = COLORA(1.0,0,0,0.9);
			app_printf ( "Error %d, correct: %d, cnt: %d\n", n, *(mSaveCnt+n), *(mNbrCnt+n) );
		}
	}
	if ( bad == 0 ) {
		app_printf ( "  OK!\n" );
	}
}



void FluidSystem::FindNbrsSlow ()
{
	// O(n^2)
	// Does not require grid

	Vector3DF dst;
	float dsq;
	float d2 = m_Param[PSIMSCALE]*m_Param[PSIMSCALE];
	
	ResetNeighbors ();

	Vector3DF *ipos, *jpos;
	ipos = mPos;
	for (int i=0; i < NumPoints(); i++ ) {
		jpos = mPos;
		ClearNeighbors ( i );
		for (int j=0; j < NumPoints(); j++ ) {
			dst = *ipos;
			dst -= *jpos;
			dsq = d2*(dst.x*dst.x + dst.y*dst.y + dst.z*dst.z);
			if ( i != j && dsq <= m_R2 ) {
				AddNeighbor( i, j, sqrt(dsq) );
			}
			jpos++;
		}
		ipos++;
	}
}

void FluidSystem::FindNbrsGrid ()
{
	// O(n^2)
	// Does not require grid

	Vector3DF dst;
	float dsq;
	int j;
	int nadj = (m_GridRes.z + 1)*m_GridRes.x + 1;
	float d2 = m_Param[PSIMSCALE]*m_Param[PSIMSCALE];
	
	ResetNeighbors ();

	Vector3DF *ipos, *jpos;
	ipos = mPos;
	for (int i=0; i < NumPoints(); i++ ) {
		ClearNeighbors ( i );
		
		if ( *(mGridCell+i) != GRID_UNDEF ) {
			for (int cell=0; cell < m_GridAdjCnt; cell++) {
				j = m_Grid [ *(mGridCell+i) - nadj + m_GridAdj[cell] ] ;
				while ( j != GRID_UNDEF ) {
					if ( i==j ) { j = *(mGridNext+j); continue; }
					dst = *ipos;
					dst -= *(mPos+j);
					dsq = d2*(dst.x*dst.x + dst.y*dst.y + dst.z*dst.z);
					if ( dsq <= m_R2 ) {
						AddNeighbor( i, j, sqrt(dsq) );
					}
					j = *(mGridNext+j);
				}
			}
		}
		ipos++;
	}
}


// Compute Pressures - Using spatial grid, and also create neighbor table
void FluidSystem::ComputePressureGrid ()
{
	int i, j, cnt = 0;
	int nbr;
	float dx, dy, dz, sum, dsq, c;
	float d = m_Param[PSIMSCALE];
	float d2 = d*d;
	float radius = m_Param[PSMOOTHRADIUS] / m_Param[PSIMSCALE];
	
	Vector3DF*	ipos	= mPos;
	float*		ipress	= mPressure;
	float*		idensity = mDensity;
	uint*		inbr	= mNbrNdx;
	uint*		inbrcnt = mNbrCnt;	

	Vector3DF	dst;
	int			nadj = (m_GridRes.z + 1)*m_GridRes.x + 1;
	int*		jnext;
	
	int nbrcnt = 0;
	int srch = 0;

	for ( i=0; i < NumPoints(); i++ ) {

		sum = 0.0;

		if ( *(mGridCell+i) != GRID_UNDEF ) {
			for (int cell=0; cell < m_GridAdjCnt; cell++) {
				j = m_Grid [  *(mGridCell+i) - nadj + m_GridAdj[cell] ] ;
				while ( j != GRID_UNDEF ) {
					if ( i==j ) { j = *(mGridNext+j) ; continue; }
					dst = *(mPos + j);
					dst -= *ipos;
					dsq = d2*(dst.x*dst.x + dst.y*dst.y + dst.z*dst.z);
					if ( dsq <= m_R2 ) {
						c =  m_R2 - dsq;
						sum += c * c * c;
						nbrcnt++;
						/*nbr = AddNeighbor();			// get memory for new neighbor						
						*(m_NeighborTable + nbr) = j;
						*(m_NeighborDist + nbr) = sqrt(dsq);
						inbr->num++;*/
					}
					srch++;
					j = *(mGridNext+j) ;
				}
			}
		}
		*idensity = sum * m_Param[PMASS] * m_Poly6Kern ;	
		*ipress = ( *idensity - m_Param[PRESTDENSITY] ) * m_Param[PINTSTIFF];		
		*idensity = 1.0f / *idensity;

		ipos++;
		idensity++;
		ipress++;
	}
	// Stats:
	m_Param [ PSTAT_NBR ] = float(nbrcnt);
	m_Param [ PSTAT_SRCH ] = float(srch);
	if ( m_Param[PSTAT_NBR] > m_Param [ PSTAT_NBRMAX ] ) m_Param [ PSTAT_NBRMAX ] = m_Param[PSTAT_NBR];
	if ( m_Param[PSTAT_SRCH] > m_Param [ PSTAT_SRCHMAX ] ) m_Param [ PSTAT_SRCHMAX ] = m_Param[PSTAT_SRCH];
}

// Compute Forces - Using spatial grid with saved neighbor table. Fastest.
void FluidSystem::ComputeForceGrid ()
{
	Vector3DF force;
	register float pterm, vterm, dterm;
	int i, j, nbr;
	float c, d;
	float dx, dy, dz;
	float mR, mR2, visc;	

	d = m_Param[PSIMSCALE];
	mR = m_Param[PSMOOTHRADIUS];
	visc = m_Param[PVISC];
	
	Vector3DF*	ipos = mPos;
	Vector3DF*	iveleval = mVelEval;
	Vector3DF*	iforce = mForce;
	float*		ipress = mPressure;
	float*		idensity = mDensity;
	
	int			jndx;
	Vector3DF	jpos;
	float		jdist;
	float		jpress;
	float		jdensity;
	Vector3DF	jveleval;
	float		dsq;
	float		d2 = d*d;
	int			nadj = (m_GridRes.z + 1)*m_GridRes.x + 1;

	for ( i=0; i < NumPoints(); i++ ) {

		iforce->Set ( 0, 0, 0 );

		if ( *(mGridCell+i) != GRID_UNDEF ) {
			for (int cell=0; cell < m_GridAdjCnt; cell++) {
				j = m_Grid [  *(mGridCell+i) - nadj + m_GridAdj[cell] ];
				while ( j != GRID_UNDEF ) {
					if ( i==j ) { j = *(mGridNext+j); continue; }
					jpos = *(mPos + j);
					dx = ( ipos->x - jpos.x);		// dist in cm
					dy = ( ipos->y - jpos.y);
					dz = ( ipos->z - jpos.z);
					dsq = d2*(dx*dx + dy*dy + dz*dz);
					if ( dsq <= m_R2 ) {

						jdist = sqrt(dsq);

						jpress = *(mPressure + j);
						jdensity = *(mDensity + j);
						jveleval = *(mVelEval + j);						
						dx = ( ipos->x - jpos.x);		// dist in cm
						dy = ( ipos->y - jpos.y);
						dz = ( ipos->z - jpos.z);
						c = (mR-jdist);
						pterm = d * -0.5f * c * m_SpikyKern * ( *ipress + jpress ) / jdist;
						dterm = c * (*idensity) * jdensity;
						vterm = m_LapKern * visc;
						iforce->x += ( pterm * dx + vterm * ( jveleval.x - iveleval->x) ) * dterm;
						iforce->y += ( pterm * dy + vterm * ( jveleval.y - iveleval->y) ) * dterm;
						iforce->z += ( pterm * dz + vterm * ( jveleval.z - iveleval->z) ) * dterm;
					}
					j = *(mGridNext+j);
				}
			}
		}
		ipos++;
		iveleval++;
		iforce++;
		ipress++;
		idensity++;
	}
}


// Compute Forces - Using spatial grid with saved neighbor table. Fastest.
void FluidSystem::ComputeForceGridNC ()
{
	Vector3DF force;
	register float pterm, vterm, dterm;
	int i, j, nbr;
	float c, d;
	float dx, dy, dz;
	float mR, mR2, visc;	

	d = m_Param[PSIMSCALE];
	mR = m_Param[PSMOOTHRADIUS];
	visc = m_Param[PVISC];
	
	Vector3DF*	ipos = mPos;
	Vector3DF*	iveleval = mVelEval;
	Vector3DF*	iforce = mForce;
	float*		ipress = mPressure;
	float*		idensity = mDensity;
	uint*		inbr =	mNbrNdx;
	uint*		inbrcnt = mNbrCnt;

	int			jndx;
	Vector3DF	jpos;
	float		jdist;
	float		jpress;
	float		jdensity;
	Vector3DF	jveleval;

	for ( i=0; i < NumPoints(); i++ ) {

		iforce->Set ( 0, 0, 0 );
		
		jndx = *inbr;
		for (int nbr=0; nbr < *inbrcnt; nbr++ ) {
			j = *(m_NeighborTable+jndx);
			jpos = *(mPos + j);
			jpress = *(mPressure + j);
			jdensity = *(mDensity + j);
			jveleval = *(mVelEval + j);
			jdist = *(m_NeighborDist + jndx);			
			dx = ( ipos->x - jpos.x);		// dist in cm
			dy = ( ipos->y - jpos.y);
			dz = ( ipos->z - jpos.z);
			c = ( mR - jdist );
			pterm = d * -0.5f * c * m_SpikyKern * ( *ipress + jpress ) / jdist;
			dterm = c * (*idensity) * jdensity;
			vterm = m_LapKern * visc;
			iforce->x += ( pterm * dx + vterm * ( jveleval.x - iveleval->x) ) * dterm;
			iforce->y += ( pterm * dy + vterm * ( jveleval.y - iveleval->y) ) * dterm;
			iforce->z += ( pterm * dz + vterm * ( jveleval.z - iveleval->z) ) * dterm;
			jndx++;
		}				
		ipos++;
		iveleval++;
		iforce++;
		ipress++;
		idensity++;
		inbr++;
	}
}


void FluidSystem::SetupRender ()
{
	glEnable ( GL_TEXTURE_2D );

	glGenTextures ( 1, (GLuint*) mTex );
	glBindTexture ( GL_TEXTURE_2D, mTex[0] );
	glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);	
	glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );	
	glPixelStorei( GL_UNPACK_ALIGNMENT, 4);	
	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB32F_ARB, 8, 8, 0, GL_RGB, GL_FLOAT, 0);

	glGenBuffersARB ( 3, (GLuint*) mVBO );

	// Construct a sphere in a VBO
	int udiv = 6;
	int vdiv = 6;
	float du = 180.0 / udiv;
	float dv = 360.0 / vdiv;
	float x,y,z, x1,y1,z1;

	float r = 1.0;

	Vector3DF* buf = (Vector3DF*) malloc ( sizeof(Vector3DF) * (udiv+2)*(vdiv+2)*2 );
	Vector3DF* dat = buf;
	
	mSpherePnts = 0;
	for ( float tilt=-90; tilt <= 90.0; tilt += du) {
		for ( float ang=0; ang <= 360; ang += dv) {
			x = sin ( ang*DEGtoRAD) * cos ( tilt*DEGtoRAD );
			y = cos ( ang*DEGtoRAD) * cos ( tilt*DEGtoRAD );
			z = sin ( tilt*DEGtoRAD ) ;
			x1 = sin ( ang*DEGtoRAD) * cos ( (tilt+du)*DEGtoRAD ) ;
			y1 = cos ( ang*DEGtoRAD) * cos ( (tilt+du)*DEGtoRAD ) ;
			z1 = sin ( (tilt+du)*DEGtoRAD );
		
			dat->x = x*r;
			dat->y = y*r;
			dat->z = z*r;
			dat++;
			dat->x = x1*r;
			dat->y = y1*r;
			dat->z = z1*r;
			dat++;
			mSpherePnts += 2;
		}
	}
	glBindBufferARB ( GL_ARRAY_BUFFER_ARB, mVBO[2] );
	glBufferDataARB ( GL_ARRAY_BUFFER_ARB, mSpherePnts*sizeof(Vector3DF), buf, GL_STATIC_DRAW_ARB);
	glVertexPointer ( 3, GL_FLOAT, 0, 0x0 );

	free ( buf );
		
	mImg.LoadPng ( "ball32.png" );
	mImg.UpdateTex ();

	// Enable Instacing shader
	//cgGLEnableProfile( vert_profile );
	//cgGLBindProgram ( cgVP );

	//cgGLEnableProfile( frag_profile );
	//cgGLBindProgram ( cgFP );
}


void FluidSystem::DrawCell ( int gx, int gy, int gz )
{
	Vector3DF gd (1, 1, 1);
	Vector3DF gc;
	gd /= m_GridDelta;		
	gc.Set ( (float) gx, (float) gy, (float) gz );
	gc /= m_GridDelta;
	gc += m_GridMin;
	glBegin ( GL_LINES );
	glVertex3f ( gc.x, gc.y, gc.z ); glVertex3f ( gc.x+gd.x, gc.y, gc.z );
	glVertex3f ( gc.x, gc.y+gd.y, gc.z ); glVertex3f ( gc.x+gd.x, gc.y+gd.y, gc.z );
	glVertex3f ( gc.x, gc.y, gc.z+gd.z ); glVertex3f ( gc.x+gd.x, gc.y, gc.z+gd.z );
	glVertex3f ( gc.x, gc.y+gd.y, gc.z+gd.z ); glVertex3f ( gc.x+gd.x, gc.y+gd.y, gc.z+gd.z );

	glVertex3f ( gc.x, gc.y, gc.z ); glVertex3f ( gc.x, gc.y+gd.y, gc.z );
	glVertex3f ( gc.x+gd.x, gc.y, gc.z ); glVertex3f ( gc.x+gd.x, gc.y+gd.y, gc.z );
	glVertex3f ( gc.x, gc.y, gc.z+gd.z ); glVertex3f ( gc.x, gc.y+gd.y, gc.z+gd.z );
	glVertex3f ( gc.x+gd.x, gc.y, gc.z+gd.z ); glVertex3f ( gc.x+gd.x, gc.y+gd.y, gc.z+gd.z );

	glVertex3f ( gc.x, gc.y, gc.z ); glVertex3f ( gc.x, gc.y, gc.z+gd.x );
	glVertex3f ( gc.x, gc.y+gd.y, gc.z ); glVertex3f ( gc.x, gc.y+gd.y, gc.z+gd.z );
	glVertex3f ( gc.x+gd.x, gc.y, gc.z ); glVertex3f ( gc.x+gd.x, gc.y, gc.z+gd.z );
	glVertex3f ( gc.x+gd.x, gc.y+gd.y, gc.z); glVertex3f ( gc.x+gd.x, gc.y+gd.y, gc.z+gd.z );
	glEnd ();
}

void FluidSystem::DrawGrid ()
{
	Vector3DF gd (1, 1, 1);
	Vector3DF gc;
	gd /= m_GridDelta;		
	
	glBegin ( GL_LINES );	
	for (int z=0; z <= m_GridRes.z; z++ ) {
		for (int y=0; y <= m_GridRes.y; y++ ) {
			gc.Set ( 1, y, z);	gc /= m_GridDelta;	gc += m_GridMin;
			glVertex3f ( m_GridMin.x, gc.y, gc.z );	glVertex3f ( m_GridMax.x, gc.y, gc.z );
		}
	}
	for (int z=0; z <= m_GridRes.z; z++ ) {
		for (int x=0; x <= m_GridRes.x; x++ ) {
			gc.Set ( x, 1, z);	gc /= m_GridDelta;	gc += m_GridMin;
			glVertex3f ( gc.x, m_GridMin.y, gc.z );	glVertex3f ( gc.x, m_GridMax.y, gc.z );
		}
	}
	for (int y=0; y <= m_GridRes.y; y++ ) {
		for (int x=0; x <= m_GridRes.x; x++ ) {
			gc.Set ( x, y, 1);	gc /= m_GridDelta;	gc += m_GridMin;
			glVertex3f ( gc.x, gc.y, m_GridMin.z );	glVertex3f ( gc.x, gc.y, m_GridMax.z );
		}
	}
	glEnd ();
}

void FluidSystem::DrawParticle ( int p, int r1, int r2, Vector3DF clr )
{
	Vector3DF* ppos = mPos + p;
	DWORD* pclr = mClr + p;
	
	glDisable ( GL_DEPTH_TEST );
	
	glPointSize ( r2 );	
	glBegin ( GL_POINTS );
	glColor3f ( clr.x, clr.y, clr.z ); glVertex3f ( ppos->x, ppos->y, ppos->z );
	glEnd ();

	glEnable ( GL_DEPTH_TEST );
}

void FluidSystem::DrawNeighbors ( int p )
{
	if ( p == -1 ) return;

	Vector3DF* ppos = mPos + p;
	Vector3DF jpos;
	CLRVAL jclr;
	int j;

	glBegin ( GL_LINES );
	int cnt = *(mNbrCnt + p);
	int ndx = *(mNbrNdx + p);
	for ( int n=0; n < cnt; n++ ) {
		j = m_NeighborTable[ ndx ];
		jpos = *(mPos + j);
		jclr = *(mClr + j);
		glColor4f ( (RED(jclr)+1.0)*0.5, (GRN(jclr)+1.0)*0.5, (BLUE(jclr)+1.0)*0.5, ALPH(jclr) );
		glVertex3f ( ppos->x, ppos->y, ppos->z );
		
		jpos -= *ppos; jpos *= 0.9;		// get direction of neighbor, 90% dist
		glVertex3f ( ppos->x + jpos.x, ppos->y + jpos.y, ppos->z + jpos.z );
		ndx++;
	}
	glEnd ();
}

void FluidSystem::DrawCircle ( Vector3DF pos, float r, Vector3DF clr, Camera3D& cam )
{
	glPushMatrix ();
	
	glTranslatef ( pos.x, pos.y, pos.z );
	glMultMatrixf ( cam.getInvView().GetDataF() );
	glColor3f ( clr.x, clr.y, clr.z );
	glBegin ( GL_LINE_LOOP );
	float x, y;
	for (float a=0; a < 360; a += 10.0 ) {
		x = cos ( a*DEGtoRAD )*r;
		y = sin ( a*DEGtoRAD )*r;
		glVertex3f ( x, y, 0 );
	}
	glEnd ();

	glPopMatrix ();
}


void FluidSystem::DrawText ()
{
	char msg[100];

	
	Vector3DF* ppos = mPos;
	DWORD* pclr = mClr;
	Vector3DF clr;
	for (int n = 0; n < NumPoints(); n++) {
	
		sprintf ( msg, "%d", n );
		glColor4f ( (RED(*pclr)+1.0)*0.5, (GRN(*pclr)+1.0)*0.5, (BLUE(*pclr)+1.0)*0.5, ALPH(*pclr) );
		//drawText3D ( ppos->x, ppos->y, ppos->z, msg );
		ppos++;
		pclr++;
	}
}


void FluidSystem::Draw ( Camera3D& cam, float rad )
{
	char* dat;
	Vector3DF* ppos;
	float* pdens;
	DWORD* pclr;
		

	glDisable ( GL_LIGHTING );

	switch ( (int) m_Param[PDRAWGRID] ) {	
	case 1: {
		glColor4f ( 0.7, 0.7, 0.7, 0.05 );
		DrawGrid ();
		} break;
	};
	if ( m_Param[PDRAWTEXT] == 1.0 ) {
		DrawText ();
	};

	// Draw Modes
	// DRAW_POINTS		0
	// DRAW_SPRITES		1
	// DRAW_
	
	switch ( (int) m_Param[PDRAWMODE] ) {
	case 0: {
		glPointSize ( 2 );
		glEnable ( GL_POINT_SIZE );		
		glEnable( GL_BLEND ); 
		glBindBufferARB ( GL_ARRAY_BUFFER_ARB, mVBO[0] );
		glBufferDataARB ( GL_ARRAY_BUFFER_ARB, NumPoints()*sizeof(Vector3DF), mPos, GL_DYNAMIC_DRAW_ARB);		
		glVertexPointer ( 3, GL_FLOAT, 0, 0x0 );				
		glBindBufferARB ( GL_ARRAY_BUFFER_ARB, mVBO[1] );
		glBufferDataARB ( GL_ARRAY_BUFFER_ARB, NumPoints()*sizeof(uint), mClr, GL_DYNAMIC_DRAW_ARB);
		glColorPointer ( 4, GL_UNSIGNED_BYTE, 0, 0x0 ); 
		glEnableClientState ( GL_VERTEX_ARRAY );
		glEnableClientState ( GL_COLOR_ARRAY );          
		glNormal3f ( 0, 0.001, 1 );
		glColor3f ( 1, 1, 1 );
		//glLoadMatrixf ( view_mat );
		glDrawArrays ( GL_POINTS, 0, NumPoints() );
		glDisableClientState ( GL_VERTEX_ARRAY );
		glDisableClientState ( GL_COLOR_ARRAY );
		} break;
	
	case 1: {

		glEnable(GL_BLEND); 
	    glEnable(GL_ALPHA_TEST); 
	    glAlphaFunc( GL_GREATER, 0.5 ); 
		//glEnable ( GL_COLOR_MATERIAL );
		//glColorMaterial ( GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE );
				
		// Point sprite size
		
		glEnable(GL_POINT_SPRITE_ARB); 		
		float quadratic[] =  { 1.0f, 0.01f, 0.0001f };
		glEnable (  GL_POINT_DISTANCE_ATTENUATION  );
		glPointParameterfvARB(  GL_POINT_DISTANCE_ATTENUATION, quadratic );
		//float maxSize = 10.0f;
		//glGetFloatv( GL_POINT_SIZE_MAX_ARB, &maxSize );		
		glPointSize ( 32 );		
		glPointParameterfARB( GL_POINT_SIZE_MAX_ARB, 32 );
		glPointParameterfARB( GL_POINT_SIZE_MIN_ARB, 1.0f );

		// Texture and blending mode
		glEnable ( GL_TEXTURE_2D );
		glBindTexture ( GL_TEXTURE_2D, mImg.getTex() );
		glTexEnvi (GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
		glTexEnvf (GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE );
		glBlendFunc ( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA ) ;

		// Point buffers
		glBindBufferARB ( GL_ARRAY_BUFFER_ARB, mVBO[0] );
		glBufferDataARB ( GL_ARRAY_BUFFER_ARB, NumPoints()*sizeof(Vector3DF), mPos, GL_DYNAMIC_DRAW_ARB);		
		glVertexPointer ( 3, GL_FLOAT, 0, 0x0 );				
		glBindBufferARB ( GL_ARRAY_BUFFER_ARB, mVBO[1] );
		glBufferDataARB ( GL_ARRAY_BUFFER_ARB, NumPoints()*sizeof(uint), mClr, GL_DYNAMIC_DRAW_ARB);
		glColorPointer ( 4, GL_UNSIGNED_BYTE, 0, 0x0 ); 
		glEnableClientState ( GL_VERTEX_ARRAY );
		glEnableClientState ( GL_COLOR_ARRAY );
          
		// Render - Point Sprites
		glNormal3f ( 0, 1, 0.001  );
		glColor4f ( 1, 1, 1, 1 );
		glDrawArrays ( GL_POINTS, 0, NumPoints() );

		// Restore state
		glDisableClientState ( GL_VERTEX_ARRAY );
		glDisableClientState ( GL_COLOR_ARRAY );
		glDisable (GL_POINT_SPRITE_ARB); 
		glDisable ( GL_ALPHA_TEST );
		glDisable ( GL_TEXTURE_2D );
		glDepthMask( GL_TRUE );   


		} break;
	case 2: {

		// Notes:
		// # particles, time(Render), time(Total), time(Sim), Render Overhead (%)
		//  250000,  12, 110, 98,  10%   - Point sprites
		//  250000,  36, 146, 110, 24%   - Direct rendering (drawSphere)
		//  250000, 140, 252, 110, 55%   - Batch instancing

		glEnable ( GL_LIGHTING );
		ppos = mPos;
		pclr = mClr;
		pdens = mDensity;
		
		for (int n = 0; n < NumPoints(); n++) {
			glPushMatrix ();
			glTranslatef ( ppos->x, ppos->y, ppos->z );		
			glScalef ( rad, rad, rad );			
			glColor4f ( RED(*pclr), GRN(*pclr), BLUE(*pclr), ALPH(*pclr) );
			//drawSphere ();
			glPopMatrix ();		
			ppos++;
			pclr++;
		}

		// --- HARDWARE INSTANCING
		/* cgGLEnableProfile( vert_profile );		
		// Sphere VBO
		glBindBufferARB ( GL_ARRAY_BUFFER_ARB, mVBO[2] );
		glVertexPointer ( 3, GL_FLOAT, 0, 0x0 );		
		glEnableClientState ( GL_VERTEX_ARRAY );
	
		glColor4f( 1,1,1,1 );

		CGparameter uParam = cgGetNamedParameter( cgVP, "modelViewProj" );
		glLoadMatrixf ( view_mat );
		cgGLSetStateMatrixParameter( uParam, CG_GL_MODELVIEW_PROJECTION_MATRIX, CG_GL_MATRIX_IDENTITY ); 

		uParam = cgGetNamedParameter( cgVP, "transformList" );
		int batches = NumPoints() / 768;
		int noff = 0;		
		for (int n=0; n < batches; n++ ) {
			cgGLSetParameterArray3f ( uParam, 0, 768, (float*) (mPos + noff) ); 
			glDrawArraysInstancedARB ( GL_TRIANGLE_STRIP, 0, mSpherePnts, 768 );
			noff += 768;
		}
		cgGLDisableProfile( vert_profile );
		glDisableClientState ( GL_VERTEX_ARRAY );
		glDisableClientState ( GL_COLOR_ARRAY );  */


		//--- Texture buffer technique
		/*
		uParam = cgGetNamedParameter( cgVP, "transformList");
		cgGLSetTextureParameter ( uParam, mTex[0] );
		cgGLEnableTextureParameter ( uParam );
		uParam = cgGetNamedParameter( cgVP, "primCnt");
		cgGLSetParameter1f ( uParam, NumPoints() );		
		glBindTexture ( GL_TEXTURE_2D, mTex[0] );
		glTexImage2D ( GL_TEXTURE_2D, 0, GL_RGB32F_ARB, 2048, int(NumPoints()/2048)+1, 0, GL_RGB, GL_FLOAT, mPos );
		glBindTexture ( GL_TEXTURE_2D, 0x0 );
		glFinish ();*/		
		} break;
	};

	//-------------------------------------- DEBUGGING
	// draw neighbors of particle i
		/*int i = 320;
		int j, jndx = (mNbrList + i )->first;
		for (int nbr=0; nbr < (mNbrList+i)->num; nbr++ ) {			
			j = *(m_NeighborTable+jndx);
			ppos = (mPos + j );
			glPushMatrix ();
			glTranslatef ( ppos->x, ppos->y, ppos->z );		
			glScalef ( 0.25, 0.25, 0.25 );			
			glColor4f ( 0, 1, 0, 1);		// green
			drawSphere ();
			glPopMatrix ();		
			jndx++;
		}
		// draw particles in grid cells of i
		Vector3DF jpos;
		Grid_FindCells ( i );
		for (int cell=0; cell < 8; cell++) {
			j = m_Grid [ *(mClusterCell+i) + m_GridAdj[cell] ];			
			while ( j != -1 ) {
				if ( i==j ) { j = *(mGridNext+j); continue; }
				jpos = *(mPos + j);
				glPushMatrix ();
				glTranslatef ( jpos.x, jpos.y, jpos.z );		
				glScalef ( 0.22, 0.22, 0.22 );
				glColor4f ( 1, 1, 0, 1);		// yellow
				drawSphere ();
				glPopMatrix ();
				j = *(mGridNext+j);
			}
		}

		// draw grid cells of particle i		
		float poff = m_Param[PSMOOTHRADIUS] / m_Param[PSIMSCALE];
		int gx = (int)( (-poff + ppos->x - m_GridMin.x) * m_GridDelta.x);		// Determine grid cell
		int gy = (int)( (-poff + ppos->y - m_GridMin.y) * m_GridDelta.y);
		int gz = (int)( (-poff + ppos->z - m_GridMin.z) * m_GridDelta.z);
		Vector3DF gd (1, 1, 1);
		Vector3DF gc;
		gd /= m_GridDelta;

		*/

	// Error particles (debugging)
	/*for (int n=0; n < NumPoints(); n++) {
		if ( ALPH(*(mClr+n))==0.9 ) 
			DrawParticle ( n, 12, 14, Vector3DF(1,0,0) );
	}

	// Draw selected particle
	DrawNeighbors ( mSelected );
	DrawParticle ( mSelected, 8, 12, Vector3DF(1,1,1) );
	DrawCircle ( *(mPos+mSelected), m_Param[PSMOOTHRADIUS]/m_Param[PSIMSCALE], Vector3DF(1,1,0), cam );
	Vector3DI gc;
	int gs = getGridCell ( mSelected, gc );	// Grid cell of selected
	
	glDisable ( GL_DEPTH_TEST );	
	glColor3f ( 0.8, 0.8, 0.9 );
	gs = *(mClusterCell + mSelected);		// Cluster cell
	for (int n=0; n < m_GridAdjCnt; n++ ) {		// Cluster group
		gc = getCell ( gs + m_GridAdj[n] );	DrawCell ( gc.x, gc.y, gc.z );
	}
	glColor3f ( 1.0, 1.0, 1.0 );
	DrawCell ( gc.x, gc.y, gc.z );
	glEnable ( GL_DEPTH_TEST );*/
}

std::string FluidSystem::getFilename ( int n )
{
	char name[100];
	sprintf ( name, "particles%04d.dat", n );
	return name;
}

void FluidSystem::StartRecord ()
{
	mFileNum = getLastRecording () + 1;	
	mFileName = getFilename ( mFileNum );
	if ( mFP != 0x0 ) fclose ( mFP );
	char name[100];
	strcpy ( name, mFileName.c_str() );
	mFP = fopen ( name, "wb" );		
	if ( mFP == 0x0 ) {
		app_printf ( "ERROR: Cannot write file %s\n", mFileName.c_str() );
		exit ( -1 );
	}
	mLastPoints = 0;
	mFileSize = 0;
}

int FluidSystem::getLastRecording ()
{
	FILE* fp;
	int num = 0;
	fp = fopen ( getFilename(num).c_str(), "rb" );	
	while ( fp != 0x0 ) {			// skip existing recordings
		fclose ( fp );
		num++;
		fp = fopen ( getFilename(num).c_str(), "rb" );	
	}
	return num-1;
}

void FluidSystem::Record ()
{
	Vector3DF*  ppos =		mPos;
	Vector3DF*  pvel =		mVel;
	float*		pdens =		mDensity;
	DWORD*		pclr =		mClr;
	
	char*		dat = mPackBuf;
	int			channels;
	int			dsize;

	fwrite ( &mNumPoints, sizeof(int), 1, mFP );
	
	// How many channels to write? 
	channels = 4;
	if ( mNumPoints == mLastPoints ) channels = 2;		// save disk space
	fwrite ( &channels, sizeof(int), 1, mFP ) ;
	
	// Write data
	if ( channels == 2 ) {	
		dsize = sizeof(Vector3DF)+sizeof(DWORD);
		for (int n=0; n < mNumPoints; n++ ) {
			*(Vector3DF*) dat = *ppos++;		dat += sizeof(Vector3DF);
			*(DWORD*)	  dat = *pclr++;		dat += sizeof(DWORD);
		}
	} else {
		dsize = sizeof(Vector3DF)+sizeof(Vector3DF)+sizeof(float)+sizeof(DWORD);
		for (int n=0; n < mNumPoints; n++ ) {
			*(Vector3DF*) dat = *ppos++;		dat += sizeof(Vector3DF);
			*(Vector3DF*) dat = *pvel++;		dat += sizeof(Vector3DF);
			*(float*)	  dat = *pdens++;		dat += sizeof(float);
			*(DWORD*)	  dat = *pclr++;		dat += sizeof(DWORD);
		}
	}
		
	fwrite ( mPackBuf, dsize, mNumPoints, mFP );

	mFileSize += float(dsize * mNumPoints) / 1048576.0;

	mLastPoints = mNumPoints;

	fflush ( mFP );
}

void FluidSystem::StartPlayback ( int p )
{
	if ( p < 0 ) return;

	m_Param[PMODE] = RUN_PLAYBACK;
	mFileNum = p;
	mFileName = getFilename ( mFileNum );
	if ( mFP != 0x0 ) { fclose ( mFP ); mFP = 0x0; }
	char name[100];
	strcpy ( name, mFileName.c_str() );
	mFP = fopen ( name, "rb" );
	if ( mFP==0x0 || ferror(mFP) ) {
		app_printf ( "ERROR: Cannot read file %s\n", mFileName.c_str() );
		perror ( "  " );
		exit ( -1 );
	}
	m_Frame = 0;
}

void FluidSystem::RunPlayback ()
{	
	if ( feof (mFP) ) StartPlayback ( mFileNum );

	// Read number of points and channels
	int result = fread ( &mNumPoints, sizeof(int), 1, mFP );
	if ( ferror (mFP) || result != 1 )		{ StartPlayback ( mFileNum ); return; }
	if ( feof(mFP) || mNumPoints <= 0 )		{ StartPlayback ( mFileNum ); return; }

	int channels, dsize;
	fread ( &channels, sizeof(int), 1, mFP );
	
	// Allocate extra memory if needed
	if ( mNumPoints > mMaxPoints ) {
		AllocateParticles ( mNumPoints );
		AllocatePackBuf ();
	}
	
	char*	dat = mPackBuf;		
	Vector3DF*  ppos =		mPos;
	Vector3DF*  pvel =		mVel;
	float*		pdens =		mDensity;
	DWORD*		pclr =		mClr;

	// Read data
	if ( channels == 2 ) {
		dsize = sizeof(Vector3DF)+sizeof(DWORD);
		result = fread ( dat, dsize, mNumPoints, mFP );
		if ( ferror (mFP) || result != mNumPoints ) { StartPlayback ( mFileNum ); return; }
		for (int n=0; n < mNumPoints; n++ ) {
			*ppos++ =  *(Vector3DF*) dat;		dat += sizeof(Vector3DF);
			*pclr++ =  *(DWORD*) dat;			dat += sizeof(DWORD);
		}
	} else {
		dsize = sizeof(Vector3DF)+sizeof(Vector3DF)+sizeof(float)+sizeof(DWORD);
		result = fread ( dat, dsize, mNumPoints, mFP );
		if ( ferror (mFP) || result != mNumPoints ) { StartPlayback ( mFileNum ); return; }
		for (int n=0; n < mNumPoints; n++ ) {
			*ppos++ =  *(Vector3DF*) dat;		dat += sizeof(Vector3DF);
			*pvel++ =  *(Vector3DF*) dat;		dat += sizeof(Vector3DF);
			*pdens++ = *(float*) dat;			dat += sizeof(float);
			*pclr++ =  *(DWORD*) dat;			dat += sizeof(DWORD);
		}
	}
}



std::string FluidSystem::getModeStr ()
{
	char buf[100];

	switch ( (int) m_Param[PMODE] ) {
	case RUN_SEARCH:		sprintf ( buf, "SEARCH ONLY (CPU)" );		break;
	case RUN_VALIDATE:		sprintf ( buf, "VALIDATE GPU to CPU");		break;
	case RUN_CPU_SLOW:		sprintf ( buf, "SIMULATE CPU Slow");		break;
	case RUN_CPU_GRID:		sprintf ( buf, "SIMULATE CPU Grid");		break;
	case RUN_CUDA_RADIX:	sprintf ( buf, "SIMULATE CUDA Radix Sort");	break;
	case RUN_CUDA_INDEX:	sprintf ( buf, "SIMULATE CUDA Index Sort" ); break;
	case RUN_CUDA_FULL:	sprintf ( buf, "SIMULATE CUDA Full Sort" );	break;
	case RUN_CUDA_CLUSTER:	sprintf ( buf, "SIMULATE CUDA Clustering" );	break;
	case RUN_PLAYBACK:		sprintf ( buf, "PLAYBACK (%s)", mFileName.c_str() ); break;
	};
	//sprintf ( buf, "RECORDING (%s, %.4f MB)", mFileName.c_str(), mFileSize ); break;
	return buf;
};


void FluidSystem::getModeClr ()
{
	glColor4f ( 1, 1, 0, 1 ); 
	/*break;
	switch ( mMode ) {
	case RUN_PLAYBACK:		glColor4f ( 0, 1, 0, 1 ); break;
	case RUN_RECORD:		glColor4f ( 1, 0, 0, 1 ); break;
	case RUN_SIM:			glColor4f ( 1, 1, 0, 1 ); break;
	}*/
}

int FluidSystem::SelectParticle ( int x, int y, int wx, int wy, Camera3D& cam )
{
	Vector4DF pnt;
	Vector3DF* ppos = mPos;
	
	for (int n = 0; n < NumPoints(); n++ ) {
		pnt = cam.project ( *ppos );
		pnt.x = (pnt.x+1.0)*0.5 * wx;
		pnt.y = (pnt.y+1.0)*0.5 * wy;

		if ( x > pnt.x-8 && x < pnt.x+8 && y > pnt.y-8 && y < pnt.y+8 ) {
			mSelected = n;
			return n;
		}
		ppos++;
	}
	mSelected = -1;
	return -1;
}


void FluidSystem::DrawParticleInfo ( int p )
{
	char disp[256];

	start2D ();

	glColor4f ( 1.0, 1.0, 1.0, 1.0 );
	sprintf ( disp, "Particle: %d", p );		drawText ( 10, 20, disp, 1, 1, 1, 1 ); 

	Vector3DI gc;
	int gs = getGridCell ( p, gc );
	sprintf ( disp, "Grid Cell:    <%d, %d, %d> id: %d", gc.x, gc.y, gc.z, gs );		drawText ( 10, 40, disp, 1,1,1,1 ); 

	int cc = *(mClusterCell + p);
	gc = getCell ( cc );
	sprintf ( disp, "Cluster Cell: <%d, %d, %d> id: %d", gc.x, gc.y, gc.z, cc );		drawText ( 10, 50, disp, 1,1,1,1 ); 

	sprintf ( disp, "Neighbors:    " );
	int cnt = *(mNbrCnt + p);
	int ndx = *(mNbrNdx + p);
	for ( int n=0; n < cnt; n++ ) {
		sprintf ( disp, "%s%d, ", disp, m_NeighborTable[ ndx ] );
		ndx++;
	}
	drawText ( 10, 70, disp );

	if ( cc != -1 ) {
		sprintf ( disp, "Cluster Group: ");		drawText ( 10, 90, disp);
		int cadj;
		int stotal = 0;
		for (int n=0; n < m_GridAdjCnt; n++ ) {		// Cluster group
			cadj = cc+m_GridAdj[n];
			gc = getCell ( cadj );
			sprintf ( disp, "<%d, %d, %d> id: %d, cnt: %d ", gc.x, gc.y, gc.z, cc+m_GridAdj[n], m_GridCnt[ cadj ] );	drawText ( 20, 100+n*10, disp );
			stotal += m_GridCnt[cadj];
		}

		sprintf ( disp, "Search Overhead: %f (%d of %d), %.2f%% occupancy", float(stotal)/ cnt, cnt, stotal, float(cnt)*100.0/stotal );
		drawText ( 10, 380, disp );
	}	

	end2D ();
}



void FluidSystem::SetupKernels ()
{
	m_Param [ PDIST ] = pow ( m_Param[PMASS] / m_Param[PRESTDENSITY], 1/3.0 );
	m_R2 = m_Param [PSMOOTHRADIUS] * m_Param[PSMOOTHRADIUS];
	m_Poly6Kern = 315.0f / (64.0f * 3.141592 * pow( m_Param[PSMOOTHRADIUS], 9) );	// Wpoly6 kernel (denominator part) - 2003 Muller, p.4
	m_SpikyKern = -45.0f / (3.141592 * pow( m_Param[PSMOOTHRADIUS], 6) );			// Laplacian of viscocity (denominator): PI h^6
	m_LapKern = 45.0f / (3.141592 * pow( m_Param[PSMOOTHRADIUS], 6) );
}

void FluidSystem::SetupDefaultParams ()
{
	//  Range = +/- 10.0 * 0.006 (r) =	   0.12			m (= 120 mm = 4.7 inch)
	//  Container Volume (Vc) =			   0.001728		m^3
	//  Rest Density (D) =				1000.0			kg / m^3
	//  Particle Mass (Pm) =			   0.00020543	kg						(mass = vol * density)
	//  Number of Particles (N) =		4000.0
	//  Water Mass (M) =				   0.821		kg (= 821 grams)
	//  Water Volume (V) =				   0.000821     m^3 (= 3.4 cups, .21 gals)
	//  Smoothing Radius (R) =             0.02			m (= 20 mm = ~3/4 inch)
	//  Particle Radius (Pr) =			   0.00366		m (= 4 mm  = ~1/8 inch)
	//  Particle Volume (Pv) =			   2.054e-7		m^3	(= .268 milliliters)
	//  Rest Distance (Pd) =			   0.0059		m
	//
	//  Given: D, Pm, N
	//    Pv = Pm / D			0.00020543 kg / 1000 kg/m^3 = 2.054e-7 m^3	
	//    Pv = 4/3*pi*Pr^3    cuberoot( 2.054e-7 m^3 * 3/(4pi) ) = 0.00366 m
	//     M = Pm * N			0.00020543 kg * 4000.0 = 0.821 kg		
	//     V =  M / D              0.821 kg / 1000 kg/m^3 = 0.000821 m^3
	//     V = Pv * N			 2.054e-7 m^3 * 4000 = 0.000821 m^3
	//    Pd = cuberoot(Pm/D)    cuberoot(0.00020543/1000) = 0.0059 m 
	//
	// Ideal grid cell size (gs) = 2 * smoothing radius = 0.02*2 = 0.04
	// Ideal domain size = k*gs/d = k*0.02*2/0.005 = k*8 = {8, 16, 24, 32, 40, 48, ..}
	//    (k = number of cells, gs = cell size, d = simulation scale)

	// "The viscosity coefficient is the dynamic viscosity, visc > 0 (units Pa.s), 
	// and to include a reasonable damping contribution, it should be chosen 
	// to be approximately a factor larger than any physical correct viscosity 
	// coefficient that can be looked up in the literature. However, care should 
	// be taken not to exaggerate the viscosity coefficient for fluid materials.
	// If the contribution of the viscosity force density is too large, the net effect 
	// of the viscosity term will introduce energy into the system, rather than 
	// draining the system from energy as intended."
	//    Actual visocity of water = 0.001 Pa.s    // viscosity of water at 20 deg C.

	m_Time = 0;							// Start at T=0
	m_DT = 0.003;	

	m_Param [ PSIMSCALE ] =		0.005;			// unit size
	m_Param [ PVISC ] =			0.35;			// pascal-second (Pa.s) = 1 kg m^-1 s^-1  (see wikipedia page on viscosity)
	m_Param [ PRESTDENSITY ] =	600.0;			// kg / m^3
	m_Param [ PSPACING ]	=	0.0;			// spacing will be computed automatically from density in most examples (set to 0 for autocompute)
	m_Param [ PMASS ] =			0.00020543;		// kg
	m_Param [ PRADIUS ] =		0.02;			// m
	m_Param [ PDIST ] =			0.0059;			// m
	m_Param [ PSMOOTHRADIUS ] =	0.01;			// m 
	m_Param [ PINTSTIFF ] =		1.5;
	m_Param [ PEXTSTIFF ] =		50000.0;
	m_Param [ PEXTDAMP ] =		100.0;
	m_Param [ PACCEL_LIMIT ] =	150.0;			// m / s^2
	m_Param [ PVEL_LIMIT ] =	3.0;			// m / s
	m_Param [ PMAX_FRAC ] = 1.0;
	m_Param [ PPOINT_GRAV_AMT ] = 0.0;

	m_Param [ PGROUND_SLOPE ] = 0.0;
	m_Param [ PFORCE_MIN ] = 0.0;
	m_Param [ PFORCE_MAX ] = 0.0;	
	m_Param [ PFORCE_FREQ ] = 8.0;	
	m_Toggle [ PWRAP_X ] = false;
	m_Toggle [ PWALL_BARRIER ] = false;
	m_Toggle [ PLEVY_BARRIER ] = false;
	m_Toggle [ PDRAIN_BARRIER ] = false;

	m_Param [ PSTAT_NBRMAX ] = 0 ;
	m_Param [ PSTAT_SRCHMAX ] = 0 ;
	
	m_Vec [ PPOINT_GRAV_POS ].Set ( 0, 50, 0 );
	m_Vec [ PPLANE_GRAV_DIR ].Set ( 0, -9.8, 0 );
	m_Vec [ PEMIT_POS ].Set ( 0, 0, 0 );
	m_Vec [ PEMIT_RATE ].Set ( 0, 0, 0 );
	m_Vec [ PEMIT_ANG ].Set ( 0, 90, 1.0 );
	m_Vec [ PEMIT_DANG ].Set ( 0, 0, 0 );

	// Default sim config
	m_Toggle [ PRUN ] = true;				// Run integrator
	m_Param [PGRIDSIZE] = m_Param[PSMOOTHRADIUS] * 2;
	m_Param [PDRAWMODE] = 1;				// Sprite drawing
	m_Param [PDRAWGRID] = 0;				// No grid 
	m_Param [PDRAWTEXT] = 0;				// No text

	// Load settings from XML (overwrite the above defaults)
	ParseXML ( "Fluid", 0, false );
}

int FluidSystem::ParseXML ( std::string name, int id, bool bStart )
{
	xml.setBase ( name, id );

	xml.assignValueD ( &m_DT, "DT" );
	xml.assignValueStr ( mSceneName, "Name" );
	if (bStart)	xml.assignValueD ( &m_Param[PNUM],			"Num" );
	xml.assignValueD ( &m_Param[PGRID_DENSITY],	"GridDensity" );
	xml.assignValueD ( &m_Param[PSIMSCALE],		"SimScale" );
	xml.assignValueD ( &m_Param[PVISC],			"Viscosity" );
	xml.assignValueD ( &m_Param[PRESTDENSITY],	"RestDensity" );
	xml.assignValueD ( &m_Param[PSPACING],		"Spacing" );
	xml.assignValueD ( &m_Param[PMASS],			"Mass" );
	xml.assignValueD ( &m_Param[PRADIUS],		"Radius" );
	xml.assignValueD ( &m_Param[PDIST],			"SearchDist" );
	xml.assignValueD ( &m_Param[PINTSTIFF],		"IntStiff" );
	xml.assignValueD ( &m_Param[PEXTSTIFF],		"BoundStiff" );
	xml.assignValueD ( &m_Param[PEXTDAMP],		"BoundDamp" );
	xml.assignValueD ( &m_Param[PACCEL_LIMIT],	"AccelLimit" );
	xml.assignValueD ( &m_Param[PVEL_LIMIT],	"VelLimit" );
	xml.assignValueD ( &m_Param[PPOINT_GRAV_AMT],	"PointGravAmt" );	
	xml.assignValueD ( &m_Param[PGROUND_SLOPE],	"GroundSlope" );
	xml.assignValueD ( &m_Param[PFORCE_MIN],	"WaveForceMin" );
	xml.assignValueD ( &m_Param[PFORCE_MAX],	"WaveForceMax" );
	xml.assignValueD ( &m_Param[PFORCE_FREQ],	"WaveForceFreq" );
	xml.assignValueD ( &m_Param[PDRAWMODE],		"DrawMode" );
	xml.assignValueD ( &m_Param[PDRAWGRID],		"DrawGrid" );
	xml.assignValueD ( &m_Param[PDRAWTEXT],		"DrawText" );
	
	xml.assignValueV3 ( &m_Vec[PVOLMIN],		"VolMin" );
	xml.assignValueV3 ( &m_Vec[PVOLMAX],		"VolMax" );
	xml.assignValueV3 ( &m_Vec[PINITMIN],		"InitMin" );
	xml.assignValueV3 ( &m_Vec[PINITMAX],		"InitMax" );
	xml.assignValueV3 ( &m_Vec[PPOINT_GRAV_POS],	"PointGravPos" );
	xml.assignValueV3 ( &m_Vec[PPLANE_GRAV_DIR],	"PlaneGravDir" );
	
	return m_Param[PNUM];
}

void FluidSystem::SetupExampleParams ( bool bStart )
{
	Vector3DF pos;
	Vector3DF min, max;
	
	switch ( (int) m_Param[PEXAMPLE] ) {

	case 0:	{	// Regression test. N x N x N static grid

		int k = ceil ( pow ( (float) m_Param[PNUM], (float) 1.0/3.0f ) );
		m_Vec [ PVOLMIN ].Set ( 0, 0, 0 );
		m_Vec [ PVOLMAX ].Set ( 2.0+(k/2), 2.0+(k/2), 2.0+(k/2) );
		m_Vec [ PINITMIN ].Set ( 1.0, 1.0, 1.0 );
		m_Vec [ PINITMAX ].Set ( 1.0+(k/2), 1.0+(k/2), 1.0+(k/2) );
		
		m_Param [ PPOINT_GRAV_AMT ] = 0.0;		// No gravity
		m_Vec [ PPLANE_GRAV_DIR ].Set ( 0.0, 0.0, 0.0 );			
		m_Param [ PSPACING ] = 0.5;				// Fixed spacing		Dx = x-axis density
		m_Param [ PSMOOTHRADIUS ] =	m_Param [PSPACING];		// Search radius
		m_Toggle [ PRUN ] = false;				// Do NOT run sim. Neighbors only.				
		m_Param [PDRAWMODE] = 1;				// Point drawing
		m_Param [PDRAWGRID] = 1;				// Grid drawing
		m_Param [PDRAWTEXT] = 1;				// Text drawing
		m_Param [PSIMSCALE ] = 1.0;
	
		} break;
	case 1:		// Wave pool						
		m_Vec [ PVOLMIN ].Set ( -100, 0, -100 );
		m_Vec [ PVOLMAX ].Set (  100, 100, 100 );
		m_Vec [ PINITMIN ].Set ( -50, 20, -90 );
		m_Vec [ PINITMAX ].Set (  90, 90,  90 );
		m_Param [ PFORCE_MIN ] = 10.0;	
		m_Param [ PGROUND_SLOPE ] = 0.04;
		break;
	case 2:		// Large coast						
		m_Vec [ PVOLMIN ].Set ( -200, 0, -40 );
		m_Vec [ PVOLMAX ].Set (  200, 200, 40 );
		m_Vec [ PINITMIN ].Set ( -120, 40, -30 );
		m_Vec [ PINITMAX ].Set (  190, 190,  30 );
		m_Param [ PFORCE_MIN ] = 20.0;	
		m_Param [ PGROUND_SLOPE ] = 0.10;
		break;
	case 3:		// Small dam break
		m_Vec [ PVOLMIN ].Set ( -40, 0, -40  );
		m_Vec [ PVOLMAX ].Set ( 40, 60, 40 );
		m_Vec [ PINITMIN ].Set ( 0, 8, -35 );
		m_Vec [ PINITMAX ].Set ( 35, 55, 35 );		
		m_Param [ PFORCE_MIN ] = 0.0;
		m_Param [ PFORCE_MAX ] = 0.0;
		m_Vec [ PPLANE_GRAV_DIR ].Set ( 0.0f, -9.8f, 0.0f );
		break;
	case 4:		// Dual-Wave pool
		m_Vec [ PVOLMIN ].Set ( -100, 0, -15 );
		m_Vec [ PVOLMAX ].Set ( 100, 100, 15 );
		m_Vec [ PINITMIN ].Set ( -80, 8, -10 );
		m_Vec [ PINITMAX ].Set ( 80, 90, 10 );
		m_Param [ PFORCE_MIN ] = 20.0;
		m_Param [ PFORCE_MAX ] = 20.0;
		m_Vec [ PPLANE_GRAV_DIR ].Set ( 0.0f, -9.8f, 0.0f );	
		break;
	case 5:		// Microgravity
		m_Vec [ PVOLMIN ].Set ( -80, 0, -80 );
		m_Vec [ PVOLMAX ].Set ( 80, 100, 80 );
		m_Vec [ PINITMIN ].Set ( -60, 40, -60 );
		m_Vec [ PINITMAX ].Set ( 60, 80, 60 );		
		m_Vec [ PPLANE_GRAV_DIR ].Set ( 0, -1, 0 );	
		m_Param [ PGROUND_SLOPE ] = 0.1;
		break;
	}
	
	// Load scene from XML file
	int cnt = ParseXML ( "Scene", (int) m_Param[PEXAMPLE], bStart );
}

void FluidSystem::SetupSpacing ()
{
	m_Param [ PSIMSIZE ] = m_Param [ PSIMSCALE ] * (m_Vec[PVOLMAX].z - m_Vec[PVOLMIN].z);	
	
	if ( m_Param[PSPACING] == 0 ) {
		// Determine spacing from density
		m_Param [PDIST] = pow ( m_Param[PMASS] / m_Param[PRESTDENSITY], 1/3.0 );	
		m_Param [PSPACING] = m_Param [ PDIST ]*0.87 / m_Param[ PSIMSCALE ];			
	} else {
		// Determine density from spacing
		m_Param [PDIST] = m_Param[PSPACING] * m_Param[PSIMSCALE] / 0.87;
		m_Param [PRESTDENSITY] = m_Param[PMASS] / pow ( m_Param[PDIST], 3.0 );
	}
	app_printf ( "Add Particles. Density: %f, Spacing: %f, PDist: %f\n", m_Param[PRESTDENSITY], m_Param [ PSPACING ], m_Param[ PDIST ] );

	// Particle Boundaries
	m_Vec[PBOUNDMIN] = m_Vec[PVOLMIN];		m_Vec[PBOUNDMIN] += 2.0*(m_Param[PGRIDSIZE] / m_Param[PSIMSCALE]);
	m_Vec[PBOUNDMAX] = m_Vec[PVOLMAX];		m_Vec[PBOUNDMAX] -= 2.0*(m_Param[PGRIDSIZE] / m_Param[PSIMSCALE]);
}


void FluidSystem::TestPrefixSum ( int num )
{
	app_printf ( "------------------\n");
	app_printf ( "TESTING PREFIX SUM\n");
	app_printf ( "Num: %d\n", num );

	srand ( 2564 );		// deterministic test
	
	// Allocate input and output lists
	int* listIn = (int*) malloc( num * sizeof(int) );
	int* listOutCPU = (int*) malloc( num * sizeof(int) );
	int* listOutGPU = (int*) malloc( num * sizeof(int) );

	// Build list of pseudo-random numbers
	for (int n=0; n < num; n++) 
		listIn[n] = int ((rand()*4.0f) / RAND_MAX);
	app_printf ( "Input: "); for (int n=num-10; n < num; n++)	printf ( "%d ", listIn[n] ); printf (" (last 10 values)\n");		// print first 10

	// Prefix Sum on CPU
	int sum = 0;
	Time start, cpu_stop, gpu_stop;
	start.SetSystemTime ();
	for (int n=0; n < num; n++) {		
		listOutCPU[n] = sum;
		sum += listIn[n];
	}
	cpu_stop.SetSystemTime (); cpu_stop = cpu_stop - start;
	app_printf ( "CPU:   "); for (int n=num-10; n < num; n++)	printf ( "%d ", listOutCPU[n] ); printf (" (last 10 values)\n");		// print first 10
	
	// Prefix Sum on GPU	
	prefixSumToGPU ( (char*) listIn, num, sizeof(int) );
	start.SetSystemTime ();
	prefixSumInt ( num );
	gpu_stop.SetSystemTime (); gpu_stop = gpu_stop - start;
	prefixSumFromGPU ( (char*) listOutGPU, num, sizeof(int) );	
	
	app_printf ( "GPU:   "); for (int n=num-10; n < num; n++)	printf ( "%d ", listOutGPU[n] ); printf (" (last 10 values)\n");		// print first 10

	app_printf ( "Time CPU: %s\n", cpu_stop.GetReadableTime().c_str() );
	app_printf ( "Time GPU: %s\n", gpu_stop.GetReadableTime().c_str() );
	
	// Validate results
	int ok = 0;
	for (int n=0; n < num; n++) {
		if ( listOutCPU[n] == listOutGPU[n] ) ok++;
	}
	app_printf ( "Validate: %d OK. (Bad: %d)\n", ok, num-ok );
	app_printf ( "Press any key to continue..\n");
	_getch();
}


void FluidSystem::CaptureVideo (int width, int height)
{
    nvImg img;
	img.Create ( width, height, IMG_RGB );			// allocates pixel memory    

    FILE *fScreenshot;
	char fileName[64];
    
    sprintf( fileName, "screen_%04d.bmp", m_Frame );
    
    fScreenshot = fopen( fileName, "wb");
												// record frame buffer directly to image pixels
    //glReadPixels( 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, mImg.getData() );	
  
	//mImg.SavePng ( fScreenshot );				// write bmp format

	fflush ( fScreenshot );						// close file
	fclose ( fScreenshot );

    //convert to BGR format    
    /*unsigned char temp;
    int i = 0;
    while (i < nSize) {
        temp = pixels[i];       //grab blue
        pixels[i] = pixels[i+2];//assign red to blue
        pixels[i+2] = temp;     //assign blue to red
        i += 3;     //skip to next blue byte
    }*/
	// TGA format
    /*unsigned char TGAheader[12]={0,0,2,0,0,0,0,0,0,0,0,0};
    unsigned char header[6] = {m_WindowWidth%256,m_WindowWidth/256,
    m_WindowHeight%256,m_WindowHeight/256,24,0};    
    fwrite(TGAheader, sizeof(unsigned char), 12, fScreenshot);
    fwrite(header, sizeof(unsigned char), 6, fScreenshot);
    fwrite(pixels, sizeof(GLubyte), nSize, fScreenshot);
    fclose(fScreenshot);*/
    
    return;
}


