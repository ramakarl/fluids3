//-----------------------------------------------------------------------------
// FLUIDS v5.0 - SPH Fluid Simulator for CPU and GPU
// Copyright (C) 2021. Rama Hoetzlein, http://fluids3.com
//-----------------------------------------------------------------------------

#include <GL/glew.h>
#include <cudaGL.h>	
#include <cuda.h>	
#include <assert.h>
#include <stdio.h>
#include "timex.h"
#include "main.h"
#include "nv_gui.h"

#include "particles.h"


#define EPSILON			0.00001f			// for collision detection

#define SCAN_BLOCKSIZE		512				// must match value in fluid_system_cuda.cu


int iDivUp (int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}
void computeNumBlocks (int numPnts, int minThreads, int &numBlocks, int &numThreads)
{
    numThreads = min( minThreads, numPnts );
    numBlocks = (numThreads==0) ? 1 : iDivUp ( numPnts, numThreads );
}


Particles::Particles ()
{
	m_bGPU = true;		// use GPU pathway
	
	mNumPoints = 0;
	mMaxPoints = 0;	
	m_Frame = 0;
	m_Module = 0;
	for (int n=0; n < FUNC_MAX; n++ ) m_Func[n] = (CUfunction) -1;
}

Particles::~Particles()
{
	Clear ();

	if (m_Module != 0) {
		cuCheck(cuModuleUnload(m_Module), "~FluidSystem()", "cuModuleUnload", "m_Module", m_bDebug);
	}
}
void Particles::Clear ()
{
  m_Points.SetNum (0);
  m_PointsTemp.SetNum (0);
}

void Particles::LoadKernel ( int fid, std::string func )
{
	char cfn[512];		strcpy ( cfn, func.c_str() );

	if ( m_Func[fid] == (CUfunction) -1 )
		cuCheck ( cuModuleGetFunction ( &m_Func[fid], m_Module, cfn ), "LoadKernel", "cuModuleGetFunction", cfn, m_bDebug );	
}

// Initialize
// - Caller must have already created a CUDA context 
void Particles::Initialize ()
{
	cuCheck ( cuModuleLoad ( &m_Module, "particles.ptx" ), "LoadKernel", "cuModuleLoad", "particles.ptx", m_bDebug);

	LoadKernel ( FUNC_INSERT,			"insertParticles" );
	LoadKernel ( FUNC_COUNTING_SORT,	"countingSortFull" );
	LoadKernel ( FUNC_COMPUTE_PRESS,	"computePressure" );
	LoadKernel ( FUNC_COMPUTE_FORCE,	"computeForce" );
	LoadKernel ( FUNC_ADVANCE,			"advanceParticles" );
	LoadKernel ( FUNC_FPREFIXSUM,		"prefixSum" );
	LoadKernel ( FUNC_FPREFIXFIXUP,		"prefixFixup" );
	
	// Assign DataX buffers to GPU
	m_Points.AssignToGPU ( "FPnts", m_Module );
	m_PointsTemp.AssignToGPU ( "FPntTmp", m_Module );
	m_Accel.AssignToGPU ( "FAccel", m_Module );
		
	m_Params.example =		2;
	m_Params.grid_density = 2.0;
	m_Params.pnum	=		65536;

	// Access the sim parameters
	// (note: should be made DataX later)
	size_t len;
	cuCheck ( cuModuleGetGlobal ( &m_cuParams, &len, m_Module, "FParams" ), "Initialize", "cuModuleGetGlobal", "cuParams", true );
}

void Particles::Restart ( int pmax )
{
	mNumPoints = 0;			// reset count
	mMaxPoints = pmax;

  // Clear points
  Clear();

	// Setup example
	SetupDefaultParams ();	
	SetupExampleParams ();	
	
	// Reallocate particles
	ReallocateParticles( mMaxPoints );

	// Build acceleration grid
	RebuildAccelGrid();
	
	// Add the particles (after reallocate)
	AddPointsInVolume ( m_Params.init_min, m_Params.init_max );		// increases mNumPoints

	UpdateParams ();		// Update parameters

	CommitAll ();			// Initial transfer
}


void Particles::SetupDefaultParams ()
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

	m_Params.time =					0;
	m_Params.dt =						0.004f;	
	m_Params.sim_scale =		0.006;			// unit size
	m_Params.pvisc =				0.10f;			// pascal-second (Pa.s) = 1 kg m^-1 s^-1  (see wikipedia page on viscosity)
	m_Params.prest_dens =		200.0f;			// kg / m^3	
	m_Params.pspacing =			0.0;			
	m_Params.pmass =				0.00020543f;	// kg
	m_Params.pradius =			0.015f;			// m
	m_Params.pdist =				0.0059f;			// m
	m_Params.psmoothradius =	0.015f;			// m 
	m_Params.pintstiff =		1.5f;	
	m_Params.bound_stiff =	50000.0f;		// boundary stiffness	
	m_Params.bound_damp =		150.0f;
	m_Params.AL =						150.0f;			// accel limit, m / s^2
	m_Params.VL =						50.0f;			// vel limit, m / s	

	m_Params.bound_slope =		0.0f;			// ground slope
	m_Params.bound_friction =	0.0f;			// ground friction	
	m_Params.bound_wall_force =	0.0f;	
	m_Params.grav_amt =			1.0f;	

	/*
	m_Params.sim_scale =		0.007;			// unit size
	m_Params.pvisc =			0.10f;			// pascal-second (Pa.s) = 1 kg m^-1 s^-1  (see wikipedia page on viscosity)
	m_Params.prest_dens =		250.0f;			// kg / m^3	
	m_Params.pspacing =			0.0;			
	m_Params.pmass =			0.00020543f;	// kg
	m_Params.pradius =			0.02f;			// m
	m_Params.pdist =			0.0059f;			// m
	m_Params.psmoothradius =	0.02f;			// m 
	m_Params.pintstiff =		2.0f;	
	m_Params.bound_stiff =		20000.0f;		// boundary stiffness	
	m_Params.bound_damp =		400.0f;
	m_Params.bound_slope =		0.0f;			// ground slope
	m_Params.bound_friction =	1.0f;			// ground friction	
	m_Params.bound_wall_force =	0.0f;	
	m_Params.AL =				150.0f;			// accel limit, m / s^2
	m_Params.VL =				50.0f;			// vel limit, m / s	
	m_Params.grav_amt =			1.0f;	*/
	
	m_Params.grav_pos.Set ( 0, 0, 0 );
	m_Params.grav_dir.Set ( 0, -4.8f, 0 );

	m_Params.emit_pos.Set ( 0, 0, 0 );
	m_Params.emit_rate.Set ( 0, 0, 0 );
	m_Params.emit_ang.Set ( 0, 90, 1.0f );
	m_Params.emit_dang.Set ( 0, 0, 0 );

	// Default sim config	
	m_Params.gridSize = m_Params.psmoothradius * 2;
}

void Particles::SetupExampleParams ()
{
	Vector3DF pos;
	Vector3DF min, max;
	
	switch ( (int) m_Params.example ) {

	case 0:	{	// Regression test. N x N x N static grid

		int k = (int) ceil ( pow ( (float) m_Params.pnum, (float) 1.0f/3.0f ) );
		m_Params.bound_min.Set ( 0, 0, 0 );
		m_Params.bound_max.Set ( 2.0f+(k/2), 2.0f+(k/2), 2.0f+(k/2) );
		m_Params.init_min.Set ( 1.0f, 1.0f, 1.0f );
		m_Params.init_max.Set ( 1.0f+(k/2), 1.0f+(k/2), 1.0f+(k/2) );
		
		m_Params.grav_amt = 0.0;		
		m_Params.grav_dir.Set ( 0.0, 0.0, 0.0 );			
		m_Params.pspacing = 0.5;				// Fixed spacing		Dx = x-axis density
		m_Params.psmoothradius = m_Params.pspacing;		// Search radius				
	
		} break;
	case 1:		// Tower
		m_Params.bound_min.Set (   0,   0,   0 );
		m_Params.bound_max.Set (  256, 128, 256 );
		m_Params.init_min.Set (  5,   5,  5 );
		m_Params.init_max.Set ( 256*0.3f, 128*0.9f, 256*0.3f );		
		break;
	case 2:		// Wave pool
		/*m_Params.bound_min.Set (   0,   0,   0 );		// sample matching Fluid 3.1
		m_Params.bound_max.Set ( 200, 100, 200 );
		m_Params.init_min.Set (  50,  20,  50 );
		m_Params.init_max.Set ( 190,  90, 190 );
		m_Params.bound_wall_force = 40.0f;			
		m_Params.bound_wall_freq = 8.0f;
		m_Params.bound_slope = 0.04f; */

		m_Params.bound_min.Set (   0,   0,   0 );		// large beach front
		m_Params.bound_max.Set ( 500, 200, 500 );
		m_Params.init_min.Set ( 120,  60,   0 );
		m_Params.init_max.Set ( 500,  195, 500 );
		m_Params.bound_wall_force = 100.0f;			
		m_Params.bound_wall_freq = 2.0f;
		m_Params.bound_slope = 0.12f; 
		break;
	case 3:		// Small dam break
		m_Params.bound_min.Set ( -40, 0, -40  );
		m_Params.bound_max.Set ( 40, 60, 40 );
		m_Params.init_min.Set ( 0, 8, -35 );
		m_Params.init_max.Set ( 35, 55, 35 );		
		m_Params.bound_wall_force = 0.0f;
		m_Params.bound_wall_freq = 0.0f;		
		break;
	}
}


// Allocate particle memory
void Particles::ReallocateParticles ( int maxpnt )
{
	mMaxPoints = maxpnt;
	m_Params.pnum = maxpnt;	

	m_Points.DeleteAllBuffers ();
	m_Points.AddBuffer ( FPOS, "pos",		sizeof(Vector3DF),	maxpnt, DT_CPU | DT_CUMEM | DT_GLVBO );		// cuda-opengl interop
	m_Points.AddBuffer ( FCLR, "clr",		sizeof(uint),		maxpnt, DT_CPU | DT_CUMEM | DT_GLVBO );		// cuda-opengl interop
	m_Points.AddBuffer ( FVEL, "vel",		sizeof(Vector3DF),	maxpnt, DT_CPU | DT_CUMEM | DT_GLVBO );		// cuda-opengl interop
	m_Points.AddBuffer ( FVEVAL, "veval",	sizeof(Vector3DF),	maxpnt, DT_CPU | DT_CUMEM );
	m_Points.AddBuffer ( FPRESS, "press",	sizeof(float),		maxpnt, DT_CPU | DT_CUMEM );
	m_Points.AddBuffer ( FFORCE, "force",	sizeof(Vector3DF),	maxpnt, DT_CPU | DT_CUMEM );
	m_Points.AddBuffer ( FGCELL, "gcell",	sizeof(int),		maxpnt, DT_CPU | DT_CUMEM );
	m_Points.AddBuffer ( FGNDX,  "gndx",	sizeof(int),		maxpnt, DT_CPU | DT_CUMEM );

	m_Points.SetBufferUsage ( FPOS, DT_FLOAT3 );			// primarily for debugging
	m_Points.SetBufferUsage ( FCLR, DT_UINT );
	m_Points.SetBufferUsage ( FVEL, DT_FLOAT3 );
	m_Points.SetBufferUsage ( FVEVAL, DT_FLOAT3 );
	m_Points.SetBufferUsage ( FPRESS, DT_FLOAT );
	m_Points.SetBufferUsage ( FFORCE, DT_FLOAT3 );
	m_Points.SetBufferUsage ( FGCELL, DT_INT );
	m_Points.SetBufferUsage ( FGNDX, DT_INT );

	m_PointsTemp.MatchAllBuffers ( &m_Points, DT_CUMEM );

	// Compute particle thread blocks
	int threadsPerBlock = 512;	
	computeNumBlocks ( m_Params.pnum, threadsPerBlock, m_Params.numBlocks, m_Params.numThreads);				// particles    
    m_Params.szPnts = (m_Params.numBlocks  * m_Params.numThreads);     
	dbgprintf ( "  Particles: %d, t:%dx%d=%d, Size:%d\n", m_Params.pnum, m_Params.numBlocks, m_Params.numThreads, m_Params.numBlocks*m_Params.numThreads, m_Params.szPnts);

	UpdateGPUAccess();	
}


int Particles::AddParticle ()
{
	if ( mNumPoints >= mMaxPoints ) return -1;

	int n = mNumPoints;
	m_Points.bufF3(FPOS,n)->Set ( 0,0,0 );
	m_Points.bufF3(FVEL,n)->Set ( 0,0,0 );
	m_Points.bufF3(FVEVAL,n)->Set ( 0,0,0 );
	m_Points.bufF3(FFORCE,n)->Set ( 0,0,0 );
	*m_Points.bufF(FPRESS,n) = 0;

	mNumPoints++;
	return n;
}

void Particles::AddPointsInVolume ( Vector3DF min, Vector3DF max )
{
	// Determine particle density/spacing
	if ( m_Params.pspacing == 0 ) {
		// Determine spacing from density
		m_Params.pdist = pow ( (float) m_Params.pmass / m_Params.prest_dens, 1/3.0f );	
		m_Params.pspacing = m_Params.pdist*0.87f / m_Params.sim_scale;
	} else {
		// Determine density from spacing
		m_Params.pdist = m_Params.pspacing * m_Params.sim_scale / 0.87f;
		m_Params.prest_dens = m_Params.pmass / pow ( (float) m_Params.pdist, 3.0f );
	}
	dbgprintf ( "  AddPointsInVolume. Density: %f, Spacing: %f, PDist: %f\n", m_Params.prest_dens, m_Params.pspacing, m_Params.pdist );

	// Distribute points at rest spacing
	float spacing = m_Params.pspacing;
	float offs = 0;
	Vector3DF pos;
	int p, cntx, cntz;
	float dx, dy, dz;	
	cntx = (int) ceil( (max.x-min.x-offs) / spacing );
	cntz = (int) ceil( (max.z-min.z-offs) / spacing );
	int cnt = cntx * cntz;
	int c2;	
	
	min += offs;
	max -= offs;
	dx = max.x-min.x;
	dy = max.y-min.y;
	dz = max.z-min.z;

	Vector3DF rnd;
		
	c2 = cnt/2;
	for (pos.y = min.y; pos.y <= max.y; pos.y += spacing ) {	
		for (int xz=0; xz < cnt; xz++ ) {
			
			pos.x = min.x + (xz % int(cntx))*spacing;
			pos.z = min.z + (xz / int(cntx))*spacing;
			p = AddParticle ();							// increments mNumPoints (will not exceed mMaxPoints)

			if ( p != -1 ) {
				rnd.Random ( 0, spacing, 0, spacing, 0, spacing );					
				*m_Points.bufF3(FPOS,p) = pos + rnd;
				
				Vector3DF pnt ( (pos.x-min.x)/dx, 0.f, (pos.z-min.z)/dz );

				Vector3DF clr ( 0.f, pnt.x, (pnt.x + pnt.z)*0.5f  );				
				clr *= 0.8f; 
				clr += 0.2f;				
				clr.Clamp (0, 1.0);								
				*m_Points.bufUI(FCLR,p) = COLORA( clr.x, clr.y, clr.z, 1); 

				// = COLORA( 0.25, +0.25 + (y-min.y)*.75/dy, 0.25 + (z-min.z)*.75/dz, 1);  // (x-min.x)/dx
			}
		}
	}		
	// Set number in use
	m_Points.SetNum ( mNumPoints );	
	m_PointsTemp.SetNum ( mNumPoints );
}


void Particles::Run ()
{
  PERF_PUSH ("insert");   InsertParticles ();     PERF_POP();
  PERF_PUSH ("prefix");   PrefixScanParticles();  PERF_POP();
  PERF_PUSH ("count");    CountingSort ();        PERF_POP();					
  PERF_PUSH ("pressure"); ComputePressure();      PERF_POP();
  PERF_PUSH ("force");    ComputeForce ();        PERF_POP();
  PERF_PUSH ("advance");  Advance ();             PERF_POP();
	
	cuCtxSynchronize();

	AdvanceTime ();
}

void Particles::AdvanceTime ()
{
	m_Params.time += m_Params.dt;
	m_Frame++;
}

void Particles::DebugPrintMemory ()
{
	int psize = 4*sizeof(Vector3DF) + sizeof(uint) + sizeof(unsigned short) + 2*sizeof(float) + sizeof(int) + sizeof(int)+sizeof(int);
	int gsize = 2*sizeof(int);
	int nsize = sizeof(int) + sizeof(float);
		
	dbgprintf ( "MEMORY:\n");	
	dbgprintf ( "  Particles:              %d, %f MB (%f)\n", mNumPoints, (psize*mNumPoints)/1048576.0, (psize*mMaxPoints)/1048576.0);
	dbgprintf ( "  Acceleration Grid:      %d, %f MB\n",	  m_Params.gridTotal, (gsize * m_Params.gridTotal)/1048576.0 );
}

void Particles::Advance ()
{	
	if ( m_bGPU ) {	

		void* args[4] = { &m_Params.time, &m_Params.dt, &m_Params.sim_scale, &m_Params.pnum };
		cuCheck ( cuLaunchKernel ( m_Func[FUNC_ADVANCE],  m_Params.numBlocks, 1, 1, m_Params.numThreads, 1, 1, 0, NULL, args, NULL), "Advance", "cuLaunch", "FUNC_ADVANCE", m_bDebug);

	} else {						

		Vector3DF norm, z;
		Vector3DF dir, accel;
		Vector3DF vnext;
		Vector3DF bmin, bmax;
		Vector4DF clr;
		float adj;
		float AL, AL2, SL, SL2, ss, radius;
		float stiff, damp, speed, diff; 
	
		AL = m_Params.AL;	AL2 = AL*AL;
		SL = m_Params.VL;	SL2 = SL*SL;
	
		stiff = m_Params.bound_stiff;
		damp = m_Params.bound_damp;
		radius = m_Params.pradius;
		bmin = m_Params.gridMin;
		bmax = m_Params.gridMax;
		ss = m_Params.sim_scale;

		// Get particle buffers
		Vector3DF*	ppos =		m_Points.bufF3(FPOS);
		Vector3DF*	pvel =		m_Points.bufF3(FVEL);
		Vector3DF*	pveleval =	m_Points.bufF3(FVEVAL);
		Vector3DF*	pforce =	m_Points.bufF3(FFORCE);
		uint*		pclr =		m_Points.bufUI(FCLR);
		float*		ppress =	m_Points.bufF(FPRESS);	

		// Advance each particle
		for ( int n=0; n < NumPoints(); n++ ) {

			if ( *m_Points.bufI(FGCELL,n) == GRID_UNDEF) continue;

			// Compute Acceleration		
			accel = *pforce;
			accel *= m_Params.pmass;
	
			// Boundary Conditions
			// Y-axis walls
			diff = radius - ( ppos->y - (bmin.y+ (ppos->x-bmin.x)*m_Params.bound_slope) )*ss;
			if (diff > EPSILON ) {			
				norm.Set ( -m_Params.bound_slope, 1.0f - m_Params.bound_slope, 0 );
				adj = stiff * diff - damp * (float) norm.Dot ( *pveleval );
				accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;
			}		
			diff = radius - ( bmax.y - ppos->y )*ss;
			if (diff > EPSILON) {
				norm.Set ( 0, -1, 0 );
				adj = stiff * diff - damp * (float) norm.Dot ( *pveleval );
				accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;
			}		
		
			// X-axis walls
			diff = radius - ( ppos->x - (bmin.x + (sin(m_Params.time * m_Params.bound_wall_freq)+1)*0.5f * m_Params.bound_wall_force) )*ss;	
			//diff = 2 * radius - ( p->pos.x - min.x + (sin(m_Time*10.0)-1) * m_Param[FORCE_XMIN_SIN] )*ss;	
			if (diff > EPSILON ) {
				norm.Set ( 1.0, 0, 0 );
				adj = stiff * diff - damp * (float) norm.Dot ( *pveleval ) ;
				accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;					
			}

			diff = radius - ( (bmax.x - (sin(m_Params.time *m_Params.bound_wall_freq)+1)*0.5f* m_Params.bound_wall_force) - ppos->x )*ss;	
			if (diff > EPSILON) {
				norm.Set ( -1, 0, 0 );
				adj = stiff * diff - damp * (float) norm.Dot ( *pveleval );
				accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;
			}

			// Z-axis walls
			diff = radius - ( ppos->z - bmin.z )*ss;			
			if (diff > EPSILON) {
				norm.Set ( 0, 0, 1 );
				adj = stiff * diff - damp * (float) norm.Dot ( *pveleval );
				accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;
			}
			diff = radius - ( bmax.z - ppos->z )*ss;
			if (diff > EPSILON) {
				norm.Set ( 0, 0, -1 );
				adj = stiff * diff - damp * (float) norm.Dot ( *pveleval );
				accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;
			}

			// Plane gravity
			accel += m_Params.gravity;

			// Point gravity
			if ( m_Params.grav_pos.x > 0 && m_Params.grav_amt > 0 ) {
				norm.x = ( ppos->x - m_Params.grav_pos.x );
				norm.y = ( ppos->y - m_Params.grav_pos.y );
				norm.z = ( ppos->z - m_Params.grav_pos.z );
				norm.Normalize ();
				norm *= m_Params.grav_amt;
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
			vnext = accel * m_Params.dt + *pvel;	// v(t+1/2) = v(t-1/2) + a(t) dt
			*pveleval = (*pvel + vnext) * 0.5f;		// v(t+1) = [v(t-1/2) + v(t+1/2)] * 0.5		used to compute forces later
			*ppos += vnext * (m_Params.dt/ss);		// p(t+1) = p(t) + v(t+1/2) dt
		
			ppos++;
			pvel++;
			pveleval++;
			pforce++;			
			ppress++;
			pclr++;
		}
	}
}

// Ideal grid cell size (gs) = 2 * smoothing radius = 0.02*2 = 0.04
// Ideal domain size = k * gs / d = k*0.02*2/0.005 = k*8 = {8, 16, 24, 32, 40, 48, ..}
//    (k = number of cells, gs = cell size, d = simulation scale)
//
void Particles::RebuildAccelGrid ()
{
	// Grid size - cell spacing in SPH units
	m_Params.grid_size = 1.5 * m_Params.psmoothradius / m_Params.grid_density;	
																					
	// Grid bounds - one cell beyond fluid domain
	m_Params.gridMin = m_Params.bound_min;		m_Params.gridMin -= float(2.0*(m_Params.grid_size / m_Params.sim_scale ));
	m_Params.gridMax = m_Params.bound_max;		m_Params.gridMax += float(2.0*(m_Params.grid_size / m_Params.sim_scale ));
	m_Params.gridSize = m_Params.gridMax - m_Params.gridMin;	
	
	float grid_size = m_Params.grid_size;
	float world_cellsize = grid_size / m_Params.sim_scale;		// cell spacing in world units
	float sim_scale = m_Params.sim_scale;

	// Grid res - grid volume uniformly sub-divided by grid size
	m_Params.gridRes.x = (int) ceil ( m_Params.gridSize.x / world_cellsize );		// Determine grid resolution
	m_Params.gridRes.y = (int) ceil ( m_Params.gridSize.y / world_cellsize );
	m_Params.gridRes.z = (int) ceil ( m_Params.gridSize.z / world_cellsize );
	m_Params.gridSize.x = m_Params.gridRes.x * world_cellsize;						// Adjust grid size to multiple of cell size
	m_Params.gridSize.y = m_Params.gridRes.y * world_cellsize;
	m_Params.gridSize.z = m_Params.gridRes.z * world_cellsize;	
	m_Params.gridDelta = Vector3DF(m_Params.gridRes) / m_Params.gridSize;		// delta = translate from world space to cell #	
	
	// Grid total - total number of grid cells
	m_Params.gridTotal = (int) (m_Params.gridRes.x * m_Params.gridRes.y * m_Params.gridRes.z);

	// Number of cells to search:
	// n = (2r / w) +1,  where n = 1D cell search count, r = search radius, w = world cell width
	//
	m_Params.gridSrch = (int) (floor(2.0f*(m_Params.psmoothradius / sim_scale) / world_cellsize) + 1.0f);
	if ( m_Params.gridSrch < 2 ) m_Params.gridSrch = 2;
	m_Params.gridAdjCnt = m_Params.gridSrch * m_Params.gridSrch * m_Params.gridSrch;
	m_Params.gridScanMax = m_Params.gridRes - Vector3DI( m_Params.gridSrch, m_Params.gridSrch, m_Params.gridSrch );

	if ( m_Params.gridSrch > 6 ) {
		dbgprintf ( "ERROR: Neighbor search is n > 6. \n " );
		exit(-1);
	}

	// Grid thread blocks
	// (not used currently)
	/* int threadsPerBlock = 512;
	int cnt = m_Params.gridTotal;
	computeNumBlocks ( m_Params.gridTotal, threadsPerBlock, m_Params.gridBlocks, m_Params.gridThreads);			// grid blocks
	m_Params.szGrid = (m_Params.gridBlocks * m_Params.gridThreads);*/

	// Auxiliary buffers - prefix sums sizes
	int blockSize = SCAN_BLOCKSIZE << 1;
	int numElem1 = m_Params.gridTotal;
	int numElem2 = int ( numElem1 / blockSize ) + 1;
	int numElem3 = int ( numElem2 / blockSize ) + 1;	

	// Allocate acceleration
	m_Accel.DeleteAllBuffers ();
	m_Accel.AddBuffer ( AGRID,		"grid",		sizeof(uint), mMaxPoints,			DT_CUMEM );
	m_Accel.AddBuffer ( AGRIDCNT,	"gridcnt",	sizeof(uint), m_Params.gridTotal,	DT_CUMEM );
	m_Accel.AddBuffer ( AGRIDOFF,	"gridoff",	sizeof(uint), m_Params.gridTotal,	DT_CUMEM );
	m_Accel.AddBuffer ( AAUXARRAY1, "aux1",		sizeof(uint), numElem2,				DT_CUMEM );
	m_Accel.AddBuffer ( AAUXSCAN1,  "scan1",	sizeof(uint), numElem2,				DT_CUMEM );
	m_Accel.AddBuffer ( AAUXARRAY2, "aux2",		sizeof(uint), numElem3,				DT_CUMEM );
	m_Accel.AddBuffer ( AAUXSCAN2,  "scan2",	sizeof(uint), numElem3,				DT_CUMEM );

	for (int b=0; b <= AAUXSCAN2; b++)
		m_Accel.SetBufferUsage ( b, DT_UINT );		// for debugging

	// Grid adjacency lookup - stride to access neighboring cells in all 6 directions
	int cell = 0;
	for (int y=0; y < m_Params.gridSrch; y++ ) 
		for (int z=0; z < m_Params.gridSrch; z++ ) 
			for (int x=0; x < m_Params.gridSrch; x++ ) 
				m_Params.gridAdj [ cell++]  = ( y * m_Params.gridRes.z+ z )*m_Params.gridRes.x +  x ;			

	// Update gpu access
	UpdateGPUAccess();	
	UpdateParams ();

	// Done
	dbgprintf ( "  Accel Grid: %d, t:%dx%d=%d, bufGrid:%d, Res: %dx%dx%d\n", m_Params.gridTotal, m_Params.gridBlocks, m_Params.gridThreads, m_Params.gridBlocks*m_Params.gridThreads, m_Params.szGrid, (int) m_Params.gridRes.x, (int) m_Params.gridRes.y, (int) m_Params.gridRes.z );		
}

int Particles::getGridCell ( int p, Vector3DI& gc )
{
	return getGridCell ( m_Points.bufF3(FPOS)[p], gc );
}
int Particles::getGridCell ( Vector3DF& pos, Vector3DI& gc )
{
	gc.x = (int)( (pos.x - m_Params.gridMin.x) * m_Params.gridDelta.x);			// Cell in which particle is located
	gc.y = (int)( (pos.y - m_Params.gridMin.y) * m_Params.gridDelta.y);
	gc.z = (int)( (pos.z - m_Params.gridMin.z) * m_Params.gridDelta.z);		
	return (int)( (gc.y*m_Params.gridRes.z + gc.z)*m_Params.gridRes.x + gc.x);		
}
Vector3DI Particles::getCell ( int c )
{
	Vector3DI gc;
	int xz = m_Params.gridRes.x*m_Params.gridRes.z;
	gc.y = c / xz;				c -= gc.y*xz;
	gc.z = c / m_Params.gridRes.x;		c -= gc.z*m_Params.gridRes.x;
	gc.x = c;
	return gc;
}

void Particles::UpdateGPUAccess()
{
	// Update GPU access 
	m_Points.UpdateGPUAccess ();
	m_PointsTemp.UpdateGPUAccess ();
	m_Accel.UpdateGPUAccess ();
}

void Particles::UpdateParams ()
{	
	m_Params.gravity = m_Params.grav_dir * m_Params.grav_amt;
	m_Params.AL2 = m_Params.AL * m_Params.AL;
	m_Params.VL2 = m_Params.VL * m_Params.VL;

	float sr = m_Params.psmoothradius;
	m_Params.r2 = sr * sr;
	m_Params.pdist = pow ( m_Params.pmass / m_Params.prest_dens, 1/3.0f );
	m_Params.poly6kern = 315.0f / (64.0f * 3.141592f * pow( sr, 9.0f) );
	m_Params.spikykern = -45.0f / (3.141592f * pow( sr, 6.0f) );
	m_Params.lapkern = 45.0f / (3.141592f * pow( sr, 6.0f) );	
	m_Params.gausskern = 1.0f / pow(3.141592f * 2.0f*sr*sr, 3.0f/2.0f);

	m_Params.d2 = m_Params.sim_scale * m_Params.sim_scale;
	m_Params.rd2 = m_Params.r2 / m_Params.d2;
	m_Params.vterm = m_Params.lapkern * m_Params.pvisc;
	
	if ( m_bGPU ) {
		// Transfer sim params to device
		cuCheck ( cuMemcpyHtoD ( m_cuParams, &m_Params,	sizeof(FParams_t) ), "FluidParamCUDA", "cuMemcpyHtoD", "cuFParams", m_bDebug);
	}
}

// DebugPrint - useful function to print contents of any buffer, and its cpu & gpu pointer locations
// 
void Particles::DebugPrint ( DataX& dat, int i, int start, int disp)
{
	char buf[256];
 
	dat.Retrieve ( i );

	cuCheck( cuCtxSynchronize (), "DebugPrint", "cuCtxSync", "", m_bDebug );

	dbgprintf ("---- cpu: %012llx, gpu: %012llx\n", dat.cpu(FPOS), dat.gpu(FPOS) );	
	for (int n=start; n < start+disp; n++ ) {
		dbgprintf ( "%d: %s\n", n, dat.printElem  (i, n, buf) );
	}
	
}

void Particles::CopyAllToTemp ()
{	
	m_Points.CopyAllBuffers ( &m_PointsTemp, DT_CUMEM );	// gpu-to-gpu copy all buffers into temp
}

void Particles::CommitAll ()
{	
	m_Points.CommitAll (); 									// send particle buffers to GPU
}

void Particles::Retrieve ( int buf )
{	
	m_Points.Retrieve ( buf );								// return particle buffers to GPU
}


//-------------------------------------------------------------------- SIMULATION CODE

void Particles::InsertParticles ()
{
	if ( m_bGPU ) {

		// Reset all grid cells to empty	
		cuCheck ( cuMemsetD8 ( m_Accel.gpu(AGRIDCNT), 0,	m_Params.gridTotal*sizeof(uint) ), "InsertParticlesCUDA", "cuMemsetD8", "AGRIDCNT", m_bDebug );
		cuCheck ( cuMemsetD8 ( m_Accel.gpu(AGRIDOFF), 0,	m_Params.gridTotal*sizeof(uint) ), "InsertParticlesCUDA", "cuMemsetD8", "AGRIDOFF", m_bDebug );

		cuCheck ( cuMemsetD8 ( m_Points.gpu(FGCELL), 0,		mNumPoints*sizeof(int) ), "InsertParticlesCUDA", "cuMemsetD8", "FGCELL", m_bDebug );
		cuCheck ( cuMemsetD8 ( m_Points.gpu(FGNDX),	 0,		mNumPoints*sizeof(int) ), "InsertParticlesCUDA", "cuMemsetD8", "FGNDX", m_bDebug );

		void* args[1] = { &mNumPoints };
		cuCheck(cuLaunchKernel(m_Func[FUNC_INSERT], m_Params.numBlocks, 1, 1, m_Params.numThreads, 1, 1, 0, NULL, args, NULL),
			"InsertParticlesCUDA", "cuLaunch", "FUNC_INSERT", m_bDebug);

	} else {
		// Reset all grid cells to empty	
		memset( m_Accel.bufUI(AGRIDCNT),	0,	m_Params.gridTotal*sizeof(uint));
		memset( m_Accel.bufUI(AGRIDOFF),	0,	m_Params.gridTotal*sizeof(uint));
		memset( m_Points.bufUI(FGCELL),		0,	mNumPoints*sizeof(int));
		memset( m_Points.bufUI(FGNDX),		0,	mNumPoints*sizeof(int));

		float poff = m_Params.psmoothradius / m_Params.sim_scale;

		// Insert each particle into spatial grid
		Vector3DF gcf;
		Vector3DI gc;
		int gs; 
		Vector3DF*	ppos =		m_Points.bufF3(FPOS);		
		uint*		pgcell =	m_Points.bufUI(FGCELL);
		uint*		pgndx =		m_Points.bufUI(FGNDX);		

		for ( int n=0; n < NumPoints(); n++ ) {		
		
			gcf = (*ppos - m_Params.gridMin) * m_Params.gridDelta; 
			gc = Vector3DI( int(gcf.x), int(gcf.y), int(gcf.z) );
			gs = (gc.y * m_Params.gridRes.z + gc.z)*m_Params.gridRes.x + gc.x;
	
			if ( gc.x >= 1 && gc.x <= m_Params.gridScanMax.x && gc.y >= 1 && gc.y <= m_Params.gridScanMax.y && gc.z >= 1 && gc.z <= m_Params.gridScanMax.z ) {
				*pgcell = gs;
				(*m_Accel.bufUI(AGRIDCNT, gs))++;
				*pgndx = *m_Accel.bufUI(AGRIDCNT, gs);	
			} else {
				*pgcell = GRID_UNDEF;				
			}			
			ppos++;			
			pgcell++;
			pgndx++;
		}

		// debugging
		pgcell =	m_Points.bufUI(FGCELL);
		for (int n=0; n < 10; n++) 
			dbgprintf ( "%d: %d\n", n, *pgcell++ );
	}
}


void Particles::PrefixScanParticles ()
{
	if ( m_bGPU ) {

		// Prefix Sum - determine grid offsets
		int blockSize = SCAN_BLOCKSIZE << 1;
		int numElem1 = m_Params.gridTotal;		
		int numElem2 = int ( numElem1 / blockSize ) + 1;
		int numElem3 = int ( numElem2 / blockSize ) + 1;
		int threads = SCAN_BLOCKSIZE;
		int zero_offsets = 1;
		int zon = 1;

		CUdeviceptr array1  = m_Accel.gpu(AGRIDCNT);		// input
		CUdeviceptr scan1   = m_Accel.gpu(AGRIDOFF);		// output
		CUdeviceptr array2  = m_Accel.gpu(AAUXARRAY1);
		CUdeviceptr scan2   = m_Accel.gpu(AAUXSCAN1);
		CUdeviceptr array3  = m_Accel.gpu(AAUXARRAY2);
		CUdeviceptr scan3   = m_Accel.gpu(AAUXSCAN2);

		if ( numElem1 > SCAN_BLOCKSIZE*xlong(SCAN_BLOCKSIZE)*SCAN_BLOCKSIZE) {
			dbgprintf ( "ERROR: Number of elements exceeds prefix sum max. Adjust SCAN_BLOCKSIZE.\n" );
		}

		void* argsA[5] = {&array1, &scan1, &array2, &numElem1, &zero_offsets }; // sum array1. output -> scan1, array2
		cuCheck ( cuLaunchKernel ( m_Func[FUNC_FPREFIXSUM], numElem2, 1, 1, threads, 1, 1, 0, NULL, argsA, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXSUM:A", m_bDebug);

		void* argsB[5] = { &array2, &scan2, &array3, &numElem2, &zon }; // sum array2. output -> scan2, array3
		cuCheck ( cuLaunchKernel ( m_Func[FUNC_FPREFIXSUM], numElem3, 1, 1, threads, 1, 1, 0, NULL, argsB, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXSUM:B", m_bDebug);

		if ( numElem3 > 1 ) {
			CUdeviceptr nptr = {0};
			void* argsC[5] = { &array3, &scan3, &nptr, &numElem3, &zon };	// sum array3. output -> scan3
			cuCheck ( cuLaunchKernel ( m_Func[FUNC_FPREFIXSUM], 1, 1, 1, threads, 1, 1, 0, NULL, argsC, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXFIXUP:C", m_bDebug);

			void* argsD[3] = { &scan2, &scan3, &numElem2 };	// merge scan3 into scan2. output -> scan2
			cuCheck ( cuLaunchKernel ( m_Func[FUNC_FPREFIXFIXUP], numElem3, 1, 1, threads, 1, 1, 0, NULL, argsD, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXFIXUP:D", m_bDebug);
		}

		void* argsE[3] = { &scan1, &scan2, &numElem1 };		// merge scan2 into scan1. output -> scan1
		cuCheck ( cuLaunchKernel ( m_Func[FUNC_FPREFIXFIXUP], numElem2, 1, 1, threads, 1, 1, 0, NULL, argsE, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXFIXUP:E", m_bDebug);

	} else {
		
		// CPU prefix scan

		int numCells = m_Params.gridTotal;		
		uint* mgcnt = m_Accel.bufUI(AGRIDCNT);
		uint* mgoff = m_Accel.bufUI(AGRIDOFF);
		int sum = 0;
		for (int n=0; n < numCells; n++) {
			mgoff[n] = sum;
			sum += mgcnt[n];
		}		
	}

}

void Particles::CountingSort ()
{
	if ( m_bGPU ) {

		// Transfer particle data to temp buffers 
		//  (required by algorithm, gpu-to-gpu copy, no sync needed)	
		CopyAllToTemp ();

		void* args[1] = { &mNumPoints };
		cuCheck ( cuLaunchKernel ( m_Func[FUNC_COUNTING_SORT], m_Params.numBlocks, 1, 1, m_Params.numThreads, 1, 1, 0, NULL, args, NULL),
					"CountingSortFullCUDA", "cuLaunch", "FUNC_COUNTING_SORT", m_bDebug );

	} else {



	}
}

// Compute Pressures - Using spatial grid, and also create neighbor table
void Particles::ComputePressure ()
{
	if ( m_bGPU ) {

		void* args[1] = { &mNumPoints };
		cuCheck ( cuLaunchKernel ( m_Func[FUNC_COMPUTE_PRESS],  m_Params.numBlocks, 1, 1, m_Params.numThreads, 1, 1, 0, NULL, args, NULL), "ComputePressureCUDA", "cuLaunch", "FUNC_COMPUTE_PRESS", m_bDebug );

	} else {

		int i, j, cnt = 0;	
		float sum, dsq, c;
		float d = m_Params.sim_scale;
		float d2 = d*d;
		float radius = m_Params.psmoothradius / m_Params.psmoothradius;
	
		// Get particle buffers
		Vector3DF*	ipos =		m_Points.bufF3(FPOS);		
		float*		ipress =	m_Points.bufF(FPRESS);			

		Vector3DF	dst;
		int			nadj = (m_Params.gridRes.z + 1)*m_Params.gridRes.x + 1;
		uint*		m_Grid =	m_Points.bufUI(AGRID);
		uint*		m_GridCnt = m_Points.bufUI(AGRIDCNT);
	
		int nbrcnt = 0;
		int srch = 0;

		for ( i=0; i < NumPoints(); i++ ) {

			sum = 0.0;

			if ( m_Points.bufI(FGCELL)[i] != GRID_UNDEF ) {
				for (int cell=0; cell < m_Params.gridAdjCnt; cell++) {
					j = m_Grid [   m_Points.bufI(FGCELL)[i] - nadj + m_Params.gridAdj[cell] ] ;
					while ( j != GRID_UNDEF ) {
						//if ( i==j ) { j = *m_Points.bufUI(FGNEXT,j); continue; }
						dst = m_Points.bufF3(FPOS)[j];
						dst -= *ipos;
						dsq = d2*(dst.x*dst.x + dst.y*dst.y + dst.z*dst.z);
						if ( dsq <= m_Params.r2 ) {
							c =  m_Params.r2 - dsq;
							sum += c * c * c;
							nbrcnt++;
							/*nbr = AddNeighbor();			// get memory for new neighbor						
							*(m_NeighborTable + nbr) = j;
							*(m_NeighborDist + nbr) = sqrt(dsq);
							inbr->num++;*/
						}
						srch++;
						//j = m_Points.bufI(FGNEXT)[j];
					}
				}
			}		
			*ipress = sum * m_Params.pmass * m_Params.poly6kern;

			ipos++;		
			ipress++;
		}
	}
}

// Compute Forces
void Particles::ComputeForce ()
{
	if ( m_bGPU ) {
	
		void* args[1] = { &mNumPoints };
		cuCheck ( cuLaunchKernel ( m_Func[FUNC_COMPUTE_FORCE],  m_Params.numBlocks, 1, 1, m_Params.numThreads, 1, 1, 0, NULL, args, NULL), "ComputeForceCUDA", "cuLaunch", "FUNC_COMPUTE_FORCE", m_bDebug);

	} else {

		Vector3DF force;
		register float pterm, vterm, dterm;
		int i, j;
		float c, d;
		float dx, dy, dz;
		float mR, visc;	

		d = m_Params.sim_scale;
		mR = m_Params.psmoothradius;
		visc = m_Params.pvisc;
	
		// Get particle buffers
		Vector3DF*	ipos =		m_Points.bufF3(FPOS);		
		Vector3DF*	iveleval =	m_Points.bufF3(FVEVAL);		
		Vector3DF*	iforce =	m_Points.bufF3(FFORCE);		
		float*		ipress =	m_Points.bufF(FPRESS);	
	
		Vector3DF	jpos;
		float		jdist;
		float		jpress;
		Vector3DF	jveleval;
		float		dsq;
		float		d2 = d*d;
		int			nadj = (m_Params.gridRes.z + 1)*m_Params.gridRes.x + 1;
		uint* m_Grid =		m_Points.bufUI(AGRID);
		uint* m_GridCnt =	m_Points.bufUI(AGRIDCNT);
	
		float pi, pj;

		for ( i=0; i < NumPoints(); i++ ) {

			iforce->Set ( 0, 0, 0 );

			if ( *m_Points.bufI(FGCELL,i) != GRID_UNDEF ) {
				for (int cell=0; cell < m_Params.gridAdjCnt; cell++) {
					j = m_Grid [  m_Points.bufI(FGCELL)[i] - nadj + m_Params.gridAdj[cell] ];
					pi = m_Points.bufF(FPRESS)[i];

					while ( j != GRID_UNDEF ) {
						//if ( i==j ) { j = *m_Points.bufUI(FGNEXT,j); continue; }
						jpos = m_Points.bufF3(FPOS)[j];
						dx = ( ipos->x - jpos.x);		// dist in cm
						dy = ( ipos->y - jpos.y);
						dz = ( ipos->z - jpos.z);
						dsq = d2*(dx*dx + dy*dy + dz*dz);
						if ( dsq <= m_Params.r2 ) {

							jdist = sqrt(dsq);
						
							pj = m_Points.bufF(FPRESS)[j];												
							jveleval = m_Points.bufF3(FVEVAL)[j];
							dx = ( ipos->x - jpos.x);		// dist in cm
							dy = ( ipos->y - jpos.y);
							dz = ( ipos->z - jpos.z);
							c = (mR-jdist);
							pterm = d * -0.5f * c * m_Params.spikykern * ( pi + pj ) / jdist;
							dterm = c / ( m_Points.bufF(FPRESS)[i] *  m_Points.bufF(FPRESS)[j] );
							vterm = m_Params.lapkern * visc;
							iforce->x += ( pterm * dx + vterm * ( jveleval.x - iveleval->x) ) * dterm;
							iforce->y += ( pterm * dy + vterm * ( jveleval.y - iveleval->y) ) * dterm;
							iforce->z += ( pterm * dz + vterm * ( jveleval.z - iveleval->z) ) * dterm;
						}
						//j = m_Points.bufI(FGNEXT,j);
					}
				}
			}
			ipos++;
			iveleval++;
			iforce++;
			ipress++;		
		}
	}
}

// Draw
// OpenGL draw using shaders and functions found in nv_gui.cpp
// VBO buffers were created in RebuildParticles using CUDA-GL interop 
//
void Particles::Draw ( int frame, Camera3D* cam, float rad )
{
	Vector3DF* ppos;
	uint* pclr;
	
	// Render points	
	PERF_PUSH ("draw");

	selfDraw3D ( cam, SPNT );
	setPreciseEye ( SPNT, cam );
	setOffset ( SPNT, 0, 0, 0, 1 );
	checkGL ( "start pnt shader" );
		
	glEnableVertexAttribArray ( 0 );
	glEnableVertexAttribArray ( 1 );
	glEnableVertexAttribArray ( 2 );

	glBindBuffer ( GL_ARRAY_BUFFER, m_Points.glid(FPOS) );	
	glVertexAttribPointer ( 0, 3, GL_FLOAT, GL_FALSE, sizeof(Vector3DF), 0 );			
	checkGL ( "bind pos" ); 
		
	glBindBuffer ( GL_ARRAY_BUFFER, m_Points.glid(FCLR) );
	glVertexAttribPointer ( 1, 1, GL_FLOAT, GL_FALSE, sizeof(uint), 0 );
	checkGL ( "bind clr" );

	glBindBuffer ( GL_ARRAY_BUFFER, m_Points.glid(FVEL) );
	glVertexAttribPointer ( 2, 3, GL_FLOAT, GL_FALSE, sizeof(Vector3DF), 0 );
		
	glDrawArrays ( GL_POINTS, 0, NumPoints() );
	checkGL ( "draw pnts" );
		
	selfEndDraw3D ();
	checkGL ( "end pnt shader" );

	PERF_POP();
}

