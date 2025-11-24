//-----------------------------------------------------------------------------
// FLUIDS v.3.1 - SPH Fluid Simulator for CPU and GPU
// Copyright (C) 2012-2013, 2021. Rama Hoetzlein, http://fluids3.com
//-----------------------------------------------------------------------------

#include <GL/glew.h>
#include <cudaGL.h>	
#include <cuda.h>	
#include <assert.h>
#include <stdio.h>

#include "timex.h"
#include "main.h"
#include "fluid_system.h"
#include "nv_gui.h"


#define EPSILON			0.00001f			// for collision detection
#define SCAN_BLOCKSIZE		512				// must match value in fluid_system_cuda.cu


bool cuCheck (CUresult launch_stat, char* method, char* apicall, char* arg, bool bDebug)
{
	CUresult kern_stat = CUDA_SUCCESS;

	if (bDebug) {
		kern_stat = cuCtxSynchronize();
	}
	if (kern_stat != CUDA_SUCCESS || launch_stat != CUDA_SUCCESS) {
		const char* launch_statmsg = "";
		const char* kern_statmsg = "";
		cuGetErrorString(launch_stat, &launch_statmsg);
		cuGetErrorString(kern_stat, &kern_statmsg);
		dbgprintf("------- CUDA ERROR:\n");
		dbgprintf("  Launch status: %s\n", launch_statmsg);
		dbgprintf("  Kernel status: %s\n", kern_statmsg);
		dbgprintf("  Caller: FluidSystem::%s\n", method);
		dbgprintf("  Call:   %s\n", apicall);
		dbgprintf("  Args:   %s\n", arg);

		if (bDebug) {
			dbgprintf("  Generating assert to examine call stack.\n");
			assert(0);		// debug - trigger break (see call stack)
		}
		else {
			exit(-1);
		}
		return false;
	}
	return true;
}

FluidSystem::FluidSystem ()
{
	mNumPoints = 0;
	mMaxPoints = 0;
	mbRecord = false;
	mbRecordBricks = false;
	mSelected = -1;
	m_Frame = 0;
	m_Module = 0;
	m_Thresh = 0;	
	m_NeighborTable = 0x0;
	m_NeighborDist = 0x0;		
	for (int n=0; n < FUNC_MAX; n++ ) m_Func[n] = (CUfunction) -1;
}

FluidSystem::~FluidSystem() 
{
	if (mSaveNdx != 0x0) free(mSaveNdx);
	if (mSaveCnt != 0x0) free(mSaveCnt);
	if (mSaveNeighbors != 0x0)	free(mSaveNeighbors);

	if (m_Module != 0) {
		cuCheck(cuModuleUnload(m_Module), "~FluidSystem()", "cuModuleUnload", "m_Module", mbDebug);
	}
}

void FluidSystem::LoadKernel ( int fid, std::string func )
{
	char cfn[512];		strcpy ( cfn, func.c_str() );

	if ( m_Func[fid] == (CUfunction) -1 )
		cuCheck ( cuModuleGetFunction ( &m_Func[fid], m_Module, cfn ), "LoadKernel", "cuModuleGetFunction", cfn, mbDebug );	
}

// Must have a CUDA context to initialize
void FluidSystem::Initialize (bool bGPU)
{
	m_bCPU = !bGPU;

	cuCheck ( cuModuleLoad ( &m_Module, "fluid_kernels.ptx" ), "LoadKernel", "cuModuleLoad", "fluid_system_cuda.ptx", mbDebug);

	LoadKernel ( FUNC_INSERT,			"insertParticles" );
	LoadKernel ( FUNC_COUNTING_SORT,	"countingSortFull" );	
	LoadKernel ( FUNC_COMPUTE_PRESS,	"computePressure" );
	LoadKernel ( FUNC_COMPUTE_FORCE,	"computeForce" );
	LoadKernel ( FUNC_ADVANCE,			"advanceParticles" );
	//LoadKernel ( FUNC_EMIT,			"emitParticles" );
	LoadKernel ( FUNC_FPREFIXSUM,		"prefixSum" );
	LoadKernel ( FUNC_FPREFIXFIXUP,		"prefixFixup" );

	size_t len = 0;
	
	cuCheck ( cuModuleGetGlobal ( &cuFBuf, &len,		m_Module, "fbuf" ),		"LoadKernel", "cuModuleGetGlobal", "cuFBuf", mbDebug);
	assert ( len == sizeof(FBufs) );		// if this assert fails, the CPU and GPU layout of FBufs struct is incorrect

	cuCheck ( cuModuleGetGlobal ( &cuFParams, &len,		m_Module, "fparam" ),		"LoadKernel", "cuModuleGetGlobal", "cuFParams", mbDebug);
	assert ( len == sizeof(FParams) );

	// Clear all buffers
	memset ( &m_Fluid, 0,		sizeof(FBufs) );	
	memset ( &m_Params, 0,		sizeof(FParams) );
	
	m_Params.mode	=		0;
	m_Params.example =		2;
	m_Params.grid_density = 2.0;
	m_Params.pnum	=		65536;

	// Allocate the sim parameters
	AllocateBuffer ( "fparams", FPARAMS,		sizeof(FParams),	1,	 GPU_SINGLE, CPU_OFF );
}

void FluidSystem::Start ( int num )
{
	#ifdef TEST_PREFIXSUM
		TestPrefixSum ( 16*1024*1024 );		
		exit(-2);
	#endif

	m_Time = 0;

	mNumPoints = 0;			// reset count
	
	SetupDefaultParams ();	
	SetupExampleParams ();		
	mMaxPoints = num;

	// Setup stuff
	SetupKernels ();
	
	SetupSpacing ();		// defined gridMin, gridMax

	SetupGrid ();	
		
	FluidSetupCUDA ( mMaxPoints, m_GridSrch, m_GridRes,  m_GridSize, m_GridDelta, m_GridMin, m_GridMax, m_GridTotal, 0 );

	// Allocate data
	Reallocate ( mMaxPoints, m_bCPU );	

	// Create the particles (after allocate)
	SetupAddVolume ( m_Params.init_min, m_Params.init_max, m_Params.pspacing, 0.1f, m_Params.pnum );		// increases mNumPoints

	UpdateParamsCUDA ();

	TransferToCUDA (FFLUID);		 // Initial transfer

	// Update all GPU access pointers
	cuCheck( cuMemcpyHtoD(cuFBuf,		&m_Fluid,	sizeof(FBufs)),		"AllocateParticles", "cuMemcpyHtoD", "cuFBuf", mbDebug);
	cuCheck( cuMemcpyHtoD(cuFParams,	&m_Params,	sizeof(FParams)),	"AllocateParticles", "cuMemcpyHtoD", "cuFParams", mbDebug);
}

void FluidSystem::Exit ()
{
	// Free fluid buffers
	for (int n=0; n < MAX_BUF; n++ ) {
		if ( m_Fluid.bufC(n) != 0x0 )
			free ( m_Fluid.bufC(n) );
	}
}

void FluidSystem::MapBuffer ( int buf_id, bool map ) 
{
	if ( *m_Fluid.grsc(buf_id) == 0 ) return;
	if (map) {
		cuCheck ( cuGraphicsMapResources(1, m_Fluid.grsc(buf_id), 0), "", "cuGraphicsMapResrc", "", mbDebug );
		size_t sz_chk = 0;
		cuCheck ( cuGraphicsResourceGetMappedPointer ( m_Fluid.gpuptr(buf_id), &sz_chk, *m_Fluid.grsc(buf_id)),  "", "cuGraphicsResrcGetMappedPtr", "", mbDebug );			
		assert ( sz_chk == m_Fluid.size(buf_id) );
	} else {
		cuCheck ( cuGraphicsUnmapResources(1, m_Fluid.grsc(buf_id), 0), "", "cuGraphicsUnmapResrc", "", mbDebug );
	}
}

void FluidSystem::AllocateBuffer ( std::string name, int buf_id, int stride, int cnt, int gpumode, int cpumode )
{
	size_t sz = cnt * stride;
	m_Fluid.sz[buf_id] = sz;

	if (cpumode == CPU_YES) {
		char* src_buf = m_Fluid.bufC(buf_id);
		char* dest_buf = (char*) malloc( sz );
		if ( dest_buf==0x0) {
			dbgprintf ( "ERROR: Out of memory.\n" );
			exit (-7);
		}
		if (src_buf != 0x0) {
			memcpy(dest_buf, src_buf, sz );			
		}
		m_Fluid.setBuf(buf_id, dest_buf);
	}
	if (gpumode & GPU_SINGLE )	{
		if ( gpumode & GPU_GL ) {
			// opengl allocate & cuda interop
			glGenBuffers ( 1, (GLuint*) m_Fluid.glid_ptr(buf_id) );			
			glBindBuffer ( GL_ARRAY_BUFFER, m_Fluid.glid(buf_id) );
			glBufferData ( GL_ARRAY_BUFFER, sz, m_Fluid.bufI(buf_id), GL_DYNAMIC_DRAW );
			cuCheck ( cuGraphicsGLRegisterBuffer ( m_Fluid.grsc(buf_id), m_Fluid.glid(buf_id), CU_GRAPHICS_REGISTER_FLAGS_NONE ), "", "cuGraphicsGLReg", "", mbDebug );
			MapBuffer ( buf_id, true );
			
		} else {
			// cuda allocate
			if (m_Fluid.gpuptr(buf_id) != 0x0) cuCheck( cuMemFree(m_Fluid.gpu(buf_id)), "AllocateBuffer", "cuMemFree", "Fluid.gpu", mbDebug);
			cuCheck( cuMemAlloc( m_Fluid.gpuptr(buf_id), sz), "AllocateBuffer", "cuMemAlloc", "Fluid.gpu", mbDebug);
		}
	}
	dbgprintf ( "Allocated: %s, num %d, size %ld bytes\n", name.c_str(), cnt, (unsigned long) sz );
}

// Allocate particle memory
void FluidSystem::Reallocate ( int cnt, bool bCPU )
{
	int cpu_opt = (bCPU) ? CPU_YES : CPU_OFF;

	// Allocate particles
	AllocateBuffer ( "fluid",		FFLUID,		sizeof(Fluid),	cnt,			GPU_SINGLE | GPU_GL, CPU_YES );
	AllocateBuffer ( "fluidtemp",	FFLUIDTEMP,	sizeof(Fluid),	cnt,			GPU_SINGLE, cpu_opt );
	m_Params.pmem = sizeof(Fluid) * 2 * cnt;		// measure particle memory usage

	// Allocate grid
	m_Params.szGrid = (m_Params.gridBlocks * m_Params.gridThreads);	
	AllocateBuffer ( "fgrid",	FGRID,		sizeof(uint),		m_Params.szPnts,	GPU_SINGLE, CPU_YES );    // # grid elements = number of points
	AllocateBuffer ( "fgridcnt",FGRIDCNT,	sizeof(uint),		m_GridTotal,		GPU_SINGLE, CPU_YES );
	AllocateBuffer ( "fgridoff",FGRIDOFF,	sizeof(uint),		m_GridTotal,		GPU_SINGLE, CPU_YES );
	AllocateBuffer ( "fgridact",FGRIDACT,	sizeof(uint),		m_GridTotal,		GPU_SINGLE, CPU_YES );

	// Allocate auxiliary buffers (prefix sums)
	int blockSize = SCAN_BLOCKSIZE << 1;
	int numElem1 = m_GridTotal;
	int numElem2 = int ( numElem1 / blockSize ) + 1;
	int numElem3 = int ( numElem2 / blockSize ) + 1;
	AllocateBuffer ( "fauxarr1",	FAUXARRAY1,	sizeof(uint),	numElem2,		GPU_SINGLE, cpu_opt );
	AllocateBuffer ( "fauxscan1",	FAUXSCAN1,		sizeof(uint),	numElem2,	GPU_SINGLE, cpu_opt );
	AllocateBuffer ( "fauxarr2",	FAUXARRAY2,	sizeof(uint),	numElem3,		GPU_SINGLE, cpu_opt );
	AllocateBuffer ( "fauxscan2",	FAUXSCAN2,		sizeof(uint),	numElem3,	GPU_SINGLE, cpu_opt );
}

Fluid* FluidSystem::AddParticle ()
{
	if ( mNumPoints >= mMaxPoints ) return 0x0;
	
	Fluid* p = m_Fluid.pnt( mNumPoints );
	p->pos.Set ( 0,0,0 );
	p->vel.Set ( 0,0,0 );
	p->veval.Set ( 0,0,0 );
	p->force.Set ( 0,0,0 );
	p->gcell = GRID_UNDEF;
	p->gndx = 0;
	p->press = 0;	

	mNumPoints++;

	return p;
}

void FluidSystem::SetupAddVolume ( Vector3DF min, Vector3DF max, float spacing, float offs, int total )
{
	Vector3DF pos;
	float dx, dy, dz;
	int cntx, cntz;
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
	Fluid* p;
		
	c2 = cnt/2;
	for (pos.y = min.y; pos.y <= max.y; pos.y += spacing ) {	
		for (int xz=0; xz < cnt; xz++ ) {
			
			pos.x = min.x + (xz % int(cntx))*spacing;
			pos.z = min.z + (xz / int(cntx))*spacing;

			p = AddParticle ();			

			if ( p != 0x0 ) {
				rnd.Random ( 0, spacing, 0, spacing, 0, spacing );					
				p->pos = pos + rnd;
				
				Vector3DF clr ( (pos.x-min.x)/dx, 0.f, (pos.z-min.z)/dz );				
				clr *= 0.8f; 
				clr += 0.2f;				
				clr.Clamp (0, 1.0);								
				p->clr = COLORA( clr.x, clr.y, clr.z, 1); 
			}
		}
	}		
}

void FluidSystem::AddEmit ( float spacing )
{
	Fluid* p;
	Vector3DF dir;
	Vector3DF pos;
	float ang_rand, tilt_rand;
	float rnd = m_Params.emit_rate.y * 0.15f;	
	int x = (int) sqrt(m_Params.emit_rate.y);

	for ( int n = 0; n < m_Params.emit_rate.y; n++ ) {
		ang_rand = (float(rand()*2.0f/RAND_MAX) - 1.0f) * m_Params.emit_spread.x;
		tilt_rand = (float(rand()*2.0f/RAND_MAX) - 1.0f) * m_Params.emit_spread.y;
		dir.x = cos ( ( m_Params.emit_ang.x + ang_rand) * DEGtoRAD ) * sin( ( m_Params.emit_ang.y + tilt_rand) * DEGtoRAD ) * m_Params.emit_ang.z;
		dir.y = sin ( ( m_Params.emit_ang.x + ang_rand) * DEGtoRAD ) * sin( ( m_Params.emit_ang.y + tilt_rand) * DEGtoRAD ) * m_Params.emit_ang.z;
		dir.z = cos ( ( m_Params.emit_ang.y + tilt_rand) * DEGtoRAD ) * m_Params.emit_ang.z;
		pos = m_Params.emit_pos;
		pos.x += spacing * (n/x);
		pos.y += spacing * (n%x);
		
		p = AddParticle ();
		p->pos = pos;
		p->vel = dir;
		p->veval = dir;
		p->clr = COLORA ( m_Time/10.0, m_Time/5.0, m_Time /4.0, 1 );
	}
}
void FluidSystem::Run ()
{
	if ( m_bCPU ) {
		InsertParticles();
		PrefixSumCells ();
		CountingSortFull ();
		ComputePressures ();
		ComputeForces ();
		Advance ();
		// copy to GPU for render
		TransferToCUDA ( FFLUID );
	} else {
		PERF_PUSH ("insert");		InsertParticlesCUDA ( 0x0, 0x0, 0x0 );		PERF_POP();
		PERF_PUSH ("prefix");		PrefixSumCellsCUDA ( 0x0, 1 );				PERF_POP();
		PERF_PUSH ("count");		CountingSortFullCUDA ( 0x0 );				PERF_POP();	
		PERF_PUSH ("pressure");		ComputePressureCUDA();						PERF_POP();
		PERF_PUSH ("force");		ComputeForceCUDA ();						PERF_POP();
		PERF_PUSH ("advance");		AdvanceCUDA ( m_Time, m_DT, m_Params.sim_scale );		PERF_POP();
		// EmitParticlesCUDA ( m_Time, (int) m_Vec[PEMIT_RATE].x );					
		// PERF_PUSH( "transfer");		TransferFromCUDA ();	// return for rendering			PERF_POP();
		cuCtxSynchronize();
	}
	
	AdvanceTime ();
}

void FluidSystem::AdvanceTime ()
{
	m_Time += m_DT;	
	m_Frame += m_FrameRange.z;

	if ( m_Frame > m_FrameRange.y && m_FrameRange.y != -1 ) {
		m_Frame = m_FrameRange.x;		
		mbRecord = false;
		mbRecordBricks = false;		

		dbgprintf ( "Exiting.\n" );
		exit ( 1 );
	}
}

void FluidSystem::DebugPrintMemory ()
{
	int psize = 4*sizeof(Vector3DF) + sizeof(uint) + sizeof(unsigned short) + 2*sizeof(float) + sizeof(int) + sizeof(int)+sizeof(int);
	int gsize = 2*sizeof(int);
	int nsize = sizeof(int) + sizeof(float);
		
	dbgprintf ( "MEMORY:\n");	
	dbgprintf ( "  Particles:              %d, %f MB (%f)\n", mNumPoints, (psize*mNumPoints)/1048576.0, (psize*mMaxPoints)/1048576.0);
	dbgprintf ( "  Acceleration Grid:      %d, %f MB\n",	   m_GridTotal, (gsize*m_GridTotal)/1048576.0 );
	dbgprintf ( "  Acceleration Neighbors: %d, %f MB (%f)\n", m_NeighborNum, (nsize*m_NeighborNum)/1048576.0, (nsize*m_NeighborMax)/1048576.0 );
}

// Ideal grid cell size (gs) = 2 * smoothing radius = 0.02*2 = 0.04
// Ideal domain size = k*gs/d = k*0.02*2/0.005 = k*8 = {8, 16, 24, 32, 40, 48, ..}
//    (k = number of cells, gs = cell size, d = simulation scale)
void FluidSystem::SetupGrid ()
{
	float border = 1.0f;
	Vector3DF gmin = m_Params.gridMin;
	Vector3DF gmax = m_Params.gridMax;
	float sim_scale = m_Params.sim_scale;
	float cell_size = m_Params.grid_size;

	float world_cellsize = cell_size / sim_scale;
	
	m_GridMin = gmin;
	m_GridMax = gmax;
	m_GridSize = m_GridMax;
	m_GridSize -= m_GridMin;	
	#if 0
		m_GridRes.Set ( 6, 6, 6 );				// Fixed grid res
	#else
		m_GridRes.x = (int) ceil ( m_GridSize.x / world_cellsize );		// Determine grid resolution
		m_GridRes.y = (int) ceil ( m_GridSize.y / world_cellsize );
		m_GridRes.z = (int) ceil ( m_GridSize.z / world_cellsize );
		m_GridSize.x = m_GridRes.x * cell_size / sim_scale;				// Adjust grid size to multiple of cell size
		m_GridSize.y = m_GridRes.y * cell_size / sim_scale;
		m_GridSize.z = m_GridRes.z * cell_size / sim_scale;
	#endif
	m_GridDelta = m_GridRes;		// delta = translate from world space to cell #
	m_GridDelta /= m_GridSize;
	
	m_GridTotal = (int)(m_GridRes.x * m_GridRes.y * m_GridRes.z);

	//m_Params.mem_grid = 12.0f * m_GridTotal;		// Grid memory used

	// Number of cells to search:
	// n = (2r / w) +1,  where n = 1D cell search count, r = search radius, w = world cell width
	//
	m_GridSrch = (int) (floor(2.0f*(m_Params.psmoothradius/sim_scale) / world_cellsize) + 1.0f);
	if ( m_GridSrch < 2 ) m_GridSrch = 2;
	m_GridAdjCnt = m_GridSrch * m_GridSrch * m_GridSrch ;			// 3D search count = n^3, e.g. 2x2x2=8, 3x3x3=27, 4x4x4=64

	if ( m_GridSrch > 6 ) {
		dbgprintf ( "ERROR: Neighbor search is n > 6. \n " );
		exit(-1);
	}

	int cell = 0;
	for (int y=0; y < m_GridSrch; y++ ) 
		for (int z=0; z < m_GridSrch; z++ ) 
			for (int x=0; x < m_GridSrch; x++ ) 
				m_GridAdj[cell++] = ( y*m_GridRes.z + z )*m_GridRes.x +  x ;			// -1 compensates for ndx 0=empty
}

int FluidSystem::getGridCell ( int p, Vector3DI& gc )
{
	return getGridCell ( m_Fluid.pnt(p)->pos, gc );
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

	// Reset all grid cells to empty		
	memset( m_Fluid.bufI(FGRIDCNT),		0,		m_Fluid.size(FGRIDCNT) );
	memset( m_Fluid.bufC(FGRIDOFF),		0,		m_Fluid.size(FGRIDOFF) );

	// Insert each particle into spatial grid
	Vector3DI gc;
	Fluid* p = m_Fluid.pnt(0);

	float poff = m_Params.psmoothradius / m_Params.sim_scale;

	int ns = (int) pow ( (float) m_GridAdjCnt, 1.0f/3.0f );
	register int xns, yns, zns;
	xns = m_GridRes.x - m_GridSrch;
	yns = m_GridRes.y - m_GridSrch;
	zns = m_GridRes.z - m_GridSrch;

	uint* m_Grid = m_Fluid.bufI(FGRID);
	uint* m_GridCnt = m_Fluid.bufI(FGRIDCNT);

	for ( int n=0; n < NumPoints(); n++ ) {
		gs = getGridCell ( p->pos, gc );
		if ( gc.x >= 1 && gc.x <= xns && gc.y >= 1 && gc.y <= yns && gc.z >= 1 && gc.z <= zns ) {
			p->gcell = gs;
			p->gndx = m_GridCnt[gs]++;			
		}  else {
			p->gcell = GRID_UNDEF;
		}
		p++;
	}
}
void FluidSystem::CountingSortFull ()
{
	Fluid* t;
	Fluid* p;

	// Transfer particles to temp 
	memcpy( m_Fluid.bufI(FFLUIDTEMP), m_Fluid.bufI(FFLUID), m_Fluid.size(FFLUID) );
	
	// Counting sort
	for (int i=0; i < NumPoints(); i++) {
		t = m_Fluid.ptemp(i);	
		
		if ( t->gcell != GRID_UNDEF ) {
			int sort_ndx = m_Fluid.bufI(FGRIDOFF)[ t->gcell ] + t->gndx;
			
			p = m_Fluid.pnt( sort_ndx );
			memcpy ( p, t, sizeof(Fluid) );

			m_Fluid.bufI(FGRID)[ sort_ndx ] = sort_ndx;
		}
	}
}


// Compute Pressures - Using spatial grid, and also create neighbor table
void FluidSystem::ComputePressures ()
{
	int i, j, cnt = 0;	
	float sum, dsq, c;
	float d = m_Params.sim_scale;
	float d2 = d*d;
	float radius = m_Params.psmoothradius / m_Params.psmoothradius;
	
	Vector3DF	dist, dst;
	int			cell;
	uint*		m_Grid = m_Fluid.bufI(FGRID);
	uint*		m_GridOff = m_Fluid.bufI(FGRIDOFF);
	uint*		m_GridCnt = m_Fluid.bufI(FGRIDCNT);

	
	// Get particle buffers
	Fluid* pi;
	Fluid* pj;


	for ( i=0; i < NumPoints(); i++ ) {

		pi = m_Fluid.pnt(i);
		pi->press = 1;

		uint gc = pi->gcell;
		if ( gc==GRID_UNDEF ) continue;
		gc -= (1*m_Params.gridRes.z + 1)*m_Params.gridRes.x + 1;

		sum = 0;

		for (int c=0; c < m_GridAdjCnt; c++) {

			cell = gc + m_GridAdj[c];			
			for (int cndx = m_GridOff[cell]; cndx < m_GridOff[cell] + m_GridCnt[cell]; cndx++) {
				j = m_Grid[cndx];

				pj = m_Fluid.pnt(j);
				dist = pi->pos - pj->pos;
				dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
				if ( dsq < m_Params.rd2 && dsq > 0 ) {
					dsq = (m_Params.rd2 - dsq) * m_Params.d2;
					sum += dsq * dsq * dsq;
				}
			}
		}			
		if (sum==0) sum = 1;
		pi->press = sum * m_Params.pmass * m_Poly6Kern; 		
	}	
}

// Compute Forces - Using spatial grid with saved neighbor table. Fastest.
void FluidSystem::ComputeForces ()
{
	Vector3DF force;
	register float pterm, vterm, dterm;
	int i, j;
	float c, d;
	float mR, visc;	

	d = m_Params.sim_scale;
	mR = m_Params.psmoothradius;
	visc = m_Params.pvisc;
	
	// Get particle buffers
	Fluid* pi = m_Fluid.pnt(0);
	Fluid* pj;
	
	Vector3DF dist;
	float	dsq;
	float	d2 = d*d;
	int		nadj = (m_GridRes.z + 1)*m_GridRes.x + 1;

	uint*	m_Grid = m_Fluid.bufI(FGRID);
	uint*	m_GridOff = m_Fluid.bufI(FGRIDOFF);
	uint*	m_GridCnt = m_Fluid.bufI(FGRIDCNT);
	
	int cnt, cell;
	float pressi, pressj;

	for ( i=0; i < NumPoints(); i++ ) {

		pi = m_Fluid.pnt(i);		

		uint gc = pi->gcell;
		if ( gc==GRID_UNDEF ) continue;
		gc -= (1*m_Params.gridRes.z + 1)*m_Params.gridRes.x + 1;

		force.Set(0,0,0);

		for (int c=0; c < m_GridAdjCnt; c++) {

			cell = gc + m_GridAdj[c];
			
			for (int cndx = m_GridOff[cell]; cndx < m_GridOff[cell] + m_GridCnt[cell]; cndx++) {
				j = m_Grid[cndx];

				pj = m_Fluid.pnt(j);
				dist = pi->pos - pj->pos;
				dsq = (dist.x*dist.x + dist.y*dist.y + dist.z*dist.z);
				if ( dsq < m_Params.rd2 && dsq > 0 ) {
					dsq = sqrt(dsq * m_Params.d2);
					pressi = (pi->press - m_Params.prest_dens ) * m_Params.pintstiff;
					pressj = (pj->press - m_Params.prest_dens ) * m_Params.pintstiff;
					pterm = m_Params.sim_scale * -0.5f * (m_Params.psmoothradius-dsq) * m_Params.spikykern * ( pressi + pressj ) / dsq;	
					force += ( dist * pterm + ( pj->veval - pi->veval ) * m_Params.vterm) * (m_Params.psmoothradius-dsq) / (pi->press * pj->press);
				}
			}
		}	
		pi->force = force;
	}
		
	//---- debugging
	/*Fluid* p;
	for (int n=0; n < 10; n++) {
		p = m_Fluid.pnt(n);		
		dbgprintf ( "%d: %f,%f,%f\n", n, p->force.x, p->force.y, p->force.z );
	}*/
}

void FluidSystem::Advance ()
{
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
	Fluid* p = m_Fluid.pnt(0);

	// Advance each particle
	for ( int n=0; n < NumPoints(); n++ ) {

		if ( p->gcell == GRID_UNDEF) continue;

		// Compute Acceleration		
		accel = p->force;
		accel *= m_Params.pmass;
	
		// Boundary Conditions
		// Y-axis walls
		diff = radius - ( p->pos.y - (bmin.y+ (p->pos.x-bmin.x)*m_Params.bound_slope) )*ss;
		if (diff > EPSILON ) {			
			norm.Set ( -m_Params.bound_slope, 1.0f - m_Params.bound_slope, 0 );
			adj = stiff * diff - damp * (float) norm.Dot ( p->veval );
			accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;
		}		
		diff = radius - ( bmax.y - p->pos.y )*ss;
		if (diff > EPSILON) {
			norm.Set ( 0, -1, 0 );
			adj = stiff * diff - damp * (float) norm.Dot ( p->veval );
			accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;
		}		
		
		// X-axis walls
		diff = radius - ( p->pos.x - (bmin.x + (sin(m_Time*m_Params.bound_wall_freq)+1)*0.5f * m_Params.bound_wall_force) )*ss;	
		//diff = 2 * radius - ( p->pos.x - min.x + (sin(m_Time*10.0)-1) * m_Param[FORCE_XMIN_SIN] )*ss;	
		if (diff > EPSILON ) {
			norm.Set ( 1.0, 0, 0 );
			adj = stiff * diff - damp * (float) norm.Dot ( p->veval ) ;
			accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;					
		}

		diff = radius - ( (bmax.x - (sin(m_Time*m_Params.bound_wall_freq)+1)*0.5f* m_Params.bound_wall_force) - p->pos.x )*ss;	
		if (diff > EPSILON) {
			norm.Set ( -1, 0, 0 );
			adj = stiff * diff - damp * (float) norm.Dot ( p->veval );
			accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;
		}

		// Z-axis walls
		diff = radius - ( p->pos.z - bmin.z )*ss;			
		if (diff > EPSILON) {
			norm.Set ( 0, 0, 1 );
			adj = stiff * diff - damp * (float) norm.Dot ( p->veval );
			accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;
		}
		diff = radius - ( bmax.z - p->pos.z )*ss;
		if (diff > EPSILON) {
			norm.Set ( 0, 0, -1 );
			adj = stiff * diff - damp * (float) norm.Dot ( p->veval );
			accel.x += adj * norm.x; accel.y += adj * norm.y; accel.z += adj * norm.z;
		}

		// Plane gravity
		accel += m_Params.gravity;

		// Point gravity
		if ( m_Params.grav_pos.x > 0 && m_Params.grav_amt > 0 ) {
			norm.x = ( p->pos.x - m_Params.grav_pos.x );
			norm.y = ( p->pos.y - m_Params.grav_pos.y );
			norm.z = ( p->pos.z - m_Params.grav_pos.z );
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
		speed = p->vel.x * p->vel.x + p->vel.y * p->vel.y + p->vel.z * p->vel.z;
		if ( speed > SL2 ) {
			speed = SL2;
			p->vel *= SL / sqrt(speed);
		}		

		// Leapfrog Integration ----------------------------
		vnext = accel * m_DT;									
		vnext += p->vel;						// v(t+1/2) = v(t-1/2) + a(t) dt

		p->veval = (p->vel + vnext) * 0.5f;		// v(t+1) = [v(t-1/2) + v(t+1/2)] * 0.5		used to compute forces later
		p->vel = vnext;		
		p->pos += vnext * (m_DT/ss);						// p(t+1) = p(t) + v(t+1/2) dt

		if ( isnan(p->pos.x) ) {
			bool stop=true;
		}

		p++;
	}
	//---- debugging
	/*for (int n=0; n < 10; n++) {
		p = m_Fluid.pnt(n);		
		dbgprintf ( "%d: %f,%f,%f\n", n, p->pos.x, p->pos.y, p->pos.z );
	}*/

}

void FluidSystem::Draw ( int frame, Camera3D* cam, float rad )
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

	Fluid* f = m_Fluid.pnt(0);

	glBindBuffer ( GL_ARRAY_BUFFER, m_Fluid.glid(FFLUID) );	
	//glBufferData ( GL_ARRAY_BUFFER, NumPoints()*sizeof(Vector3DF), m_Fluid.bufF3(FPOS), GL_DYNAMIC_DRAW );		
	glVertexAttribPointer ( 0, 3, GL_FLOAT, GL_FALSE, sizeof(Fluid), 0 );			
	glVertexAttribPointer ( 1, 1, GL_FLOAT, GL_FALSE, sizeof(Fluid), (void*) 24 );
	glVertexAttribPointer ( 2, 3, GL_FLOAT, GL_FALSE, sizeof(Fluid), (void*) 12 );
	checkGL ( "bind fluid" );	
		
	glDrawArrays ( GL_POINTS, 0, NumPoints() );
	checkGL ( "draw pnts" );
		
	selfEndDraw3D ();
	checkGL ( "end pnt shader" );

	PERF_POP();
}


void FluidSystem::SetupKernels ()
{
	float sr = m_Params.psmoothradius;
	m_Params.pdist = pow ( (float) m_Params.pmass / m_Params.prest_dens, 1.0f/3.0f );
	m_R2 = sr * sr;
	m_Poly6Kern = 315.0f / (64.0f * 3.141592f * pow( sr, 9.0f) );	// Wpoly6 kernel (denominator part) - 2003 Muller, p.4
	m_SpikyKern = -45.0f / (3.141592f * pow( sr, 6.0f) );			// Laplacian of viscocity (denominator): PI h^6
	m_LapKern = 45.0f / (3.141592f * pow( sr, 6.0f) );
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

	m_Time = 0.0f;							// Start at T=0
	m_DT = 0.003f;	

	m_Params.sim_scale =		0.006;			// unit size
	m_Params.pvisc =			0.10f;			// pascal-second (Pa.s) = 1 kg m^-1 s^-1  (see wikipedia page on viscosity)
	m_Params.prest_dens =		250.0f;			// kg / m^3	
	m_Params.pspacing =			0.0;			
	m_Params.pmass =			0.00020543f;	// kg
	m_Params.pradius =			0.02f;			// m
	m_Params.pdist =			0.0059f;			// m
	m_Params.psmoothradius =	0.02f;			// m 
	m_Params.pintstiff =		2.0f;	
	m_Params.bound_stiff =		15000.0f;		// boundary stiffness	
	m_Params.bound_damp =		400.0f;
	m_Params.bound_slope =		0.0f;			// ground slope
	m_Params.bound_friction =	1.0f;			// ground friction	
	m_Params.bound_wall_force =	0.0f;	
	m_Params.AL =				150.0f;			// accel limit, m / s^2
	m_Params.VL =				50.0f;			// vel limit, m / s	
	m_Params.grav_amt =			1.0f;	
	
	m_Params.grav_pos.Set ( 0, 0, 0 );
	m_Params.grav_dir.Set ( 0, -9.8f, 0 );

	m_Params.emit_pos.Set ( 0, 0, 0 );
	m_Params.emit_rate.Set ( 0, 0, 0 );
	m_Params.emit_ang.Set ( 0, 90, 1.0f );
	m_Params.emit_dang.Set ( 0, 0, 0 );

	// Default sim config	
	m_Params.gridSize = m_Params.psmoothradius * 2;
}

void FluidSystem::SetupExampleParams ()
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
		m_Params.bound_min.Set (   0,   0,   0 );
		m_Params.bound_max.Set ( 500, 100, 500 );
		m_Params.init_min.Set ( 100,  20,  20 );
		m_Params.init_max.Set ( 490,  95, 400 );
		m_Params.bound_wall_force = 120.0f;			
		m_Params.bound_wall_freq = 2.0f;
		m_Params.bound_slope = 0.1f;
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

void FluidSystem::SetupSpacing ()
{	
	m_Params.grid_size = 1.5 * m_Params.psmoothradius / m_Params.grid_density;

	if ( m_Params.pspacing == 0 ) {
		// Determine spacing from density
		m_Params.pdist = pow ( (float) m_Params.pmass / m_Params.prest_dens, 1/3.0f );	
		m_Params.pspacing = m_Params.pdist*0.87f / m_Params.sim_scale;
	} else {
		// Determine density from spacing
		m_Params.pdist = m_Params.pspacing * m_Params.sim_scale / 0.87f;
		m_Params.prest_dens = m_Params.pmass / pow ( (float) m_Params.pdist, 3.0f );
	}
	dbgprintf ( "Add Particles. Density: %f, Spacing: %f, PDist: %f\n", m_Params.prest_dens, m_Params.pspacing, m_Params.pdist );

	// Particle Boundaries
	m_Params.gridMin = m_Params.bound_min;		m_Params.gridMin -= float(2.0*(m_Params.grid_size / m_Params.sim_scale ));
	m_Params.gridMax = m_Params.bound_max;		m_Params.gridMax += float(2.0*(m_Params.grid_size / m_Params.sim_scale ));
}


int iDivUp (int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}
void computeNumBlocks (int numPnts, int minThreads, int &numBlocks, int &numThreads)
{
    numThreads = min( minThreads, numPnts );
    numBlocks = (numThreads==0) ? 1 : iDivUp ( numPnts, numThreads );
}
	
void FluidSystem::FluidSetupCUDA ( int num, int gsrch, Vector3DI res, Vector3DF size, Vector3DF delta, Vector3DF gmin, Vector3DF gmax, int total, int chk )
{	
	m_Params.pnum = num;	
	m_Params.gridRes = res;
	m_Params.gridSize = size;
	m_Params.gridDelta = delta;
	m_Params.gridMin = gmin;
	m_Params.gridMax = gmax;
	m_Params.gridTotal = total;
	m_Params.gridSrch = gsrch;
	m_Params.gridAdjCnt = gsrch*gsrch*gsrch;
	m_Params.gridScanMax = res;	
	m_Params.gridScanMax -= Vector3DI( m_Params.gridSrch, m_Params.gridSrch, m_Params.gridSrch );
	m_Params.chk = chk;

	// Build Adjacency Lookup
	int cell = 0;
	for (int y=0; y < gsrch; y++ ) 
		for (int z=0; z < gsrch; z++ ) 
			for (int x=0; x < gsrch; x++ ) 
				m_Params.gridAdj [ cell++]  = ( y * m_Params.gridRes.z+ z )*m_Params.gridRes.x +  x ;			
	
	// Compute number of blocks and threads
	int threadsPerBlock = 512;

    computeNumBlocks ( m_Params.pnum, threadsPerBlock, m_Params.numBlocks, m_Params.numThreads);				// particles
    computeNumBlocks ( m_Params.gridTotal, threadsPerBlock, m_Params.gridBlocks, m_Params.gridThreads);		// grid cell
    
	// Compute particle buffer & grid dimensions
    m_Params.szPnts = (m_Params.numBlocks  * m_Params.numThreads);     
    dbgprintf ( "CUDA Config: \n" );
	dbgprintf ( "  Pnts: %d, t:%dx%d=%d, Size:%d\n", m_Params.pnum, m_Params.numBlocks, m_Params.numThreads, m_Params.numBlocks*m_Params.numThreads, m_Params.szPnts);
    dbgprintf ( "  Grid: %d, t:%dx%d=%d, bufGrid:%d, Res: %dx%dx%d\n", m_Params.gridTotal, m_Params.gridBlocks, m_Params.gridThreads, m_Params.gridBlocks*m_Params.gridThreads, m_Params.szGrid, (int) m_Params.gridRes.x, (int) m_Params.gridRes.y, (int) m_Params.gridRes.z );		
	
	// Initialize random numbers
	int blk = int(num/16)+1;
	//randomInit<<< blk, 16 >>> ( rand(), gFluidBufs., num );
}

void FluidSystem::UpdateParamsCUDA ()
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

	// Transfer sim params to device
	cuCheck ( cuMemcpyHtoD ( cuFParams,	&m_Params,		sizeof(FParams) ), "FluidParamCUDA", "cuMemcpyHtoD", "cuFParams", mbDebug);
}

void FluidSystem::TransferToCUDA (int bufid)
{
	// Send particle buffers	
	//MapBuffer ( bufid, true );
	cuCheck( cuMemcpyHtoD ( m_Fluid.gpu(bufid),		m_Fluid.bufC(bufid),	m_Fluid.size(bufid) ),			"TransferToCUDA", "cuMemcpyHtoD", "FPOS", mbDebug);	
	//MapBuffer ( bufid, false );
}

void FluidSystem::TransferFromCUDA (int bufid)
{
	// Return particle buffers		
	cuCheck( cuMemcpyDtoH ( m_Fluid.bufC(bufid),	m_Fluid.gpu(bufid),		m_Fluid.size(bufid) ), "TransferFromCUDA", "cuMemcpyDtoH", "FPOS", mbDebug);	
}
void FluidSystem::TransferCopyCUDA (int src, int dest)
{
	cuCheck( cuMemcpyDtoD ( m_Fluid.gpu(dest),		m_Fluid.gpu(src),		m_Fluid.size(src) ), "TransferFromCUDA", "cuMemcpyDtoH", "FPOS", mbDebug);	
}

void FluidSystem::InsertParticlesCUDA ( uint* gcell, uint* gndx, uint* gcnt )
{
	cuCheck ( cuMemsetD8 ( m_Fluid.gpu(FGRIDCNT), 0,	m_Fluid.size(FGRIDCNT) ), "InsertParticlesCUDA", "cuMemsetD8", "FGRIDCNT", mbDebug );
	cuCheck ( cuMemsetD8 ( m_Fluid.gpu(FGRIDOFF), 0,	m_Fluid.size(FGRIDOFF) ), "InsertParticlesCUDA", "cuMemsetD8", "FGRIDOFF", mbDebug );

	void* args[1] = { &mNumPoints };
	cuCheck(cuLaunchKernel(m_Func[FUNC_INSERT], m_Params.numBlocks, 1, 1, m_Params.numThreads, 1, 1, 0, NULL, args, NULL),
		"InsertParticlesCUDA", "cuLaunch", "FUNC_INSERT", mbDebug);

	// Transfer data back if requested (for validation)
	if (gcell != 0x0) {
		//cuCheck( cuMemcpyDtoH ( gcell,	m_Fluid.gpu(FGCELL),		mNumPoints *sizeof(uint) ), "InsertParticlesCUDA", "cuMemcpyDtoH", "FGCELL", mbDebug );
		//cuCheck( cuMemcpyDtoH ( gndx,	m_Fluid.gpu(FGNDX),		mNumPoints *sizeof(uint) ), "InsertParticlesCUDA", "cuMemcpyDtoH", "FGNDX", mbDebug);
		cuCheck( cuMemcpyDtoH ( gcnt,	m_Fluid.gpu(FGRIDCNT),	m_GridTotal*sizeof(uint) ), "InsertParticlesCUDA", "cuMemcpyDtoH", "FGRIDCNT", mbDebug);
		cuCtxSynchronize ();
	}
}


void FluidSystem::PrefixSumCells ()
{
	int numCells = m_GridTotal;
		
	uint* mgcnt = m_Fluid.bufI(FGRIDCNT);
	uint* mgoff = m_Fluid.bufI(FGRIDOFF);

	int sum = 0;
	for (int n=0; n < numCells; n++) {
		mgoff[n] = sum;
		sum += mgcnt[n];
	}

	bool wait=true;
}

void FluidSystem::PrefixSumCellsCUDA ( uint* goff, int zero_offsets )
{
	// Prefix Sum - determine grid offsets
	int blockSize = SCAN_BLOCKSIZE << 1;
	int numElem1 = m_GridTotal;		
	int numElem2 = int ( numElem1 / blockSize ) + 1;
	int numElem3 = int ( numElem2 / blockSize ) + 1;
	int threads = SCAN_BLOCKSIZE;
	int zon=1;

	CUdeviceptr array1  = m_Fluid.gpu(FGRIDCNT);		// input
	CUdeviceptr scan1   = m_Fluid.gpu(FGRIDOFF);		// output
	CUdeviceptr array2  = m_Fluid.gpu(FAUXARRAY1);
	CUdeviceptr scan2   = m_Fluid.gpu(FAUXSCAN1);
	CUdeviceptr array3  = m_Fluid.gpu(FAUXARRAY2);
	CUdeviceptr scan3   = m_Fluid.gpu(FAUXSCAN2);

	if ( numElem1 > SCAN_BLOCKSIZE*xlong(SCAN_BLOCKSIZE)*SCAN_BLOCKSIZE) {
		dbgprintf ( "ERROR: Number of elements exceeds prefix sum max. Adjust SCAN_BLOCKSIZE.\n" );
	}

	void* argsA[5] = {&array1, &scan1, &array2, &numElem1, &zero_offsets }; // sum array1. output -> scan1, array2
	cuCheck ( cuLaunchKernel ( m_Func[FUNC_FPREFIXSUM], numElem2, 1, 1, threads, 1, 1, 0, NULL, argsA, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXSUM", mbDebug);

	void* argsB[5] = { &array2, &scan2, &array3, &numElem2, &zon }; // sum array2. output -> scan2, array3
	cuCheck ( cuLaunchKernel ( m_Func[FUNC_FPREFIXSUM], numElem3, 1, 1, threads, 1, 1, 0, NULL, argsB, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXSUM", mbDebug);

	if ( numElem3 > 1 ) {
		CUdeviceptr nptr = {0};
		void* argsC[5] = { &array3, &scan3, &nptr, &numElem3, &zon };	// sum array3. output -> scan3
		cuCheck ( cuLaunchKernel ( m_Func[FUNC_FPREFIXSUM], 1, 1, 1, threads, 1, 1, 0, NULL, argsC, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXFIXUP", mbDebug);

		void* argsD[3] = { &scan2, &scan3, &numElem2 };	// merge scan3 into scan2. output -> scan2
		cuCheck ( cuLaunchKernel ( m_Func[FUNC_FPREFIXFIXUP], numElem3, 1, 1, threads, 1, 1, 0, NULL, argsD, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXFIXUP", mbDebug);
	}

	void* argsE[3] = { &scan1, &scan2, &numElem1 };		// merge scan2 into scan1. output -> scan1
	cuCheck ( cuLaunchKernel ( m_Func[FUNC_FPREFIXFIXUP], numElem2, 1, 1, threads, 1, 1, 0, NULL, argsE, NULL ), "PrefixSumCellsCUDA", "cuLaunch", "FUNC_PREFIXFIXUP", mbDebug);

	// Transfer data back if requested
	if ( goff != 0x0 ) {
		cuCheck( cuMemcpyDtoH ( goff,		m_Fluid.gpu(FGRIDOFF),	numElem1*sizeof(int) ), "PrefixSumCellsCUDA", "cuMemcpyDtoH", "FGRIDOFF", mbDebug);
		cuCtxSynchronize ();
	}
}

void FluidSystem::CountingSortFullCUDA ( Vector3DF* ppos )
{
	// Transfer particle data to temp buffers
	//  (gpu-to-gpu copy, no sync needed)	
	TransferCopyCUDA ( FFLUID, FFLUIDTEMP );	

	void* args[1] = { &mNumPoints };
	cuCheck ( cuLaunchKernel ( m_Func[FUNC_COUNTING_SORT], m_Params.numBlocks, 1, 1, m_Params.numThreads, 1, 1, 0, NULL, args, NULL),
				"CountingSortFullCUDA", "cuLaunch", "FUNC_COUNTING_SORT", mbDebug );
}

void FluidSystem::ComputePressureCUDA ()
{
	cuCheck( cuMemcpyDtoH( &m_Params,	cuFParams,	sizeof(FParams)),	"AllocateParticles", "cuMemcpyHtoD", "cuFParams", mbDebug);	

	void* args[1] = { &mNumPoints };
	cuCheck ( cuLaunchKernel ( m_Func[FUNC_COMPUTE_PRESS],  m_Params.numBlocks, 1, 1, m_Params.numThreads, 1, 1, 0, NULL, args, NULL), "ComputePressureCUDA", "cuLaunch", "FUNC_COMPUTE_PRESS", mbDebug);
}

void FluidSystem::ComputeForceCUDA ()
{
	void* args[1] = { &mNumPoints };
	cuCheck ( cuLaunchKernel ( m_Func[FUNC_COMPUTE_FORCE],  m_Params.numBlocks, 1, 1, m_Params.numThreads, 1, 1, 0, NULL, args, NULL), "ComputeForceCUDA", "cuLaunch", "FUNC_COMPUTE_FORCE", mbDebug);
}
void FluidSystem::AdvanceCUDA ( float tm, float dt, float ss )
{
	void* args[4] = { &tm, &dt, &ss, &mNumPoints };
	cuCheck ( cuLaunchKernel ( m_Func[FUNC_ADVANCE],  m_Params.numBlocks, 1, 1, m_Params.numThreads, 1, 1, 0, NULL, args, NULL), "AdvanceCUDA", "cuLaunch", "FUNC_ADVANCE", mbDebug);
}
void FluidSystem::EmitParticlesCUDA ( float tm, int cnt )
{
	void* args[3] = { &tm, &cnt, &mNumPoints };
	cuCheck ( cuLaunchKernel ( m_Func[FUNC_EMIT],  m_Params.numBlocks, 1, 1, m_Params.numThreads, 1, 1, 0, NULL, args, NULL), "EmitParticlesCUDA", "cuLaunch", "FUNC_EMIT", mbDebug);
}
