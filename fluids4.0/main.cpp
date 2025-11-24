//-----------------------------------------------------------------------------
// FLUIDS v.3.1 - SPH Fluid Simulator for CPU and GPU
// Copyright (C) 2012-2013, 2021. Rama Hoetzlein, http://fluids3.com
//-----------------------------------------------------------------------------

// Sample utils
#include "main.h"			// window system 
#include "nv_gui.h"			// gui system
#include <GL/glew.h>
#include "timex.h"

#include <algorithm>
//#include "fluid_defs.h"
#include "fluid_system.h"

#define MOVE_GOAL_XY	1
#define MOVE_GOAL_XZ	2

class Sample : public Application {
public:
	virtual bool init();
	virtual void display();
	virtual void reshape(int w, int h);	
	virtual void keyboardchar(unsigned char key, int mods, int x, int y);
	virtual void mouse (AppEnum button, AppEnum state, int mods, int x, int y);
	virtual void motion( AppEnum button, int x, int y, int dx, int dy);
	virtual void startup ();

	void			Reset ();

	int				mouse_down;
	int				mouse_action;
	int				m_adjust;
	int				m_frame;
	bool			m_pause;

	Camera3D*		m_cam;	
  Vector3DF m_fps;

	FluidSystem		m_fluid;

	CUcontext		m_ctx;
	CUdevice		m_dev; 
};

// Application object
Sample sample_obj;			

#define DEV_FIRST		-1
#define DEV_CURRENT		-2
#define DEV_EXISTING	-3

extern bool cuCheck (CUresult launch_stat, char* method, char* apicall, char* arg, bool bDebug);		// comes from fluid_system.cpp

void cudaStart ( int devsel, CUcontext ctxsel, CUdevice& dev, CUcontext& ctx, CUstream* strm, bool verbose)
{
	int version = 0;
    char name[128];
    
	int cnt = 0;
	CUdevice dev_id;	
	cuInit(0);

	//--- List devices
	cuDeviceGetCount ( &cnt );

	if (cnt == 0) {
		dbgprintf("ERROR: No CUDA devices found.\n");
		dev = NULL; ctx = NULL;		
		exit(-1);
		return;
	}	
	if (verbose) dbgprintf ( "  Device List:\n" );
	for (int n=0; n < cnt; n++ ) {
		cuDeviceGet(&dev_id, n);
		cuDeviceGetName ( name, 128, dev_id);

		int w1,h1,d1;
		cuDeviceGetAttribute ( &w1, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH, dev_id ) ;		
		cuDeviceGetAttribute ( &h1, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT, dev_id ) ;		
		cuDeviceGetAttribute ( &d1, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH, dev_id ) ;		

		if (verbose) dbgprintf ( "   %d. %s  maxtex (%d,%d,%d)\n", n, name, w1,h1,d1);
	}			

	if (devsel == DEV_CURRENT) {
		//--- Get currently active context 
		cuCheck( cuCtxGetCurrent(&ctx), "", "cuCtxGetCurrent", "", false );
		cuCheck( cuCtxGetDevice(&dev), "", "cuCtxGetDevice", "", false );
	}
	if (devsel == DEV_EXISTING) {
		//--- Use existing context passed in
		ctx = ctxsel;	
		cuCheck(cuCtxSetCurrent(ctx), "", "cuCtxSetCurrent", "", false );
		cuCheck(cuCtxGetDevice(&dev), "", "cuCtxGetDevice", "", false );
	}	
	if (devsel == DEV_FIRST || devsel >= cnt ) devsel = 0;
	if (devsel >= cnt) devsel = 0;				// Fallback to dev 0 if addition GPUs not found

	if (devsel >= 0) {		
		//--- Create new context with Driver API 
		cuCheck(cuDeviceGet( &dev, devsel),  "", "cuDeviceGet", "", false );
		cuCheck(cuCtxCreate( &ctx, CU_CTX_SCHED_AUTO, dev), "", "cuCtxCreate", "", false );
	}
	cuDeviceGetName(name, 128, dev);
	if (verbose) dbgprintf("   Using Device: %d, %s, Context: %p\n", (int) dev, name, (void*) ctx);
	
	cuCtxSetCurrent( NULL );
	cuCtxSetCurrent( ctx );
}

Vector3DF cudaGetMemUsage ()
{
	Vector3DF mem;
	size_t free, total;	
	cuMemGetInfo ( &free, &total );
	free /= (1024.0f*1024.0f);		// MB
	total /= (1024.0f*1024.0f);
	mem.x = total - free;	// used
	mem.y = free;
	mem.z = total;
	return mem;
}


void Sample::Reset ()
{

}

bool Sample::init ()
{	
	int w = getWidth(), h = getHeight();			// window width & height
	m_frame = 0;
	m_pause = false;

	// Initialize app
	init2D ( "arial" );
	setview2D ( w, h );	
	glViewport ( 0, 0, w, h );

	PERF_INIT ( 64, true, false, false, 0, "" );		// profiling

	m_cam = new Camera3D;
	m_cam->setFov ( 80 );
	m_cam->setNearFar ( 1, 10000 );
	m_cam->setOrbit ( Vector3DF(40,50,0), Vector3DF(200,0,200), 1200, 70 );
	m_adjust = -1;

	// Start CUDA
	cudaStart ( DEV_FIRST, 0, m_dev, m_ctx, 0, true );
	
	// Initialize fluid 
	m_fluid.SetDebug ( false );		// enable sync after every cuCheck

	m_fluid.Initialize ();
	
	m_fluid.Start ( 4000000 );			// number of particles here

	m_fluid.SetupRender ();

	// Reset fluid
	Reset ();	

	appSetVSync(false);		// no vsync

	return true;
}


void Sample::display()
{
	int w = getWidth(), h = getHeight();
	Vector3DF a,b,c,p;

	clearGL();

	// Run fluid simulation!
	if ( !m_pause ) {

    PERF_PUSH("run");

    m_fluid.Run ();

    float e = PERF_POP();
    
    m_fps.x += 1000.0 / e;
    m_fps.y++;
    m_fps.z = m_fps.x / m_fps.y;

  }
  dbgprintf ( "%f FPS (%f msec), %d Particles\n", m_fps.z, 1000.0/m_fps.z, m_fluid.NumPoints() );

	
	// Draw fluid
	m_fluid.Draw ( m_frame, m_cam, 1.0f );
	
	// Sketch a grid
	start3D( m_cam );	
	for (int i = 0; i <= 500; i += 50) {
		drawLine3D(float(i),-0.01f,  0.f, float(i), -0.01f, 500.f, .2f, .2f, .2f, 1.f);
		drawLine3D( 0.f,	-0.01f, float(i), 500.f, -0.01f, float(i), .2f, .2f, .2f, 1.f);
	}	
	end3D();

	// Use nvGui to draw in 2D/3D
	draw3D ();										// Render the 3D drawing groups
	drawGui (0);									// Render the GUI
	draw2D (); 

	appPostRedisplay();								// Post redisplay since simulation is continuous

	m_frame++;
}

void Sample::motion( AppEnum button, int x, int y, int dx, int dy) 
{
	// Get camera for GVDB Scene
	bool shift = (getMods() & KMOD_SHIFT);		// Shift-key to modify light

	float fine = 0.5;

	switch ( mouse_down ) {	
	case AppEnum::BUTTON_LEFT: {

		// Adjust camera orbit 
		Vector3DF angs = m_cam->getAng();
		angs.x += dx*0.2f*fine;
		angs.y -= dy*0.2f*fine;				
		m_cam->setOrbit ( angs, m_cam->getToPos(), m_cam->getOrbitDist(), m_cam->getDolly() );				
	
		appPostRedisplay();	// Update display
		} break;
	
	case AppEnum::BUTTON_MIDDLE: {
		// Adjust target pos		
		m_cam->moveRelative ( float(dx) * fine*m_cam->getOrbitDist()/1000, float(-dy) * fine*m_cam->getOrbitDist()/1000, 0 );	
		appPostRedisplay();	// Update display
		} break;
	
	case AppEnum::BUTTON_RIGHT: {	
		
		// Adjust dist
		float dist = m_cam->getOrbitDist();
		dist -= dy*fine;
		m_cam->setOrbit ( m_cam->getAng(), m_cam->getToPos(), dist, m_cam->getDolly() );		

		appPostRedisplay();	// Update display
		} break;
	}
}

void Sample::mouse ( AppEnum button, AppEnum state, int mods, int x, int y)
{
	if ( guiHandler ( button, state, x, y ) ) return;	
	
	mouse_down = (state == AppEnum::BUTTON_PRESS) ? button : -1;		// Track when we are in a mouse drag

	mouse_action = 0;

}

void Sample::keyboardchar(unsigned char key, int mods, int x, int y)
{
	switch ( key ) {
	case ' ':			m_pause = !m_pause; break;
	case 'c': case 'C':	m_adjust = -1;	break;
	case 'r':		
		Reset ();
		break;
	case '0':		 	m_adjust = 0;	break;
	case '1':		 	m_adjust = 1;	break;
	case '2':		 	m_adjust = 2;	break;
	case '3':		 	m_adjust = 3;	break;
	};
}


void Sample::reshape (int w, int h)
{
	glViewport ( 0, 0, w, h );
	appPostRedisplay();
}

void Sample::startup () 
{
	int w=1270, h=800;
	
	appStart ( "Fluids v4.0", "Fluids v4.0", w, h, 4, 0, 1 );	
}

