//-----------------------------------------------------------------------------
// FLUIDS v.5 - SPH Fluid Simulator for CPU and GPU
// Copyright (C) 2012-2013, 2021. Rama Hoetzlein, http://fluids3.com
//-----------------------------------------------------------------------------

// Sample utils
#include "main.h"			// window system 
#include "nv_gui.h"			// gui system
#include <GL/glew.h>
#include "timex.h"

#include <algorithm>
#include "particles.h"

#define MOVE_GOAL_XY	1
#define MOVE_GOAL_XZ	2

class Sample : public Application {
public:
	virtual bool init();
	virtual void display();
	virtual void reshape(int w, int h);	
	virtual void mousewheel(int delta);
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
  float     m_elapsed;

	Camera3D*		m_cam;	
  Vector3DF   m_fps;

	Particles		m_psys;

	CUcontext		m_ctx;
	CUdevice		m_dev; 
};

// Application object
Sample sample_obj;			


void Sample::Reset ()
{
    dbgprintf ( "Starting particles: \n" );
    m_psys.Restart (4000000);
}

bool Sample::init ()
{	
  PERF_INIT(64, true, false, false, 0, "");

	int w = getWidth(), h = getHeight();			// window width & height
	m_frame = 0;
	m_pause = false;

  addSearchPath ( ASSET_PATH );

	// Initialize app
	init2D ( "arial" );
  setText ( 16, 1 );
	setview2D ( w, h );	
	glViewport ( 0, 0, w, h );

	m_cam = new Camera3D;
	m_cam->setFov ( 80 );
	m_cam->setNearFar ( 1, 10000 );
	m_cam->setOrbit ( Vector3DF(40,50,0), Vector3DF(250,0,250), 1000, 70 );
	m_adjust = -1;

	// Start CUDA
	cuStart ( DEV_FIRST, 0, m_dev, m_ctx, 0, true );
	
	// Initialize fluid 
	m_psys.SetDebug ( false );			// true = enable gpu sync after every cuCheck 

	m_psys.Initialize ();
	
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
      
    PERF_PUSH ( "sim" );
    
    m_psys.Run ();

    m_elapsed = PERF_POP ();

    m_fps.x += 1000.0 / m_elapsed;
    m_fps.y++;
    m_fps.z = m_fps.x / m_fps.y;
  }
	


	// Draw fluid
	m_psys.Draw ( m_frame, m_cam, 1.0f );
	
	// Sketch a grid
	start3D( m_cam );	
	for (int i = 0; i <= 500; i += 50) {
		drawLine3D(float(i),-0.01f,  0.f, float(i), -0.01f, 500.f, .2f, .2f, .2f, 1.f);
		drawLine3D( 0.f,	-0.01f, float(i), 500.f, -0.01f, float(i), .2f, .2f, .2f, 1.f);
	}	
	end3D();

  start2D ();
    drawText ( 10, 10, "Space = Pause sim", 1, 1, 1, 1 );
    drawText ( 10, 30, "R key = Reset sim", 1, 1, 1, 1);
    char msg[512];
    sprintf (msg, "# Particles:  %d\n", m_psys.NumPoints() );
    drawText ( 10, 50, msg, 1, 1, 1, 1);
    sprintf (msg, "Compute: %4.0f FPS (%4.2f msec)\n", m_fps.z, 1000.0/m_fps.z);
    drawText ( 10, 70, msg, 1, 1, 1, 1 );
  end2D();

	// Use nvGui to draw in 2D/3D
	draw3D ();										// Render the 3D drawing groups
	drawGui (0);									// Render the GUI
	draw2D (); 

	appPostRedisplay();								// Post redisplay since simulation is continuous

	m_frame++;
}

void Sample::mousewheel(int delta)
{
	// Adjust zoom
	float zoomamt = 1.0;
	float dist = m_cam->getOrbitDist();
	float dolly = m_cam->getDolly();
	float zoom = (dist - dolly) * 0.001f;
	dist -= delta * zoom * zoomamt;
	
	m_cam->setOrbit(m_cam->getAng(), m_cam->getToPos(), dist, dolly);		
}


void Sample::motion( AppEnum button, int x, int y, int dx, int dy) 
{
	// Get camera for GVDB Scene
	bool shift = (getMods() & KMOD_SHIFT);
	bool alt = (getMods() & KMOD_ALT);	
	float fine = 0.5;

	switch ( mouse_down ) {	
	case AppEnum::BUTTON_LEFT: {
		if (alt) {
			// Adjust camera orbit 
			Vector3DF angs = m_cam->getAng();
			angs.x += dx*0.2f*fine;
			angs.y -= dy*0.2f*fine;				
			m_cam->setOrbit ( angs, m_cam->getToPos(), m_cam->getOrbitDist(), m_cam->getDolly() );		
		}				
	
		appPostRedisplay();	// Update display
		} break;
	
	case AppEnum::BUTTON_MIDDLE: {
		// Adjust target pos		
		m_cam->moveRelative ( float(dx) * fine*m_cam->getOrbitDist()/1000, float(-dy) * fine*m_cam->getOrbitDist()/1000, 0 );	
		appPostRedisplay();	// Update display
		} break;
	
	case AppEnum::BUTTON_RIGHT: {	
		
		// Adjust camera orbit 
		Vector3DF angs = m_cam->getAng();
		angs.x += dx*0.2f*fine;
		angs.y -= dy*0.2f*fine;				
		m_cam->setOrbit ( angs, m_cam->getToPos(), m_cam->getOrbitDist(), m_cam->getDolly() );		

		/*// Adjust dist
		float dist = m_cam->getOrbitDist();
		dist -= dy*fine;
		m_cam->setOrbit ( m_cam->getAng(), m_cam->getToPos(), dist, m_cam->getDolly() );		*/

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
	
	appStart ( "Fluids v5.0", "Fluids v5.0", w, h, 4, 0, 1 );	
}

