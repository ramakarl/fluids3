
#include <time.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <windows.h>

#include "app_opengl.h"
#include "timex.h"
#include "vec.h"

#include "fluid_defs.h"

#ifdef BUILD_CUDA
	#include "fluid_system_host.h"	
#endif
#include "fluid_system.h"

#include "main.h"
#include "nv_gui.h"

// Different things we can move around
#define MODE_CAM		0
#define MODE_CAM_TO		1
#define MODE_OBJ		2
#define MODE_OBJPOS		3
#define MODE_OBJGRP		4
#define MODE_LIGHTPOS	5

#define MODE_DOF		6

class Sample : public Application {
public:
	virtual bool init();
	virtual void display();
	virtual void reshape(int w, int h);	
	virtual void keyboardchar(unsigned char key, int mods, int x, int y);
	virtual void mouse (AppEnum button, AppEnum state, int mods, int x, int y);
	virtual void motion( AppEnum button, int x, int y, int dx, int dy);
	virtual void startup ();

	void		drawScene ( float* viewmat, bool bShade );
	void		drawInfo ();
	void		UpdateEmit ();

	int			frame;
	bool		bTiming = true;
	bool		bRec = false;
	float		frameTime;
	int			frameFPS;
	int			mFrame = 0;
	float		gravity = 9.8;
	float		visc = 0.35;
	float		restdens = 600.0;
	Vector3DF	obj_from, obj_angs, obj_dang;
	Vector4DF	light[2], light_to[2];				// Light stuff
	float		light_fov;	

	int			psys_count = 16384;
	int			psys_rate = 0;							// Particle stuff
	int			psys_freq = 1;
	int			psys_playback;

	// Globals
	FluidSystem		psys;
	Camera3D		cam;

  Vector3DF m_fps;

	int			mouse_down;
	int			mouse_action;
	bool		bHelp = true;					// Toggles
	int			iShade = 0;						// Shading mode (default = no shadows)
	int			iClrMode = 0;
	bool		bPause = false;

	// View matricies
	float	model_matrix[16];					// Model matrix (M)


	int		last_x = -1, last_y = -1;		// mouse vars
	int		mode = 0;
	int		dragging = 0;
	int		psel;

	GLuint	screen_id;
	GLuint	depth_id;
	GLuint  screenBufferObject;
	GLuint  depthBufferObject;
	GLuint  envid;
};

Sample sample_obj;


void Sample::drawScene ( float* viewmat, bool bShade )
{
	if ( iShade <= 1 && bShade ) {		
	
		glEnable ( GL_LIGHTING );
		glEnable ( GL_LIGHT0 );
		glDisable ( GL_COLOR_MATERIAL );

		Vector4DF amb, diff, spec;
		float shininess = 5.0;
		
		glColor3f ( 1, 1, 1 );
		glLoadIdentity ();
		glLoadMatrixf ( viewmat );

		float pos[4];
		pos[0] = light[0].x;
		pos[1] = light[0].y;
		pos[2] = light[0].z;
		pos[3] = 1;
		amb.Set ( 0,0,0,1 ); diff.Set ( 1,1,1,1 ); spec.Set(1,1,1,1);
		glLightfv ( GL_LIGHT0, GL_POSITION, (float*) &pos[0]);
		glLightfv ( GL_LIGHT0, GL_AMBIENT, (float*) &amb.x );
		glLightfv ( GL_LIGHT0, GL_DIFFUSE, (float*) &diff.x );
		glLightfv ( GL_LIGHT0, GL_SPECULAR, (float*) &spec.x ); 
		
		amb.Set ( 0,0,0,1 ); diff.Set ( .3, .3, .3, 1); spec.Set (.1,.1,.1,1);
		glMaterialfv (GL_FRONT_AND_BACK, GL_AMBIENT, (float*) &amb.x );
		glMaterialfv (GL_FRONT_AND_BACK, GL_DIFFUSE, (float*) &diff.x);
		glMaterialfv (GL_FRONT_AND_BACK, GL_SPECULAR, (float*) &spec.x);
		glMaterialfv (GL_FRONT_AND_BACK, GL_SHININESS, (float*) &shininess);
		

		//glColorMaterial ( GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE );
		
		glLoadMatrixf ( viewmat );
		
		glBegin ( GL_QUADS );
		glNormal3f ( 0, 1, 0.001  );
		for (float x=-1000; x <= 1000; x += 100.0 ) {
			for (float y=-1000; y <= 1000; y += 100.0 ) {
				glVertex3f ( x, 0.0, y );
				glVertex3f ( x+100, 0.0 , y );
				glVertex3f ( x+100, 0.0, y+100 );
				glVertex3f ( x, 0.0, y+100 );
			}
		}
		glEnd ();
		
		glColor3f ( 0.1, 0.1, 0.2 );
		glDisable ( GL_LIGHTING );
		glBegin ( GL_LINES );		
		for (float n=-100; n <= 100; n += 10.0 ) {
			glVertex3f ( -100, 0.1, n );
			glVertex3f ( 100, 0.1, n );
			glVertex3f ( n, 0.1, -100 );
			glVertex3f ( n, 0.1, 100 );
		}
		glVertex3f ( light[0].x, light[0].y, 0 );
		glVertex3f ( light[0].x, light[0].y, light[0].z );
		glEnd ();

		psys.Draw ( &cam, 0.8 );				// Draw particles		

	} else {
		glDisable ( GL_LIGHTING );
		psys.Draw ( &cam, 0.55 );			// Draw particles
	}
}

void Sample::drawInfo ()
{
	char disp[1024];
	

	start2D ();

	glColor4f ( 1.0, 1.0, 0.0, 1.0 );	
	
  // average fps
  m_fps.x += frameFPS;
  m_fps.y ++;
  m_fps.z = m_fps.x / m_fps.y;

	int y = 0;
	sprintf ( disp, "# Particles:         %d", psys.NumPoints() ) ; 	drawText ( 20, y+20, disp , 1,1,1,1);
	sprintf ( disp,	"FPS:                 %4.2f", m_fps.z );			drawText ( 20, y+40,  disp, 1,1,1,1 );		
	sprintf ( disp,	"Time (ms):           %f", frameTime );			drawText ( 20, y+60,  disp, 1,1,1,1 );			
	sprintf ( disp,	"Grid Resolution:     %d x %d x %d (%d)", (int) psys.GetGridRes().x, (int) psys.GetGridRes().y, (int) psys.GetGridRes().z, psys.getGridTotal() );		drawText ( 20, y+80,  disp, 1,1,1,1 );
	int nsrch = pow ( psys.getSearchCnt(), 1/3.0 );
	sprintf ( disp,	"Performance:         %d particles/sec", (int) ((psys.NumPoints()*1000.0)/frameTime) );			drawText ( 20, y+100,  disp, 1,1,1,1 );
	sprintf ( disp,	"Particle Memory:     %.4f MB", (float) psys.GetParam(PSTAT_PMEM)/1000000.0f );		drawText ( 20, y+120,  disp, 1,1,1,1 );
	sprintf ( disp,	"Grid Memory:         %.4f MB", (float) psys.GetParam(PSTAT_GMEM)/1000000.0f );		drawText ( 20, y+140,  disp, 1,1,1,1 );

	end2D();
}


void Sample::display () 
{
	PERF_START ();

	PERF_PUSH ( "FRAME" );	

	// Check for slider interaction
	if ( guiChanged(2) ) {	
		psys.SetParam ( PNUM, psys_count, 4, 40000000 );
		psys.Setup ( false );
	}
	if ( guiChanged(3) ) 	psys.SetVec ( PPLANE_GRAV_DIR, Vector3DF(0.f,-gravity,0.f) );
	if ( guiChanged(4) ) 	psys.SetParam ( PVISC, visc );
	if ( guiChanged(5) ) 	psys.SetParam ( PRESTDENSITY, restdens );
	

	// Do simulation!
	if ( !bPause ) psys.Run (window_width, window_height);
	

	PERF_PUSH ( "Render" );	

	frame++;
	//measureFPS ();

	glEnable ( GL_DEPTH_TEST );

	// Clear frame buffer
	if ( iShade<=1 ) 	glClearColor( 0.05, 0.05, 0.05, 1.0 );
	else				glClearColor ( 0, 0, 0, 0 );
	glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	glDisable ( GL_CULL_FACE );
	glShadeModel ( GL_SMOOTH );

	// Draw scene
	drawScene ( cam.getViewMatrix().GetDataF() , true );

	// Draw 2D overlay
	PERF_PUSH ( "DrawInfo" );
	drawInfo ();
	PERF_POP ();
 
	draw3D();
	//drawGui(0);
	draw2D();

	PERF_POP ();

	if ( !bPause ) {
		frameTime = PERF_STOP ();	
		frameFPS = int(1000.0 / frameTime);
	}

	appPostRedisplay();
}

void Sample::reshape ( int width, int height ) 
{
  // set window height and width
  window_width  = (float) width;
  window_height = (float) height;
  glViewport( 0, 0, width, height );    
  setview2D ( width, height );
}

void Sample::UpdateEmit ()
{	
	obj_from = psys.GetVec ( PEMIT_POS );
	obj_angs = psys.GetVec ( PEMIT_ANG );
	obj_dang = psys.GetVec ( PEMIT_RATE );
}


void Sample::keyboardchar ( unsigned char key, int mods, int x, int y)
{
	Vector3DF fp = cam.getPos ();
	Vector3DF tp = cam.getToPos ();

	switch( key ) {
	case 'R': case 'r': {
		psys.StartRecord ();
		} break;
	//case 'P': case 'p': 	
		/*switch ( psys.getMode() ) {
		case MODE_RECORD: case MODE_SIM:	psys_playback = psys.getLastRecording();	break;
		case MODE_PLAYBACK:					psys_playback--; if ( psys_playback < 0 ) psys_playback = psys.getLastRecording();	break;
		};
		psys.StartPlayback ( psys_playback );
		} break;*/
	case 'M': case 'm': {
		psys_count *= 2;
		psys.SetParam ( PNUM, psys_count, 4, 40000000 );
		psys.Setup ( false );
		} break;
	case 'N': case 'n': {
		psys_count /= 2;
		psys.SetParam ( PNUM, psys_count, 4, 40000000 );
		psys.Setup ( false );
		} break;
	case '0':
		UpdateEmit ();
		psys_freq++;	
		psys.SetVec ( PEMIT_RATE, Vector3DF(psys_freq, psys_rate, 0) );
		break;  
	case '9':
		UpdateEmit ();
		psys_freq--;  if ( psys_freq < 0 ) psys_freq = 0;
		psys.SetVec ( PEMIT_RATE, Vector3DF(psys_freq, psys_rate, 0) );
		break;
	case '.': case '>':
		UpdateEmit ();
		if ( ++psys_rate > 100 ) psys_rate = 100;
		psys.SetVec ( PEMIT_RATE, Vector3DF(psys_freq, psys_rate, 0) );
		break;
	case ',': case '<':
		UpdateEmit ();
		if ( --psys_rate < 0 ) psys_rate = 0;
		psys.SetVec ( PEMIT_RATE, Vector3DF(psys_freq, psys_rate, 0) );
		break;
	
	case 'f': case 'F':		psys.IncParam ( PMODE, -1, 1, 8 );		psys.Setup (false); break;
	case 'g': case 'G':		psys.IncParam ( PMODE, 1, 1, 8 );		psys.Setup (false); break;
	case ' ':				bPause = !bPause;	break;		// pause

	case '1':				psys.IncParam ( PDRAWGRID, 1, 0, 1 );		break;
	case '2':				psys.IncParam ( PDRAWTEXT, 1, 0, 1 );		break;

	case 'C':	mode = MODE_CAM_TO;	break;
	case 'c': 	mode = MODE_CAM;	break; 
	case 'h': case 'H':	bHelp = !bHelp; break;
	case 'i': case 'I':	
		UpdateEmit ();
		mode = MODE_OBJPOS;	
		break;
	case 'o': case 'O':	
		UpdateEmit ();
		mode = MODE_OBJ;
		break;  
	case 'x': case 'X':
		if ( ++iClrMode > 2) iClrMode = 0;
		psys.SetParam ( PCLR_MODE, iClrMode );
		break;
	case 'l': case 'L':	mode = MODE_LIGHTPOS;	break;
	case 'j': case 'J': {
		int d = psys.GetParam ( PDRAWMODE ) + 1;
		if ( d > 2 ) d = 0;
		psys.SetParam ( PDRAWMODE, d );
		} break;	
	case 'k': case 'K':	if ( ++iShade > 3 ) iShade = 0;		break;

	case 'a': case 'A':		cam.setToPos( tp.x - 1, tp.y, tp.z ); break;
	case 'd': case 'D':		cam.setToPos( tp.x + 1, tp.y, tp.z ); break;
	case 'w': case 'W':		cam.setToPos( tp.x, tp.y - 1, tp.z ); break;
	case 's': case 'S':		cam.setToPos( tp.x, tp.y + 1, tp.z ); break;
	case 'q': case 'Q':		cam.setToPos( tp.x, tp.y, tp.z + 1 ); break;
	case 'z': case 'Z':		cam.setToPos( tp.x, tp.y, tp.z - 1 ); break;

		
	case 27:			    exit( 0 ); break;
	
	case '`':				psys.Toggle ( PCAPTURE ); break;
	
	case 't': case 'T':		psys.Setup (true); break;  

	case '-':  case '_':
		psys.IncParam ( PGRID_DENSITY, -0.2, 1, 10 );	
		psys.Setup (true);
		break;
	case '+': case '=':
		psys.IncParam ( PGRID_DENSITY, 0.2, 1, 10 );	
		psys.Setup (true);
		break;
	case '[':
		psys.IncParam ( PEXAMPLE, -1, 0, 10 );
		psys.Setup (true);
		UpdateEmit ();
		break;
	case ']':
		psys.IncParam ( PEXAMPLE, +1, 0, 10 );
		psys.Setup (true);
		UpdateEmit ();
		break;  
	default:
	break;
  }
}


void Sample::mouse (AppEnum button, AppEnum state, int mods, int x, int y)
{
	mouse_down = (state == AppEnum::BUTTON_PRESS) ? button : -1;		// Track when we are in a mouse drag

	  if( state == GLUT_DOWN ) {

			// Handle 2D gui interaction first
			if ( guiMouseDown ( x, y ) ) return;

			if ( button == GLUT_LEFT_BUTTON )		dragging = AppEnum::BUTTON_LEFT;
			else if ( button == GLUT_RIGHT_BUTTON ) dragging = AppEnum::BUTTON_RIGHT;	
			last_x = x;
			last_y = y;	

	  } else if ( state==GLUT_UP ) {

			dragging = 0;
	  }
}

void Sample::motion( AppEnum button, int x, int y, int dx, int dy) 
{

	// Handle GUI interaction in nvGui by calling guiMouseDrag
	if ( guiMouseDrag ( x, y ) ) return;

	// Get camera for GVDB Scene
	bool shift = (getMods() & KMOD_SHIFT);		// Shift-key to modify light

	float fine = 0.5;

	switch ( mouse_down ) {	
	case AppEnum::BUTTON_LEFT: {

		// Adjust camera orbit 
		Vector3DF angs = cam.getAng();
		angs.x += dx*0.2f*fine;
		angs.y -= dy*0.2f*fine;				
		cam.setOrbit ( angs, cam.getToPos(), cam.getOrbitDist(), cam.getDolly() );				
	
		appPostRedisplay();	// Update display
		} break;
	
	case AppEnum::BUTTON_MIDDLE: {
		// Adjust target pos		
		cam.moveRelative ( float(dx) * fine*cam.getOrbitDist()/1000, float(-dy) * fine*cam.getOrbitDist()/1000, 0 );	
		appPostRedisplay();	// Update display
		} break;
	
	case AppEnum::BUTTON_RIGHT: {	
		
		// Adjust dist
		float dist = cam.getOrbitDist();
		dist -= dy*fine;
		cam.setOrbit ( cam.getAng(), cam.getToPos(), dist, cam.getDolly() );		

		appPostRedisplay();	// Update display
		} break;
	}
}

bool Sample::init ()
{
	#ifdef BUILD_CUDA
		// Initialize CUDA
		cudaInit ();
	#endif

	PERF_INIT ( 64, false, true, false, 0, "" );	
		
	addGui (  20,   20, 200, 12, "Frame Time - FPS ",	GUI_PRINT,  GUI_INT,	&frameFPS, 0, 0 );					
	addGui (  20,   35, 200, 12, "Frame Time - msec ",	GUI_PRINT,  GUI_FLOAT,	&frameTime, 0, 0 );							
	addGui (  20,   50, 200, 27, "# of Particles",		GUI_SLIDER, GUI_INT,	&psys_count, 1024, 8000000 );
	addGui (  20,   80, 200, 27, "Gravity",				GUI_SLIDER, GUI_FLOAT,	&gravity, 0, 20.0 );
	addGui (  20,  110, 200, 27, "Viscosity",			GUI_SLIDER, GUI_FLOAT,	&visc, 0, 1 );
	addGui (  20,  140, 200, 27, "Rest Density",			GUI_SLIDER, GUI_FLOAT,	&restdens, 0, 1000.0 );
		
	init2D ( "arial" );		// specify font file (.bin/tga)
	setText ( 16.0, -0.5 );		// scale by 0.5, kerning adjust -0.5 pixels
	setview2D ( window_width, window_height );
	setorder2D ( true, -0.001 );
	
	psys.SetupRender ();

	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);	
	glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);	

	srand ( time ( 0x0 ) );

	glClearColor( 0.49, 0.49, 0.49, 1.0 );
	glShadeModel( GL_SMOOTH );

	glEnable ( GL_COLOR_MATERIAL );
	glEnable (GL_DEPTH_TEST);  
	glEnable (GL_BLEND);
	glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 
	glDepthMask ( 1 );
	glEnable ( GL_TEXTURE_2D );


	// glutSetCursor ( GLUT_CURSOR_NONE );
	
	// Initialize camera
	cam.setOrbit ( Vector3DF(200,30,0), Vector3DF(2,2,2), 600, 400 );
	cam.setFov ( 50 );
	cam.updateMatricies ();
	
	light[0].x = 0;		light[0].y = 200;	light[0].z = 0; light[0].w = 1;
	light_to[0].x = 0;	light_to[0].y = 0;	light_to[0].z = 0; light_to[0].w = 1;

	light[1].x = 55;		light[1].y = 140;	light[1].z = 50;	light[1].w = 1;
	light_to[1].x = 0;	light_to[1].y = 0;	light_to[1].z = 0;		light_to[1].w = 1;

	light_fov = 45;

	obj_from.x = 0;		obj_from.y = 0;		obj_from.z = 20;		// emitter
	obj_angs.x = 118.7;	obj_angs.y = 200;	obj_angs.z = 1.0;
	obj_dang.x = 1;	obj_dang.y = 1;		obj_dang.z = 0;

	psys.Setup (true);
	psys.SetVec ( PEMIT_ANG, Vector3DF ( obj_angs.x, obj_angs.y, obj_angs.z ) );
	psys.SetVec ( PEMIT_POS, Vector3DF ( obj_from.x, obj_from.y, obj_from.z ) );

	psys.SetParam ( PCLR_MODE, iClrMode );	

	psys.SetParam ( PNUM, 4000000 );

	psys_playback = psys.getLastRecording ();
	
	// Get initial number of particle (from scene XML)
	psys_count = psys.GetParam ( PNUM );
	

	return true;
}

void Sample::startup () 
{
	int w=1270, h=800;
	
	appStart ( "Fluids v3.2", "Fluids v3.2", w, h, 4, 0, 1 );	
}
