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

#include <time.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <windows.h>

#include "app_opengl.h"
#include "app_perf.h"

#include "fluid_defs.h"

#ifdef BUILD_CUDA
	#include "fluid_system_host.cuh"	
#endif
#include "fluid_system.h"

bool bTiming = true;
bool bRec = false;
float frameTime;
int frameFPS;
int mFrame = 0;
float gravity = 9.8;
float visc = 0.35;
float restdens = 600.0;

// Globals
FluidSystem		psys;
Camera3D		cam;

Vector3DF	obj_from, obj_angs, obj_dang;
Vector4DF	light[2], light_to[2];				// Light stuff
float		light_fov;	

int		psys_count = 16384;
int		psys_rate = 0;							// Particle stuff
int		psys_freq = 1;
int		psys_playback;

bool	bHelp = true;					// Toggles
int		iShade = 0;						// Shading mode (default = no shadows)
int		iClrMode = 0;
bool    bPause = false;

// View matricies
float model_matrix[16];					// Model matrix (M)

// Mouse control
#define DRAG_OFF		0				// mouse states
#define DRAG_LEFT		1
#define DRAG_RIGHT		2
int		last_x = -1, last_y = -1;		// mouse vars
int		mode = 0;
int		dragging = 0;
int		psel;

GLuint	screen_id;
GLuint	depth_id;


// Different things we can move around
#define MODE_CAM		0
#define MODE_CAM_TO		1
#define MODE_OBJ		2
#define MODE_OBJPOS		3
#define MODE_OBJGRP		4
#define MODE_LIGHTPOS	5

#define MODE_DOF		6

GLuint screenBufferObject;
GLuint depthBufferObject;
GLuint envid;

void drawScene ( float* viewmat, bool bShade )
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

		psys.Draw ( cam, 0.8 );				// Draw particles		

	} else {
		glDisable ( GL_LIGHTING );
		psys.Draw ( cam, 0.55 );			// Draw particles
	}
}

void drawInfo ()
{
	
	Time start, stop;

	glDisable ( GL_LIGHTING );  
	glDisable ( GL_DEPTH_TEST );

	glMatrixMode ( GL_PROJECTION );
	glLoadIdentity ();  
	glScalef ( 2.0/window_width, -2.0/window_height, 1 );		// Setup view (0,0) to (800,600)
	glTranslatef ( -window_width/2.0, -window_height/2, 0.0);


	float view_matrix[16];
	glMatrixMode ( GL_MODELVIEW );
	glLoadIdentity ();

	// Particle Information
	if ( psys.GetSelected() != -1 ) {	
		psys.DrawParticleInfo ();
		return;	
	}

	char disp[200];
		
	/*psys.getModeClr ();
	strcpy ( disp, psys.getModeStr().c_str() ); drawText ( 20, 40, disp );*/

	
	start2D ();

	glColor4f ( 1.0, 1.0, 1.0, 1.0 );	
	strcpy ( disp, "Press H for help." );		drawText ( 5, 10, disp );  

	glColor4f ( 1.0, 1.0, 0.0, 1.0 );	
	strcpy ( disp, "" );
	if ( psys.GetToggle(PCAPTURE) ) strcpy ( disp, "CAPTURING VIDEO");
	drawText ( 200, 20, disp );

	//-- Additional info (CPU only)
	/* sprintf ( disp,	"Mode:                %s", psys.getModeStr().c_str() );					drawText ( 20, 40,  disp );		
	sprintf ( disp,	"Scene:               %s (id: %d)", psys.getSceneName().c_str(), (int) psys.GetParam(PEXAMPLE) );				drawText ( 20, 60,  disp );		
	sprintf ( disp,	"Grid Density:        %f", psys.GetParam (PGRID_DENSITY) );		drawText ( 20, 100,  disp );
	sprintf ( disp,	"Grid Count:          %f", (float) psys.GetParam( PSTAT_GRIDCNT ) / psys.GetParam(PSTAT_OCCUPY) );	drawText ( 20, 110,  disp );
	sprintf ( disp,	"Grid Occupancy:      %f%%", (float) psys.GetParam( PSTAT_OCCUPY ) / psys.getGridTotal() );		drawText ( 20, 130,  disp );
	sprintf ( disp,	"Grid Resolution:     %d x %d x %d (%d)", (int) psys.GetGridRes().x, (int) psys.GetGridRes().y, (int) psys.GetGridRes().z, psys.getGridTotal() );		drawText ( 20, 140,  disp );
	int nsrch = pow ( psys.getSearchCnt(), 1/3.0 );
	sprintf ( disp,	"Grid Search:         %d x %d x %d", nsrch, nsrch, nsrch );			drawText ( 20, 150,  disp );
	sprintf ( disp,	"Search Count:        %d, ave: %f, max: %f", (int) psys.GetParam(PSTAT_SRCH), psys.GetParam(PSTAT_SRCH)/psys.NumPoints(), psys.GetParam(PSTAT_SRCHMAX)/psys.NumPoints() );		drawText ( 20, 160,  disp );
	sprintf ( disp,	"Neighbor Count:      %d, ave: %f, max: %f", (int) psys.GetParam(PSTAT_NBR), psys.GetParam(PSTAT_NBR)/psys.NumPoints(), psys.GetParam(PSTAT_NBRMAX)/psys.NumPoints() );		drawText ( 20, 170,  disp );
	sprintf ( disp,	"Search Overhead:     %.2fx", psys.GetParam(PSTAT_SRCH)/psys.GetParam(PSTAT_NBR) );		drawText ( 20, 180,  disp );
	sprintf ( disp,	"Performance:         %d particles/sec", (int) ((psys.NumPoints()*1000.0)/st) );			drawText ( 20, 270,  disp );
	sprintf ( disp,	"Particle Memory:     %.4f MB", (float) psys.GetParam(PSTAT_PMEM)/1000000.0f );		drawText ( 20, 290,  disp );
	sprintf ( disp,	"Grid Memory:         %.4f MB", (float) psys.GetParam(PSTAT_GMEM)/1000000.0f );		drawText ( 20, 300,  disp );*/

	end2D();
}

int frame;

void display () 
{
	PERF_PUSH ( "FRAME" );	

	// Check for slider interaction
	if ( guiChanged(2) ) {	
		psys.SetParam ( PNUM, psys_count, 4, 40000000 );
		psys.Setup ( false );
	}
	if ( guiChanged(3) ) 	psys.SetVec ( PPLANE_GRAV_DIR, Vector3DF(0,-gravity,0) );
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

	// Compute camera view
	cam.updateMatricies ();
	glMatrixMode ( GL_PROJECTION );
	glLoadMatrixf ( cam.getProjMatrix().GetDataF() );
	
	// Draw 3D	
	glEnable ( GL_LIGHTING );  
	glMatrixMode ( GL_MODELVIEW );
	glLoadMatrixf ( cam.getViewMatrix().GetDataF() );
	drawScene ( cam.getViewMatrix().GetDataF() , true );

	// Draw 2D overlay
	PERF_PUSH ( "DrawInfo" );
	drawInfo ();
	PERF_POP ();

	//if ( psys.GetToggle(PPROFILE) ) { rstop.SetSystemTime ( ACC_NSEC ); rstop = rstop - rstart; printf ( "RENDER: %s\n", rstop.GetReadableTime().c_str() ); }
 
	if ( bHelp ) drawGui ();

	draw2D ();

	// Swap buffers
	SwapBuffers ( g_hDC );

	PERF_POP ();


	frameTime = PERF_POP ();	
	frameFPS = int(1000.0 / frameTime);
}

void reshape ( int width, int height ) 
{
  // set window height and width
  window_width  = (float) width;
  window_height = (float) height;
  glViewport( 0, 0, width, height );    
  setview2D ( width, height );
}

void UpdateEmit ()
{	
	obj_from = psys.GetVec ( PEMIT_POS );
	obj_angs = psys.GetVec ( PEMIT_ANG );
	obj_dang = psys.GetVec ( PEMIT_RATE );
}


void keyboard_func ( unsigned char key, int x, int y )
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

Vector3DF cangs;
Vector3DF ctp;
float cdist;

void mouse_click_func ( int button, int state, int x, int y )
{
  cangs = cam.getAng();
  ctp = cam.getToPos();
  cdist = cam.getOrbitDist();

  

  if( state == GLUT_DOWN ) {

		// Handle 2D gui interaction first
		if ( guiMouseDown ( x, y ) ) return;

		if ( button == GLUT_LEFT_BUTTON )		dragging = DRAG_LEFT;
		else if ( button == GLUT_RIGHT_BUTTON ) dragging = DRAG_RIGHT;	
		last_x = x;
		last_y = y;	

  } else if ( state==GLUT_UP ) {

		dragging = DRAG_OFF;
  }
}

void mouse_move_func ( int x, int y )
{
	//psys.SelectParticle ( x, y, window_width, window_height, cam );
}

void mouse_drag_func ( int x, int y )
{
	int dx = x - last_x;
	int dy = y - last_y;

	// Handle GUI interaction in nvGui by calling guiMouseDrag
	if ( guiMouseDrag ( x, y ) ) return;

	switch ( mode ) {
	case MODE_CAM:
		if ( dragging == DRAG_LEFT ) {
			cam.moveOrbit ( dx, dy, 0, 0 );
		} else if ( dragging == DRAG_RIGHT ) {
			cam.moveOrbit ( 0, 0, 0, dy*0.15 );
		} 
		break;
	case MODE_CAM_TO:
		if ( dragging == DRAG_LEFT ) {
			cam.moveToPos ( dx*0.1, 0, dy*0.1 );
		} else if ( dragging == DRAG_RIGHT ) {
			cam.moveToPos ( 0, dy*0.1, 0 );
		}
		break;	
	case MODE_OBJ:
		if ( dragging == DRAG_LEFT ) {
			obj_angs.x -= dx*0.1;
			obj_angs.y += dy*0.1;
			app_printf ( "Obj Angs:  %f %f %f\n", obj_angs.x, obj_angs.y, obj_angs.z );
			//force_x += dx*.1;
			//force_y += dy*.1;
		} else if (dragging == DRAG_RIGHT) {
			obj_angs.z -= dy*.005;			
			app_printf ( "Obj Angs:  %f %f %f\n", obj_angs.x, obj_angs.y, obj_angs.z );
		}
		psys.SetVec ( PEMIT_ANG, Vector3DF ( obj_angs.x, obj_angs.y, obj_angs.z ) );
		break;
	case MODE_OBJPOS:
		if ( dragging == DRAG_LEFT ) {
			obj_from.x -= dx*.1;
			obj_from.y += dy*.1;
			app_printf ( "Obj:  %f %f %f\n", obj_from.x, obj_from.y, obj_from.z );
		} else if (dragging == DRAG_RIGHT) {
			obj_from.z -= dy*.1;
			app_printf ( "Obj:  %f %f %f\n", obj_from.x, obj_from.y, obj_from.z );
		}
		psys.SetVec ( PEMIT_POS, Vector3DF ( obj_from.x, obj_from.y, obj_from.z ) );
		//psys.setPos ( obj_x, obj_y, obj_z, obj_ang, obj_tilt, obj_dist );
		break;
	case MODE_LIGHTPOS:
		if ( dragging == DRAG_LEFT ) {
			light[0].x -= dx*.1;
			light[0].z -= dy*.1;		
			app_printf ( "Light: %f %f %f\n", light[0].x, light[0].y, light[0].z );
		} else if (dragging == DRAG_RIGHT) {
			light[0].y -= dy*.1;			
			app_printf ( "Light: %f %f %f\n", light[0].x, light[0].y, light[0].z );
		}	
		break;
	}

	last_x = x;
	last_y = y;
}


void idle_func ()
{
}

void init ()
{
	
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
	cam.setOrbit ( Vector3DF(200,30,0), Vector3DF(2,2,2), 400, 400 );
	cam.setFov ( 35 );
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

	psys_playback = psys.getLastRecording ();
	
	// Get initial number of particle (from scene XML)
	psys_count = psys.GetParam ( PNUM );
}


void initialize ()
{
	#ifdef BUILD_CUDA
		// Initialize CUDA
		cudaInit ();
	#endif

	PERF_INIT ( true );
	PERF_SET ( true, 0, true, "" );
		
	addGui (  20,   20, 200, 12, "Frame Time - FPS ",	GUI_PRINT,  GUI_INT,	&frameFPS, 0, 0 );					
	addGui (  20,   35, 200, 12, "Frame Time - msec ",	GUI_PRINT,  GUI_FLOAT,	&frameTime, 0, 0 );							
	addGui (  20,   50, 200, 27, "# of Particles",		GUI_SLIDER, GUI_INTLOG,	&psys_count, 1024, 8000000 );
	addGui (  20,   80, 200, 27, "Gravity",				GUI_SLIDER, GUI_FLOAT,	&gravity, 0, 20.0 );
	addGui (  20,  110, 200, 27, "Viscosity",			GUI_SLIDER, GUI_FLOAT,	&visc, 0, 1 );
	addGui (  20,  140, 200, 27, "Rest Density",			GUI_SLIDER, GUI_FLOAT,	&restdens, 0, 1000.0 );
		
	init2D ( "arial_12" );		// specify font file (.bin/tga)
	setText ( 1.0, -0.5 );		// scale by 0.5, kerning adjust -0.5 pixels
	setview2D ( window_width, window_height );
	setorder2D ( true, -0.001 );
	
	init();	
	
	psys.SetupRender ();

}


void shutdown ()
{
	
}
