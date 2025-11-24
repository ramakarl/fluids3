//--------------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2017, NVIDIA Corporation. 
//
// Redistribution and use in source and binary forms, with or without modification, 
// are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer 
//    in the documentation and/or  other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived 
//    from this software without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
// BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
// SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE 
// OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 
// Version 1.0: Rama Hoetzlein, 5/1/2017
//----------------------------------------------------------------------------------

/*!
 * This file provides a utility classes for 2D Drawing and GUIs
 * Functionality in this file:
 *  - nvMesh: Construct, load, and render meshes. PLY format supported
 *  - nvImg: Cosntruct, load, and render images. PNG and TGA format supported
 *  - nvDraw: A lightweight, efficient, 2D drawing API. Uses VBOs to render
 *     lines, circles, triangles, and text. Allows for both static and dynamic 
 *     groups (rewrite per frame), and immediate style (out-of-order) calling.
 *  - nvGui: A lightweight class for creating on-screen GUIs. Currently only checkboxes
 *    or sliders supported. Relies on nvDraw to render.
 * Useage: 
 *    1. Main programs implement the functions required by app_opengl/directx.
 *    2. During display(), first do any rendering desired by your demo or application.
 *    3. Then call drawGui to render GUI items to the 2D layer.
 *    4. Then call draw2D to actually render all 2D objects (gui and user-specified)
 *    5. Finally call SwapBuffers 
 */

#include "nv_gui.h"
#include "string_helper.h"
#include "file_tga.h"		// for tga fonts

#ifdef USE_VEC_HELPER
	#include "vec.h"
#endif
#ifdef USE_VEC
	#include "gvdb_vec.h"	
#endif 
#include "quaternion.h"

// Globals
typedef void (*CallbackFunc)(int, float);
nvDraw*			g_2D;
nvGui*			g_Gui;
CallbackFunc	g_GuiCallback;

struct MatrixBuffer 
{
    float m[16];    
};

#ifndef checkGL


#endif

// Utility functions

bool init2D ( const char* fontName )		{ return g_2D->Initialize( fontName ); }
void drawGL ()		{ g_2D->drawGL(); }
void setLight (int s, float x1, float y1, float z1 )	{ g_2D->setLight(s,x1,y1,z1); }
void setOffset (int s, float x1, float y1, float z1, float sz )	{ g_2D->setOffset(s,x1,y1,z1, sz ); }
void setPreciseEye (int s, Camera3D* cam )						{ g_2D->setPreciseEye(s, cam); }
void start2D ()		{ g_2D->start2D(); }
void start2D (bool bStatic)		{ g_2D->start2D(bStatic); }
void updatestatic2D ( int n )	{ g_2D->updateStatic2D (n); }
void static2D ()	{ g_2D->start2D(true); }
void end2D ()		{ g_2D->end2D(); }
void draw2D ()		{ g_2D->draw2D(); }
void setview2D ( int w, int h)			{ g_2D->setView2D( w,h ); }
void setview2D ( float* model, float* view, float* proj )		{ g_2D->setView2D( model, view, proj ); }
void setorder2D ( bool zt, float zfactor )		{ g_2D->setOrder2D( zt, zfactor ); }
void setdepth3D(bool z) { g_2D->setOrder3D(z); }
void setText ( float scale, float kern )	{ g_2D->setText(scale,kern); }
void drawLine ( float x1, float y1, float x2, float y2, float r, float g, float b, float a )	{ g_2D->drawLine(x1,y1,x2,y2,r,g,b,a); }
void drawRect ( float x1, float y1, float x2, float y2, float r, float g, float b, float a )	{ g_2D->drawRect(x1,y1,x2,y2,r,g,b,a); }
void drawImg ( int img_glid,  float x1, float y1, float x2, float y2, float r, float g, float b, float a )	{ g_2D->drawImg ( img_glid, x1,y1,x2,y2,r,g,b,a ); }
void drawFill ( float x1, float y1, float x2, float y2, float r, float g, float b, float a )	{ g_2D->drawFill(x1,y1,x2,y2,r,g,b,a); }
void drawTri  ( float x1, float y1, float x2, float y2, float x3, float y3, float r, float g, float b, float a )	{ g_2D->drawTri(x1,y1,x2,y2,x3,y3,r,g,b,a); }
void drawCircle ( float x1, float y1, float radius, float r, float g, float b, float a )		{ g_2D->drawCircle(x1,y1,radius,r,g,b,a); }
void drawCircleDash ( float x1, float y1, float radius, float r, float g, float b, float a )	{ g_2D->drawCircleDash(x1,y1,radius,r,g,b,a); }
void drawCircleFill ( float x1, float y1, float radius, float r, float g, float b, float a )	{ g_2D->drawCircleFill(x1,y1,radius,r,g,b,a); }
void drawText ( float x1, float y1, char* msg, float r, float g, float b, float a )				{ g_2D->drawText(x1,y1,msg,r,g,b,a); }
void getTextSize ( const char* msg, float& sx, float& sy )	{ return g_2D->getTextSize(msg, sx, sy); }
void getGlyphSize ( const char c, float& sx, float& sy )	{ return g_2D->getGlyphSize(c, sx, sy); }

void start3D ( Camera3D* cam )		{ g_2D->start3D( cam ); }
void selfDraw3D ( Camera3D* cam, int sh )	{ g_2D->selfDraw3D( cam, sh ); }
void selfEndDraw3D ()				{ g_2D->selfEndDraw3D(); }
void drawLine3D ( float x1, float y1, float z1, float x2, float y2, float z2, float r, float g, float b, float a ) { g_2D->drawLine3D(x1,y1,z1,x2,y2,z2,r,g,b,a); }
void drawCyl3D ( Vector3DF p1, Vector3DF p2, float r1, float r2, Vector4DF clr )  { g_2D->drawCyl3D(p1,p2,r1,r2,clr ); }
void drawTri3D(float x1, float y1, float z1, float x2, float y2, float z2, float x3, float y3, float z3, float nx, float ny, float nz, float r, float g, float b, float a) { g_2D->drawTri3D(x1, y1, z1, x2, y2, z2, x3, y3, z3, nx, ny, nz, r, g, b, a); }
void drawFace3D(float x1, float y1, float z1, float x2, float y2, float z2, float x3, float y3, float z3, float x4, float y4, float z4, float nx, float ny, float nz, float r, float g, float b, float a) { g_2D->drawFace3D(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4,y4,z4, nx, ny, nz, r, g, b, a); }
void drawCircle3D ( Vector3DF p1, Vector3DF p2, float r1, Vector4DF clr  )					 { g_2D->drawCircle3D(p1, p2, r1, clr);  }
void drawBox3D ( Vector3DF p, Vector3DF pa, Vector3DF pb, Matrix4F xform, float r, float g, float b, float a ) { g_2D->drawBox3D(p,pa,pb,xform,r,g,b,a); }
void drawBox3D (Vector3DF p, Vector3DF q, float r, float g, float b, float a)				{ g_2D->drawBox3D(p, q, r, g, b, a); }
void drawBox3DXform ( Vector3DF b1, Vector3DF b2, Vector3DF clr, Matrix4F& xform )			{ g_2D->drawBox3DXform ( b1, b2, clr, xform ); }
void drawPoint3D (Vector3DF p,float r,float g,float b,float a)								{ g_2D->drawPoint3D (p,r,g,b,a); }
void drawCube3D (Vector3DF p, Vector3DF q, float r, float g, float b, float a)				{ g_2D->drawCube3D(p, q, r, g, b, a); }
void drawImg3D ( int img_glid, Vector3DF p1, Vector3DF p2, Vector3DF p3, Vector3DF p4 )		{ g_2D->drawImg3D(img_glid, p1, p2, p3, p4); }
void drawPnts3D ( Vector4DF* pnts, Vector4DF* clrs, int pnt_num )							{ g_2D->drawPnts3D(pnts, clrs, pnt_num); }
void drawText3D ( Camera3D* cam, float x1, float y1, float z1, char* msg, float r, float g, float b, float a )				{ g_2D->drawText3D(cam,x1,y1,z1,msg,r,g,b,a); }
void end3D ()		{ g_2D->end3D(); }
void draw3D ()		{ g_2D->draw3D(); }

void drawGui ( nvImg* img)		{ g_Gui->Draw( img ); }
void clearGuis ()				{ g_Gui->Clear(); }
int  addGui ( int x, int y, int w, int h, char* name, int gtype, int dtype, void* data, float vmin, float vmax ) { return g_Gui->AddGui ( float(x), float(y), float(w), float(h), name, gtype, dtype, data, vmin, vmax ); }
void setBackclr ( float r, float g, float b, float a )	{ return g_Gui->SetBackclr ( r,g,b,a ); }
int  addItem ( char* name )		{ return g_Gui->AddItem ( name ); }
int  addItem ( char* name, char* imgname ) { return g_Gui->AddItem ( name, imgname ); }
std::string guiItemName ( int n, int v )	{ return g_Gui->getItemName ( n, v ); }
bool guiChanged ( int n )  { return g_Gui->guiChanged(n); }
bool guiMouseDown ( float x, float y )	{ return g_Gui->MouseDown(x,y); }
bool guiMouseUp ( float x, float y )	{ return g_Gui->MouseUp(x,y); }
bool guiMouseDrag ( float x, float y )	{ return g_Gui->MouseDrag(x,y); }
void guiSetCallback ( CallbackFunc f )  { g_GuiCallback = f; }


#ifndef USE_STR_HELPER

bool readword ( char *line, char delim, char *word, int max_size )
{
	char *buf_pos;
	char *start_pos;	

	// read past spaces/tabs, or until end of line/string
	for (buf_pos=line; (*buf_pos==' ' || *buf_pos=='\t') && *buf_pos!='\n' && *buf_pos!='\0';)
		buf_pos++;
	
	// if end of line/string found, then no words found, return null
	if (*buf_pos=='\n' || *buf_pos=='\0') {*word = '\0'; return false;}

	// mark beginning of word, read until end of word
	for (start_pos = buf_pos; *buf_pos != delim && *buf_pos!='\t' && *buf_pos!='\n' && *buf_pos!='\0';)
		buf_pos++;
	
	if (*buf_pos=='\n' || *buf_pos=='\0') {	// buf_pos now points to the end of buffer
        strncpy_s (word, max_size, start_pos, max_size);	// copy word to output string
		if ( *buf_pos=='\n') *(word + strlen(word)-1) = '\0';
		*line = '\0';						// clear input buffer
	} else {
											// buf_pos now points to the delimiter after word
		*buf_pos++ = '\0';					// replace delimiter with end-of-word marker
		strncpy_s ( word, max_size, start_pos, (long long) (buf_pos-line) );	// copy word(s) string to output string			
											// move start_pos to beginning of entire buffer
		strcpy ( start_pos, buf_pos );		// copy remainder of buffer to beginning of buffer
	}
	return true;						// return word(s) copied	
}

#endif


/*void save_png ( char* fname, unsigned char* img, int w, int h )
{
	unsigned error = lodepng::encode ( "test.png", img, w, h );	  
	if (error) printf ( "png write error: %s\n", lodepng_error_text(error) );
}*/



bool guiHandler ( int button, int action, int x, int y )
{
	switch ( action ) {
	case 0:		return g_Gui->MouseUp( float(x), float(y) );		break;
	case 1:		return g_Gui->MouseDown( float(x), float(y) );	break;
	case 2:		return g_Gui->MouseDrag( float(x), float(y) );	break;
	}
	return false;
}

void enable_nvgui()
{
	g_Gui = new nvGui;
	g_2D = new nvDraw;
}
void disable_nvgui()
{
	delete g_Gui;
	delete g_2D;
}

nvDraw::nvDraw ()
{
	mCurrZ = 0;	
	mDynNum = 0;
	mTextScale = 1;
	mTextKern = 0;
	mWidth = 0;
	mHeight = 0;
	mVAO = 65535;
	m3DNum = 0;
	setOrder2D ( false, 1.0 );
}
nvDraw::~nvDraw()
{
	Clear();
}

void nvDraw::Clear()
{
	std::vector<nvSet>* pList;
	nvSet* s;
	
	for (int iter = 0; iter < 3; iter++ ) {

		// process the three kinds of lists
		switch (iter) {
		case 0:	pList = &mStatic;	break;
		case 1:	pList = &mDynamic;	break;
		case 2: pList = &m3D;		break;
		};

		// process each set in list (ie. calls to start/end3d)
		for (int n=0; n < pList->size(); n++) {
			s = &pList->at(n);			
			// process each group (ie. lines, tris, imgs, etc.)
			for (int g = 0; g < GRP_MAX; g++) {
				if (s->mGeom[g] != 0x0) { free(s->mGeom[g]); s->mGeom[g] = 0; }				
			}
			pList->clear ();
		}
	}
}

void nvDraw::setView2D ( int w, int h )
{
	mWidth = float(w); mHeight = float(h);
}
void nvDraw::setView2D ( float* model, float* view, float* proj )
{
	mWidth = -1; mHeight = -1;

	memcpy ( mModelMtx, model, 16 * sizeof(float) );
	memcpy ( mViewMtx, view, 16 * sizeof(float) );
	memcpy ( mProjMtx, proj, 16 * sizeof(float) );
}
void nvDraw::updateStatic2D ( int n )
{
	if ( n < 0 || n >= mStatic.size()) return;
	if ( mWidth==-1 ) {		
		SetMatrixView ( mStatic[n], mModelMtx, mViewMtx, mProjMtx, (float) mZFactor );
	} else {
		SetDefaultView ( mStatic[n], mWidth, mHeight, (float) mZFactor );
	}
}
void nvDraw::setView3D ( nvSet& s, Camera3D* cam )
{
	Matrix4F ident; 
	ident.Identity ();
	memcpy ( s.model, ident.GetDataF(), 16 * sizeof(float) );
	memcpy ( s.view, cam->getViewMatrix().GetDataF(), 16 * sizeof(float) );
	memcpy ( s.proj, cam->getProjMatrix().GetDataF(), 16 * sizeof(float) );
}

void nvDraw::start3D ( Camera3D* cam )
{
	if ( mVAO == 65535 ) {
		dbgprintf ( "ERROR: nv_gui was not initialized. Must call init2D.\n" );
	}
	if ( m3DNum >= m3D.size() ) {
		nvSet new_set;	
		memset ( &new_set, 0, sizeof(nvSet) );
		m3D.push_back ( new_set );		
	}	
	nvSet* s = &m3D[ m3DNum ];	
	m3DNum++;
	mCurrSet = s;	
	s->zfactor = 1;				// default = DEPTH TEST ENABLE
	for (int n=0; n < GRP_MAX; n++ ) {
		s->mNum[n] = 0;	
		s->mNumI[n] = 0;	
	}
	setView3D ( *s, cam );		// set 3D transforms
}		

void nvDraw::drawCircle3D ( Vector3DF p1, Vector3DF p2, float r1, Vector4DF clr  )
{
	int ndx;
	int UR = 32;
	float UR1 = (UR-1.0f);	
	nvVert* v = allocGeom( UR*2, GRP_LINES, mCurrSet, ndx);	

	// determine orientation of vector from p1 to p2
	Vector3DF q1, q2, n;
	quaternion rot;	
	rot.rotationFromTo ( Vector3DF(0,0,1), p2-p1 );

	q1.Set ( cos( (0/UR1)*PI*2.0 )*r1, sin( (0/UR1)*PI*2.0 )*r1, 0.0f ); q1 = rot.rotateVec(q1); n = q1; q1 += p1;
	v->x = q1.x; v->y = q1.y; v->z = q1.z;	v->r = clr.x; v->g = clr.y; v->b = clr.z; v->a = clr.w; v->tx = 0; v->ty = 0;  v->nx = n.x; v->ny = n.y; v->nz = n.z; v++;   // repeat first for jump
	v->x = q1.x; v->y = q1.y; v->z = q1.z;	v->r = clr.x; v->g = clr.y; v->b = clr.z; v->a = clr.w; v->tx = 0; v->ty = 0;  v->nx = n.x; v->ny = n.y; v->nz = n.z; v++;
	q2 = q1;
		
	for (int u=1; u < UR; u++ ) {
		// draw circle, oriented along p2-p1
		q1.Set ( cos( (u/UR1)*PI*2.0 )*r1, sin( (u/UR1)*PI*2.0 )*r1, 0.0f ); q1 = rot.rotateVec(q1); n = q1; q1 += p1;		
		v->x = q1.x; v->y = q1.y; v->z = q1.z;	v->r = clr.x; v->g = clr.y; v->b = clr.z; v->a = clr.w; v->tx = 0; v->ty = 0;  v->nx = n.x; v->ny = n.y; v->nz = n.z; v++;
		v->x = q2.x; v->y = q2.y; v->z = q2.z;	v->r = clr.x; v->g = clr.y; v->b = clr.z; v->a = clr.w; v->tx = 0; v->ty = 0;  v->nx = n.x; v->ny = n.y; v->nz = n.z; v++;
		q2 = q1;
	}			
}

void nvDraw::drawPoint3D (Vector3DF p, float r,float g,float b,float a)
{
	int ndx;
	nvVert* v = allocGeom (1, GRP_POINTS,mCurrSet,ndx);
	v->x = p.x; v->y = p.y; v->z = p.z; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0; v->nx=0; v->ny=1; v->nz=0;	v++;	
}

void nvDraw::drawCyl3D ( Vector3DF p1, Vector3DF p2, float r1, float r2, Vector4DF clr ) 
{
	int ndx;
	int UR = 8;
	float UR1 = (UR-1.0f);
	nvVert* v = allocGeom ( UR*2+2, GRP_TRI, mCurrSet, ndx );	
	uint* i = allocIdx( UR*2+2, GRP_TRI, mCurrSet );

	// GRI_TRI is triangle strip

	// determine orientation of vector from p1 to p2
	Vector3DF q1, q2, n;
	quaternion rot;	
	rot.rotationFromTo ( Vector3DF(0,0,1), p2-p1 );
	
	q1.Set ( cos( (0/UR1)*PI*2.0 )*r1, sin( (0/UR1)*PI*2.0 )*r1, 0.0f ); q1 = rot.rotateVec(q1); n = q1; q1 += p1;
	q2.Set ( cos( (0/UR1)*PI*2.0 )*r2, sin( (0/UR1)*PI*2.0 )*r2, 0.0f ); q2 = rot.rotateVec(q2); q2 += p2;
	v->x = q1.x; v->y = q1.y; v->z = q1.z;	v->r = clr.x; v->g = clr.y; v->b = clr.z; v->a = clr.w; v->tx = 0; v->ty = 0;  v->nx = n.x; v->ny = n.y; v->nz = n.z; v++;   // repeat first for jump
	v->x = q1.x; v->y = q1.y; v->z = q1.z;	v->r = clr.x; v->g = clr.y; v->b = clr.z; v->a = clr.w; v->tx = 0; v->ty = 0;  v->nx = n.x; v->ny = n.y; v->nz = n.z; v++;
	v->x = q2.x; v->y = q2.y; v->z = q2.z;	v->r = clr.x; v->g = clr.y; v->b = clr.z; v->a = clr.w; v->tx = 0; v->ty = 0;  v->nx = n.x; v->ny = n.y; v->nz = n.z; v++;
		
	for (int u=1; u < UR; u++ ) {
		// draw circles at the caps, orient them along p2-p1
		q1.Set ( cos( (u/UR1)*PI*2.0 )*r1, sin( (u/UR1)*PI*2.0 )*r1, 0.0f ); q1 = rot.rotateVec(q1); n = q1; q1 += p1;
		q2.Set ( cos( (u/UR1)*PI*2.0 )*r2, sin( (u/UR1)*PI*2.0 )*r2, 0.0f ); q2 = rot.rotateVec(q2); q2 += p2;
		// triangle strip
		v->x = q1.x; v->y = q1.y; v->z = q1.z;	v->r = clr.x; v->g = clr.y; v->b = clr.z; v->a = clr.w; v->tx = 0; v->ty = 0;  v->nx = n.x; v->ny = n.y; v->nz = n.z; v++;
		v->x = q2.x; v->y = q2.y; v->z = q2.z;	v->r = clr.x; v->g = clr.y; v->b = clr.z; v->a = clr.w; v->tx = 0; v->ty = 0;  v->nx = n.x; v->ny = n.y; v->nz = n.z; v++;
	}			
	// repeat last for jump	
	v->x = q2.x; v->y = q2.y; v->z = q2.z;	v->r = clr.x; v->g = clr.y; v->b = clr.z; v->a = clr.w; v->tx = 0; v->ty = 0;  v->nx = n.x; v->ny = n.y; v->nz = n.z; v++;

	for (int j=0; j < UR*2+2; j++)
		*i++ = ndx++; 
}

void nvDraw::drawLine3D ( float x1, float y1, float z1, float x2, float y2, float z2, float r, float g, float b, float a )
{
	int ndx;
	nvVert* v = allocGeom ( 2, GRP_LINES, mCurrSet, ndx );
	v->x = x1; v->y = y1; v->z = z1; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;
	v->x = x2; v->y = y2; v->z = z2; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;
}
void nvDraw::drawTri3D(float x1, float y1, float z1, float x2, float y2, float z2, float x3, float y3, float z3, float nx, float ny, float nz, float r, float g, float b, float a)
{
	int ndx;
	nvVert* v = allocGeom(3, GRP_TRI, mCurrSet, ndx);
	uint* i = allocIdx(4, GRP_TRI, mCurrSet);

	v->x = x1; v->y = y1; v->z = z1; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v->nx = nx; v->ny = ny; v->nz = nz; v++;
	v->x = x2; v->y = y2; v->z = z2; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v->nx = nx; v->ny = ny; v->nz = nz; v++;
	v->x = x3; v->y = y3; v->z = z3; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;  v->nx = nx; v->ny = ny; v->nz = nz;
	*i++ = ndx++; *i++ = ndx++; *i++ = ndx++; *i++ = (ndx-1);
}
void nvDraw::drawFace3D(float x1, float y1, float z1, float x2, float y2, float z2, float x3, float y3, float z3, float x4, float y4, float z4, float nx, float ny, float nz, float r, float g, float b, float a)
{
	int ndx;
	nvVert* v = allocGeom(8, GRP_TRI, mCurrSet, ndx);
	uint* i = allocIdx(8, GRP_TRI, mCurrSet);

	// repeat first for jump
	v->x = x1; v->y = y1; v->z = z1; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0; 	v->nx = nx; v->ny = ny; v->nz = nz; v++;

	v->x = x1; v->y = y1; v->z = z1; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0; 	v->nx = nx; v->ny = ny; v->nz = nz; v++;
	v->x = x2; v->y = y2; v->z = z2; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v->nx = nx; v->ny = ny; v->nz = nz; v++;
	v->x = x3; v->y = y3; v->z = z3; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;  v->nx = nx; v->ny = ny; v->nz = nz; v++;

	v->x = x1; v->y = y1; v->z = z1; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v->nx = nx; v->ny = ny; v->nz = nz; v++;
	v->x = x3; v->y = y3; v->z = z3; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v->nx = nx; v->ny = ny; v->nz = nz; v++;
	v->x = x4; v->y = y4; v->z = z4; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;  v->nx = nx; v->ny = ny; v->nz = nz; v++;

	// repeat last for jump
	v->x = x4; v->y = y4; v->z = z4; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;  v->nx = nx; v->ny = ny; v->nz = nz; v++;

	for (int j=0; j < 8; j++)
		*i++ = ndx++; 
}

void nvDraw::drawBox3D ( Vector3DF p, Vector3DF pa, Vector3DF pb, Matrix4F xform, float r, float g, float b, float a )
{
	Vector3DF q, t[3];
	q    = Vector3DF(pa.x, pa.y, pa.z);			q *= xform;	
	t[0] = Vector3DF(pb.x-pa.x, 0.f, 0.f);			t[0] *= xform;	t[0] -= p;
	t[1] = Vector3DF(0.f, pb.y-pa.y, 0.f);			t[1] *= xform;	t[1] -= p;
	t[2] = Vector3DF(0.f, 0.f, pb.z-pa.z);			t[2] *= xform;	t[2] -= p;
	
	drawLine3D ( q.x, q.y, q.z, q.x+t[0].x, q.y+t[0].y, q.z+t[0].z, r, g, b, a );	q += t[0];
	drawLine3D ( q.x, q.y, q.z, q.x+t[1].x, q.y+t[1].y, q.z+t[1].z, r, g, b, a );	q += t[1];
	drawLine3D ( q.x, q.y, q.z, q.x-t[0].x, q.y-t[0].y, q.z-t[0].z, r, g, b, a );	q -= t[0];
	drawLine3D ( q.x, q.y, q.z, q.x-t[1].x, q.y-t[1].y, q.z-t[1].z, r, g, b, a );	q -= t[1];	

	drawLine3D ( q.x, q.y, q.z, q.x+t[2].x, q.y+t[2].y, q.z+t[2].z, r, g, b, a );	q += t[2];

	drawLine3D ( q.x, q.y, q.z, q.x+t[0].x, q.y+t[0].y, q.z+t[0].z, r, g, b, a );	q += t[0];
	drawLine3D ( q.x, q.y, q.z, q.x+t[1].x, q.y+t[1].y, q.z+t[1].z, r, g, b, a );	q += t[1];
	drawLine3D ( q.x, q.y, q.z, q.x-t[0].x, q.y-t[0].y, q.z-t[0].z, r, g, b, a );	q -= t[0];
	drawLine3D ( q.x, q.y, q.z, q.x-t[1].x, q.y-t[1].y, q.z-t[1].z, r, g, b, a );	q -= t[1];	
}


void nvDraw::drawBox3DXform ( Vector3DF b1, Vector3DF b2, Vector3DF clr, Matrix4F& xform )
{
	Vector3DF p[8];
	p[0].Set ( b1.x, b1.y, b1.z );	p[0] *= xform;
	p[1].Set ( b2.x, b1.y, b1.z );  p[1] *= xform;
	p[2].Set ( b2.x, b1.y, b2.z );  p[2] *= xform;
	p[3].Set ( b1.x, b1.y, b2.z );  p[3] *= xform;

	p[4].Set ( b1.x, b2.y, b1.z );	p[4] *= xform;
	p[5].Set ( b2.x, b2.y, b1.z );  p[5] *= xform;
	p[6].Set ( b2.x, b2.y, b2.z );  p[6] *= xform;
	p[7].Set ( b1.x, b2.y, b2.z );  p[7] *= xform;

	drawLine3D ( p[0].x, p[0].y, p[0].z, p[1].x, p[1].y, p[1].z, clr.x, clr.y, clr.z, 1 );
	drawLine3D ( p[1].x, p[1].y, p[1].z, p[2].x, p[2].y, p[2].z, clr.x, clr.y, clr.z, 1 );
	drawLine3D ( p[2].x, p[2].y, p[2].z, p[3].x, p[3].y, p[3].z, clr.x, clr.y, clr.z, 1 );
	drawLine3D ( p[3].x, p[3].y, p[3].z, p[0].x, p[0].y, p[0].z, clr.x, clr.y, clr.z, 1 );

	drawLine3D ( p[4].x, p[4].y, p[4].z, p[5].x, p[5].y, p[5].z, clr.x, clr.y, clr.z, 1 );
	drawLine3D ( p[5].x, p[5].y, p[5].z, p[6].x, p[6].y, p[6].z, clr.x, clr.y, clr.z, 1 );
	drawLine3D ( p[6].x, p[6].y, p[6].z, p[7].x, p[7].y, p[7].z, clr.x, clr.y, clr.z, 1 );
	drawLine3D ( p[7].x, p[7].y, p[7].z, p[4].x, p[4].y, p[4].z, clr.x, clr.y, clr.z, 1 );

	drawLine3D ( p[0].x, p[0].y, p[0].z, p[4].x, p[4].y, p[4].z, clr.x, clr.y, clr.z, 1 );
	drawLine3D ( p[1].x, p[1].y, p[1].z, p[5].x, p[5].y, p[5].z, clr.x, clr.y, clr.z, 1 );
	drawLine3D ( p[2].x, p[2].y, p[2].z, p[6].x, p[6].y, p[6].z, clr.x, clr.y, clr.z, 1 );
	drawLine3D ( p[3].x, p[3].y, p[3].z, p[7].x, p[7].y, p[7].z, clr.x, clr.y, clr.z, 1 );
}

void nvDraw::drawBox3D(Vector3DF p, Vector3DF q, float r, float g, float b, float a)
{
	drawLine3D (p.x, q.y, p.z, q.x, q.y, p.z, r, g, b, a);		// y++ face
	drawLine3D (q.x, q.y, p.z, q.x, q.y, q.z, r, g, b, a);
	drawLine3D (q.x, q.y, q.z, p.x, q.y, q.z, r, g, b, a);
	drawLine3D (p.x, q.y, q.z, p.x, q.y, p.z, r, g, b, a);

	drawLine3D (p.x, p.y, p.z, q.x, p.y, p.z, r, g, b, a);		// y-- face
	drawLine3D (q.x, p.y, p.z, q.x, p.y, q.z, r, g, b, a);
	drawLine3D (q.x, p.y, q.z, p.x, p.y, q.z, r, g, b, a);
	drawLine3D (p.x, p.y, q.z, p.x, p.y, p.z, r, g, b, a);

	drawLine3D(p.x, p.y, p.z, p.x, q.y, p.z, r, g, b, a);		// verticals
	drawLine3D(p.x, p.y, q.z, p.x, q.y, q.z, r, g, b, a);
	drawLine3D(q.x, p.y, q.z, q.x, q.y, q.z, r, g, b, a);
	drawLine3D(q.x, p.y, p.z, q.x, q.y, p.z, r, g, b, a);
}


void nvDraw::drawCube3D(Vector3DF p, Vector3DF q, float r, float g, float b, float a)
{
	drawFace3D(p.x, q.y, p.z,  q.x, q.y, p.z,  q.x, q.y, q.z,  p.x, q.y, q.z,  0,  1, 0, r, g, b, a);
	drawFace3D(p.x, p.y, p.z,  q.x, p.y, p.z,  q.x, p.y, q.z,  p.x, p.y, q.z,  0, -1, 0, r, g, b, a);

	drawFace3D(p.x, p.y, q.z,  q.x, p.y, q.z,  q.x, q.y, q.z,  p.x, q.y, q.z,  0, 0,  1, r, g, b, a);
	drawFace3D(p.x, p.y, p.z,  q.x, p.y, p.z,  q.x, q.y, p.z,  p.x, q.y, p.z,  0, 0, -1, r, g, b, a);

	drawFace3D(q.x, p.y, p.z,  q.x, q.y, p.z,  q.x, q.y, q.z,  q.x, p.y, q.z,  1, 0, 0, r, g, b, a);
	drawFace3D(p.x, p.y, p.z,  p.x, q.y, p.z,  p.x, q.y, q.z,  p.x, p.y, q.z,  -1, 0, 0, r, g, b, a);
}

void nvDraw::end3D ()
{
	mCurrSet = 0x0;	
}

int nvDraw::start2D ( bool bStatic )
{
	if ( mVAO == 65535 ) {
		dbgprintf ( "ERROR: nv_gui was not initialized. Must call init2D.\n" );
	}
	nvSet new_set;
	nvSet* s;
	
	if ( bStatic ) {		
		mStatic.push_back ( new_set );		
		s = &mStatic[ mStatic.size()-1 ]; 		
		for (int n=0; n < GRP_MAX; n++ ) s->mGeom[n] = 0x0;		
		mCurrSet = s;		
	} else {
		int curr = mDynNum;					
		if ( mDynNum >= mDynamic.size() ) {			
			mDynamic.push_back ( new_set );						
			mDynNum = (int) mDynamic.size();
			s = &mDynamic[curr];	
			for (int n=0; n < GRP_MAX; n++ ) s->mGeom[n] = 0x0;			
		} else {		
			mDynNum++;
			s = &mDynamic[curr];	
		}		
		mCurrSet = s;
	}
	for (int n=0; n < GRP_MAX; n++ ) {
		s->mNum[n] = 0;		s->mMax[n] = 0;
		s->mNumI[n] = 0;	s->mMaxI[n] = 0;	s->mIdx[n] = 0;		
	}
	if ( mWidth==-1 ) {
		SetMatrixView ( *s, mModelMtx, mViewMtx, mProjMtx, (float) mZFactor );
	} else {
		SetDefaultView ( *s, mWidth, mHeight, (float) mZFactor );
	}

	return mCurr;
}

void nvDraw::setOrder2D ( bool zt, float zfactor )
{	
	if ( zt == false ) zfactor = 1;
	mZFactor = zfactor;
}
void nvDraw::setOrder3D ( bool z )
{
	mZFactor = z;
	mCurrSet->zfactor = z ? 1 : 0;
}

void nvDraw::SetDefaultView ( nvSet& s, float w, float h, float zf )
{
	s.zfactor = zf;

	Matrix4F proj, view, model;
	view.Scale ( 2.0/w, -2.0/h, s.zfactor );	
	model.Translate ( -w/2.0, -h/2.0, 0 );
	view *= model;	
	model.Identity ();
	proj.Identity ();

	memcpy ( s.model, model.GetDataF(), 16 * sizeof(float) );
	memcpy ( s.view,  view.GetDataF(), 16 * sizeof(float) );
	memcpy ( s.proj,  proj.GetDataF(), 16 * sizeof(float) );	
}
void nvDraw::SetMatrixView ( nvSet& s, float* model, float* view, float* proj, float zf )
{
	s.zfactor = zf;
	memcpy ( s.model, model, 16 * sizeof(float) );
	memcpy ( s.view,  view, 16 * sizeof(float) );
	memcpy ( s.proj,  proj, 16 * sizeof(float) );	
}

void nvDraw::end2D ()
{
	mCurrSet = 0x0;	
}

nvVert* nvDraw::allocGeom ( int cnt, int grp, nvSet* s, int& ndx )
{
	if ( s == 0x0 ) {
		dbgprintf ( "ERROR: draw must be inside of draw2D/end2D or draw3D/end3D\n" );
	}
	if ( s->mNum[grp] + cnt >= s->mMax[grp] ) {		
		xlong new_max = s->mMax[grp] * 8 + cnt;		
		//	dbgprintf  ( "allocGeom: expand, %lu\n", new_max );
		nvVert* new_data = (nvVert*) malloc ( new_max*sizeof(nvVert) );
		if ( s->mGeom[grp] != 0x0 ) {
			memcpy ( new_data, s->mGeom[grp], s->mNum[grp]*sizeof(nvVert) );
			free ( s->mGeom[grp] );
		}
		s->mGeom[grp] = new_data;
		s->mMax[grp] = new_max;
	}
	nvVert* start = s->mGeom[grp] + s->mNum[grp];
	ndx = (int) s->mNum[grp];
	s->mNum[grp] += cnt;		
	return start;
}

// allocate image
//
nvVert* nvDraw::allocImg ( int img_glid, int grp, nvSet* s, int& ndx )
{
	int stride = 4;	

	uint* i = allocIdx ( 1, GRP_IMG, s );
	*i = img_glid;
	
	if ( s->mNum[grp] + stride >= s->mMax[grp] ) {
		xlong new_max = s->mMax[grp] * 2 + stride;
		nvVert* new_data = (nvVert*) malloc ( new_max*sizeof(nvVert) );
		if ( s->mGeom[grp] != 0x0 ) {			
			memcpy ( new_data, s->mGeom[grp], s->mNum[grp]*sizeof(nvVert) );			
			free ( s->mGeom[grp] );			
		}
		s->mGeom[grp] = new_data;		
		s->mMax[grp] = new_max;		
	}
	nvVert* start = s->mGeom[grp] + s->mNum[grp];	
	ndx = (int) s->mNum[grp];
	s->mNum[grp] += stride;					// 4 verts per img 
	return start;
}

uint* nvDraw::allocIdx ( int cnt, int grp, nvSet* s )
{
	if ( s->mNumI[grp] + cnt >= s->mMaxI[grp] ) {		
		xlong new_max = s->mMaxI[grp] * 8 + cnt;
		// dbgprintf  ( "allocIdx: expand, %lu\n", new_max );
		uint* new_data = (uint*) malloc ( new_max*sizeof(uint) );
		if ( s->mIdx[grp] != 0x0 ) {
			memcpy ( new_data, s->mIdx[grp], s->mNumI[grp]*sizeof(int) );
			delete s->mIdx[grp];
		}
		s->mIdx[grp] = new_data;
		s->mMaxI[grp] = new_max;
	}
	uint* start = s->mIdx[grp] + s->mNumI[grp];		
	s->mNumI[grp] += cnt;
	return (uint*) start;
}

void nvDraw::remove2D ( int id )
{


}
void nvDraw::drawLine ( float x1, float y1, float x2, float y2, float r, float g, float b, float a )
{
#ifdef DEBUG_UTIL
	dbgprintf  ( "Draw line.\n" );
#endif
	int ndx;
	nvVert* v = allocGeom ( 2, GRP_LINES, mCurrSet, ndx );

	v->x = x1; v->y = y1; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;
	v->x = x2; v->y = y2; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	
}
void nvDraw::drawRect ( float x1, float y1, float x2, float y2, float r, float g, float b, float a )
{
#ifdef DEBUG_UTIL
	dbgprintf  ( "Draw rect.\n" );
#endif
	int ndx;
	nvVert* v = allocGeom ( 8, GRP_LINES, mCurrSet, ndx );

	v->x = x1; v->y = y1; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0; v++;
	v->x = x2; v->y = y1; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;
	v->x = x2; v->y = y1; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0; v++;
	v->x = x2; v->y = y2; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;
	v->x = x2; v->y = y2; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0; v++;
	v->x = x1; v->y = y2; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;
	v->x = x1; v->y = y2; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0; v++;
	v->x = x1; v->y = y1; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	
}

void nvDraw::drawImg ( int img_glid, float x1, float y1, float x2, float y2, float r, float g, float b, float a )
{
	if ( img_glid==-1 ) return;
	int ndx;
	nvVert* v = allocImg ( img_glid, GRP_IMG, mCurrSet, ndx );	

	v->x = x1; v->y = y1; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;
	v->x = x2; v->y = y1; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 1; v->ty = 0; 	v++;
	v->x = x2; v->y = y2; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 1; v->ty = 1;	v++;
	v->x = x1; v->y = y2; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 1;	v++;	
}

void nvDraw::drawPnts3D ( Vector4DF* pnts, Vector4DF* clrs, int pnts_num )
{
	int ndx;
	nvVert* v = allocGeom ( pnts_num, GRP_POINTS, mCurrSet, ndx );
	memset ( v, 0, sizeof(nvVert)*pnts_num );
	
	for (int n=0; n < pnts_num; n++) {
		memcpy ( &v->x, pnts, sizeof(Vector3DF) );		// x,y,z
		memcpy ( &v->r, clrs, sizeof(Vector4DF) );		// r,g,b,a
		v->a = 1.0;
		pnts++;
		clrs++;
		v++;
	}
}

void nvDraw::drawImg3D ( int img_glid, Vector3DF p1, Vector3DF p2, Vector3DF p3, Vector3DF p4 )
{
	if ( img_glid==-1 ) return;
	int ndx;
	nvVert* v = allocImg ( img_glid, GRP_IMG, mCurrSet, ndx );
	float r,g,b,a;
	r=g=b=a=1.0f;

	v->x = p1.x; v->y = p1.y; v->z = p1.z; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;
	v->x = p2.x; v->y = p2.y; v->z = p2.z; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 1; v->ty = 0;	v++;
	v->x = p3.x; v->y = p3.y; v->z = p3.z; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 1; v->ty = 1; 	v++;	
	v->x = p4.x; v->y = p4.y; v->z = p4.z; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 1;	v++;
}

void nvDraw::drawFill ( float x1, float y1, float x2, float y2, float r, float g, float b, float a )
{
#ifdef DEBUG_UTIL
	dbgprintf  ( "Draw fill.\n" );
#endif
	int ndx;
	nvVert* v = allocGeom ( 6, GRP_TRI, mCurrSet, ndx );
	uint* i = allocIdx ( 6, GRP_TRI, mCurrSet );

	// repeat first for jump
	v->x = x2; v->y = y1; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;

	// two triangles (as strip, must start top right)
	v->x = x2; v->y = y1; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;
	v->x = x2; v->y = y2; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 1; v->ty = 0; 	v++;
	v->x = x1; v->y = y1; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 1;	v++;
	v->x = x1; v->y = y2; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 1; v->ty = 1;	v++;	

	// repeat last for jump
	v->x = x1; v->y = y2; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 1; v->ty = 1;	v++;	

	*i++ = ndx++; *i++ = ndx++; *i++ = ndx++; *i++ = ndx++; *i++ = ndx++; *i++ = ndx++; 
}
void nvDraw::drawTri ( float x1, float y1, float x2, float y2, float x3, float y3, float r, float g, float b, float a )
{
#ifdef DEBUG_UTIL
	dbgprintf  ( "Draw tri.\n" );
#endif
	int ndx;
	nvVert* v = allocGeom ( 3, GRP_TRI, mCurrSet, ndx );
	uint* i = allocIdx ( 4, GRP_TRI, mCurrSet );

	v->x = x1; v->y = y1; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;
	v->x = x2; v->y = y2; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;
	v->x = x3; v->y = y3; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;		
	*i++ = ndx++; *i++ = ndx++; *i++ = ndx++; *i++ = IDX_NULL;
}

void nvDraw::drawCircle ( float x1, float y1, float radius, float r, float g, float b, float a )
{
#ifdef DEBUG_UTIL
	dbgprintf  ( "Draw circle.\n" );
#endif
	int ndx;
	nvVert* v = allocGeom ( 62, GRP_LINES, mCurrSet, ndx );	
	
	float dx, dy, dxl, dyl;	
	dxl = (float) cos( (0/31.0)*3.141592*2.0 )*radius;
	dyl = (float) sin( (0/31.0)*3.141592*2.0 )*radius;
	for (int n=1; n < 32; n++ ) {
		dx = (float) cos( (n/31.0)*3.141592*2.0 )*radius;
		dy = (float) sin( (n/31.0)*3.141592*2.0 )*radius;
		v->x = x1+dxl; v->y = y1+dyl; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;
		v->x = x1+dx; v->y = y1+dy; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;
		dxl = dx; dyl = dy;
	}		
}
void nvDraw::drawCircleDash ( float x1, float y1, float radius, float r, float g, float b, float a )
{
	int ndx;
	nvVert* v = allocGeom ( 32, GRP_LINES, mCurrSet, ndx );	
	
	float dx, dy;		
	for (int n=0; n < 32; n++ ) {
		dx = (float) cos( (n/31.0)*3.141592*2.0 )*radius;
		dy = (float) sin( (n/31.0)*3.141592*2.0 )*radius;		
		v->x = x1+dx; v->y = y1+dy; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;		
	}		
}
void nvDraw::drawCircleFill ( float x1, float y1, float radius, float r, float g, float b, float a )
{
	int ndx;
	nvVert* v = allocGeom ( 66, GRP_TRI, mCurrSet, ndx );
	uint* i = allocIdx ( 66, GRP_TRI, mCurrSet );
	
	float dx, dy;

	v->x = x1; v->y = y1; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;
	*i++ = ndx++;
	for (int n=0; n < 32; n++ ) {				// 64 points
		dx = (float) cos( (n/31.0)*3.141592*2.0 )*radius;
		dy = (float) sin( (n/31.0)*3.141592*2.0 )*radius;
		v->x = x1; v->y = y1; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;
		*i++ = ndx++;
		v->x = x1+dx; v->y = y1+dy; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;
		*i++ = ndx++;
	}
	v->x = x1+dx; v->y = y1+dy; v->z = 0; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;
	*i++ = ndx++;
	// 66 total points
}

void nvDraw::drawText3D ( Camera3D* cam, float x1, float y1, float z1, char* msg, float r, float g, float b, float a )
{
	int len = (int) strlen ( msg );
	if ( len == 0 ) return;
	if ( len > 128 ) { dbgprintf ( "ERROR: Drawing text over 128 chars.\n" ); exit(-3); }
	int ndx;
	nvVert* v = allocGeom ( len*6, GRP_TRITEX, mCurrSet, ndx );
	uint* i = allocIdx ( len*6, GRP_TRITEX, mCurrSet );

	int glyphHeight = mGlyphInfos.pix.ascent + mGlyphInfos.pix.descent + mGlyphInfos.pix.linegap;
	float lX = 0;
	float lY = 0;
	const char* c = msg;
	int cnt = 0;
	
	// TextScale = height of font in pixels
	float textSz = mTextScale / glyphHeight;	// glyph scale
	float textStartPy = textSz;					// start location in pixels
	
	while (*c != '\0' && cnt < len ) {
		if ( *c == '\n' ) {
			lX = 0;
			lY += mTextScale;			
		} else if ( *c >=0 && *c <= 128 ) {
			GlyphInfo& gly = mGlyphInfos.glyphs[*c];
			float pX = x1 + lX + gly.pix.offX*textSz;
			float pY = y1 + lY - gly.pix.offY*textSz; 
			float pZ = z1; 
			float pW = gly.pix.width*textSz;
			float pH = -gly.pix.height*textSz;
	
			// GRP_TRITEX is a triangle strip!
			// repeat first point (jump), zero alpha
			v->x = pX;		v->y = pY+pH;	v->z = pZ;		v->r = r; v->g = g; v->b = b; v->a = 0;		v->tx = gly.norm.u; v->ty = gly.norm.v;	v++;
			// four corners of glyph, *NOTE*: Negative alpha indicates to shader we are drawing a font (not an image)			
			v->x = pX;		v->y = pY+pH;	v->z = pZ; 		v->r = r; v->g = g; v->b = b; v->a = -a;	v->tx = gly.norm.u; v->ty = gly.norm.v;	v++;
			v->x = pX;		v->y = pY;		v->z = pZ;		v->r = r; v->g = g; v->b = b; v->a = -a;	v->tx = gly.norm.u; v->ty = gly.norm.v + gly.norm.height;	v++;			
			v->x = pX+pW;	v->y = pY+pH;	v->z = pZ;		v->r = r; v->g = g; v->b = b; v->a = -a;	v->tx = gly.norm.u + gly.norm.width; v->ty = gly.norm.v;	v++;			
			v->x = pX+pW;	v->y = pY;		v->z = pZ;		v->r = r; v->g = g; v->b = b; v->a = -a;	v->tx = gly.norm.u + gly.norm.width; v->ty = gly.norm.v + gly.norm.height;	v++;
			// repeat last point (jump), zero alpha
			v->x = pX+pW;	v->y = pY;		v->z = pZ;		v->r = r; v->g = g; v->b = b; v->a = 0;		v->tx = gly.norm.u + gly.norm.width; v->ty = gly.norm.v + gly.norm.height;	v++;

			*i++ = ndx++; *i++ = ndx++;	*i++ = ndx++; *i++ = ndx++; *i++ = ndx++; *i++ = ndx++;			
			lX += (gly.pix.advance + mTextKern ) * textSz;
            lY += 0;
			cnt++;
		}
		c++;
	}

	// clear remainder of VBO entries
	for (int n=cnt; n < len; n++ ) {
		memset ( v, 0, 6*sizeof(nvVert) ); v += 6;		
		*i++ = ndx++; *i++ = ndx++;	*i++ = ndx++; *i++ = ndx++; *i++ = ndx++; *i++ = ndx++;
	}
}

void nvDraw::drawText ( float x1, float y1, char* msg, float r, float g, float b, float a )
{
	int len = (int) strlen ( msg );
	if ( len == 0 ) 
		return;

#ifdef DEBUG_UTIL
	if ( len > 128 ) {
		MessageBox ( NULL, "Error: drawText over 128 chars\n", "drawText", MB_OK );
		exit(-3);
	}
#endif
	int ndx;
	nvVert* v = allocGeom ( len*6, GRP_TRITEX, mCurrSet, ndx );
	uint* i = allocIdx ( len*6, GRP_TRITEX, mCurrSet );

	int glyphHeight = mGlyphInfos.pix.ascent + mGlyphInfos.pix.descent + mGlyphInfos.pix.linegap;
	float lX = x1;
	float lY = y1;
	float lLinePosX = x1;
	float lLinePosY = y1;
	const char* c = msg;
	int cnt = 0;
	
	// TextScale = height of font in pixels
	float textSz = mTextScale / glyphHeight;	// glyph scale
	float textStartPy = textSz;					// start location in pixels
	
	while (*c != '\0' && cnt < len ) {
		if ( *c == '\n' ) {
			lX = lLinePosX;
			lLinePosY += mTextScale;
			lY = lLinePosY;
		} else if ( *c >=0 && *c <= 128 ) {
			GlyphInfo& gly = mGlyphInfos.glyphs[*c];
			float pX = lX + gly.pix.offX*textSz;
			float pY = lY + gly.pix.offY*textSz; 
			float pW = gly.pix.width*textSz;
			float pH = gly.pix.height*textSz;
	
			// GRP_TRITEX is a triangle strip!
			// repeat first point (jump), zero alpha
			v->x = pX;		v->y = pY+pH;	v->z = 0;		v->r = r; v->g = g; v->b = b; v->a = 0;		v->tx = gly.norm.u; v->ty = gly.norm.v;	v++;

			// four corners of glyph, *NOTE*: Negative alpha indicates to shader we are drawing a font (not an image)			
			v->x = pX;		v->y = pY+pH;	v->z = 0; 		v->r = r; v->g = g; v->b = b; v->a = -a;	v->tx = gly.norm.u; v->ty = gly.norm.v;	v++;
			v->x = pX;		v->y = pY ;		v->z = 0;		v->r = r; v->g = g; v->b = b; v->a = -a;	v->tx = gly.norm.u; v->ty = gly.norm.v + gly.norm.height;	v++;			
			v->x = pX+pW;	v->y = pY+pH;	v->z = 0;		v->r = r; v->g = g; v->b = b; v->a = -a;	v->tx = gly.norm.u + gly.norm.width; v->ty = gly.norm.v;	v++;			
			v->x = pX+pW;	v->y = pY ;		v->z = 0; 		v->r = r; v->g = g; v->b = b; v->a = -a;	v->tx = gly.norm.u + gly.norm.width; v->ty = gly.norm.v + gly.norm.height;	v++;

			// repeat last point (jump), zero alpha
			v->x = pX+pW;	v->y = pY;	v->z = 0;			v->r = r; v->g = g; v->b = b; v->a = 0;		v->tx = gly.norm.u + gly.norm.width; v->ty = gly.norm.v + gly.norm.height;	v++;

			*i++ = ndx++; *i++ = ndx++;	*i++ = ndx++; *i++ = ndx++; *i++ = ndx++; *i++ = ndx++;
			
			lX += (gly.pix.advance + mTextKern ) * textSz;
            lY += 0;
			cnt++;
		}
		c++;
	}

	// clear remainder of VBO entries
	for (int n=cnt; n < len; n++ ) {
		memset ( v, 0, 6*sizeof(nvVert) ); v += 6;		
		*i++ = ndx++; *i++ = ndx++;	*i++ = ndx++; *i++ = ndx++; *i++ = ndx++; *i++ = ndx++;
	}
}

void nvDraw::getTextSize ( const char* msg, float& sx, float& sy)
{
	sx = 0;
	sy = 0;
	
	int len = (int) strlen ( msg );
	if ( len == 0 ) return;
	int glyphHeight = mGlyphInfos.pix.ascent + mGlyphInfos.pix.descent + mGlyphInfos.pix.linegap;		
	float lLinePosX=0, lLinePosY=0;
	const char* c = msg;
	int cnt = 0;	
	// TextScale = height of font in pixels
	float textSz = mTextScale / glyphHeight;	// glyph scale
	float textStartPy = textSz;					// start location in pixels	
	while (*c != '\0' && cnt < len ) {
		if ( *c == '\n' ) {			
			sy += mTextScale;
		} else if ( *c >=0 && *c <= 128 ) {
			GlyphInfo& gly = mGlyphInfos.glyphs[*c];			
			sx += (gly.pix.advance + mTextKern) * textSz ;            
			cnt++;
		}
		c++;
	}
}
void nvDraw::getGlyphSize ( const char c, float& sx, float& sy)
{
	if ( c <=0 || c > 128 ) {sx = 0; sy = 0; return;}
	float textSz = mTextScale / (mGlyphInfos.pix.ascent + mGlyphInfos.pix.descent + mGlyphInfos.pix.linegap);
	GlyphInfo &gly = mGlyphInfos.glyphs[c];			// find glyph
	sx = (gly.pix.advance + mTextKern) * textSz;
	sy = mTextScale;
}

void nvDraw::CreateSColor ()
{
	#ifdef USE_DX

		// DirectX - Create shaders
		CHAR* g_strVS = 
			"cbuffer MatrixBuffer : register( b0 ) { row_major matrix Model; }\n"
			"cbuffer MatrixBuffer : register( b1 ) { row_major matrix View; }\n"
			"cbuffer MatrixBuffer : register( b2 ) { row_major matrix Proj; }\n"			
			"\n"
			"struct VS_IN { \n"
			"   float3 pos:POSITION; float4 clr:COLOR; float2 tex:TEXCOORD;\n"			
			//"   matrix instmodel: WORLDVIEW; \n"
			"}; \n"
			"struct VS_OUT { float4 pos:SV_POSITION; float4 color:COLOR; float2 tex:TEXCOORD0; }; \n"			
			"VS_OUT VS (VS_IN input, unsigned int InstID : SV_InstanceID ) { \n"
			"   VS_OUT output = (VS_OUT) 0;\n"			
			"   output.color = input.clr; \n"
			"   output.tex = input.tex;\n"
			"   output.pos = mul ( mul ( mul ( float4(input.pos,1), Model ), View ), Proj );\n"			
			"   return output;\n"
			"}\n";

		 CHAR *g_strPS = 			
			"Texture2D intex;  \n"
			"SamplerState sampleLinear { Filter = MIN_MAG_MIP_LINEAR; AddressU = Wrap; AddressV = Wrap; };\n"
			"struct PS_IN { float4 pos:SV_POSITION; float4 color:COLOR; float2 tex:TEXCOORD0; }; \n"
			"float4 PS ( PS_IN input ) : SV_Target\n"
			"{\n"
			//"    return  input.color;\n"
			"    return input.color * float4( 1, 1, 1, intex.Sample ( sampleLinear, input.tex ).x) ;\n"
			"}\n";

		DWORD dwShaderFlags = D3D10_SHADER_ENABLE_STRICTNESS;
		#ifdef _DEBUG
			dwShaderFlags |= D3D10_SHADER_DEBUG;
		#endif
		ID3D10Blob* pBlobVS = NULL;
		ID3D10Blob* pBlobError = NULL;
		ID3D10Blob* pBlobPS = NULL;

		// Create vertex shader
		HRESULT hr = D3DCompile( g_strVS, lstrlenA( g_strVS ) + 1, "VS", NULL, NULL, "VS", "vs_4_0", dwShaderFlags, 0, &pBlobVS, &pBlobError );
		checkSHADER ( hr, pBlobError );	
		checkHR ( g_pDevice->CreateVertexShader( pBlobVS->GetBufferPointer(), pBlobVS->GetBufferSize(), NULL, &mVS ), "CreateVertexShader" );
    	
		// Create pixel shader
		hr = D3DCompile( g_strPS, lstrlenA( g_strPS ) + 1, "PS", NULL, NULL, "PS", "ps_4_0", dwShaderFlags, 0, &pBlobPS, &pBlobError ) ;		
		checkSHADER ( hr, pBlobError );
		checkHR ( g_pDevice->CreatePixelShader( pBlobPS->GetBufferPointer(), pBlobPS->GetBufferSize(), NULL, &mPS ), "CreatePixelShader" );
		
		// Create input-assembler layout
		D3D11_INPUT_ELEMENT_DESC vs_layout[] =
		{
			{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT,		0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
			{ "COLOR",    0, DXGI_FORMAT_R32G32B32A32_FLOAT,	1, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
			{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,			2, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		};
		UINT numElements = sizeof( vs_layout ) / sizeof( vs_layout[0] );		
		checkHR ( g_pDevice->CreateInputLayout( vs_layout, numElements, pBlobVS->GetBufferPointer(), pBlobVS->GetBufferSize(), &mLO ), "CreateInputLayout"  );

	#else
		// OpenGL - Create shaders
		char buf[16384];
		int len = 0;
		//checkGL( "Start shaders" );

		// OpenGL 4.2 Core
		// -- Cannot use hardware lighting pipeline (e.g. glLightfv, glMaterialfv)
		GLuint vs = glCreateShader(GL_VERTEX_SHADER);
		GLchar const * vss =
			"#version 300 es\n"
			"#extension GL_ARB_explicit_attrib_location : enable\n"
			"\n"
			"layout(location = 0) in vec3 inVertex;\n"
			"layout(location = 1) in vec4 inColor;\n"
			"layout(location = 2) in vec2 inTexCoord;\n"
			"out vec3 position;\n"		
			"out vec4 color;\n"				
			"out vec2 texcoord;\n"
			"uniform mat4 modelMatrix;\n"
			"uniform mat4 viewMatrix;\n"
			"uniform mat4 projMatrix;\n"
			"\n"
			"void main()\n"
			"{\n"		
			"	 position = (modelMatrix * vec4(inVertex,1)).xyz;\n"
			"    color = inColor;\n"
			"    texcoord = inTexCoord;\n"
			"    gl_Position = projMatrix * viewMatrix * modelMatrix * vec4(inVertex,1);\n"
			"}\n"
		;
		glShaderSource(vs, 1, &vss, 0);
		glCompileShader(vs);
		glGetShaderInfoLog ( vs, 16384, (GLsizei*) &len, buf );
		if ( len > 0 ) dbgprintf  ( "ERROR SColor vert: %s\n", buf );
		checkGL( "Compile vertex shader" );

		GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
		GLchar const * fss =
			"#version 300 es\n"
			"\n"					
			"  precision mediump float;\n"
			"  precision mediump int;\n"			
			"uniform sampler2D imgTex;\n"
			"in vec3 position;\n"
			"in vec4 color;\n"
			"in vec2 texcoord;\n"
			"layout(location = 0) out vec4 outColor;\n"
			"\n"
			"void main()\n"
			"{\n"					
			"    vec4 imgclr = texture(imgTex, texcoord);\n"
			"    outColor = (color.w > 0.f) ? color * imgclr : vec4( color.x, color.y, color.z, -color.w*imgclr.x); \n"
			"}\n"
		;


		glShaderSource(fs, 1, &fss, 0);
		glCompileShader(fs);
		glGetShaderInfoLog ( fs, 16384, (GLsizei*) &len, buf );
		if ( len > 0 ) dbgprintf  ( "ERROR SColor frag: %s\n", buf );
		checkGL( "Compile fragment shader" );

		mSH[ SCOLOR ] = glCreateProgram();
		glAttachShader( mSH[ SCOLOR ], vs);
		glAttachShader( mSH[ SCOLOR ], fs);
		checkGL( "Attach program" );
		glLinkProgram( mSH[SCOLOR] );
		checkGL( "Link program" );
		glUseProgram( mSH[SCOLOR] );
		checkGL( "Use program" );

		mProj[SCOLOR] =		glGetUniformLocation ( mSH[SCOLOR], "projMatrix" );
        mModel[SCOLOR] =	glGetUniformLocation ( mSH[SCOLOR], "modelMatrix" );
        mView[SCOLOR] =		glGetUniformLocation ( mSH[SCOLOR], "viewMatrix" );
        mTex[SCOLOR] =		glGetUniformLocation ( mSH[SCOLOR], "imgTex" );		

		checkGL( "Get Shader Matrices" );
	#endif
}

void nvDraw::CreateSInst ()
{

	// OpenGL - Create shaders
	char buf[16384];
	int len = 0;
	checkGL( "Start shaders" );

	// OpenGL 4.2 Core
	// -- Cannot use hardware lighting pipeline (e.g. glLightfv, glMaterialfv)
	GLuint vs = glCreateShader(GL_VERTEX_SHADER);
	GLchar const * vss =
		"#version 300 es\n"
		"  precision mediump float;\n"
		"  precision mediump int;\n"					
		"layout(location = 0) in vec3   inVertex;		// geometry attributes\n"
		"layout(location = 1) in vec4	inColor;		\n"
		"layout(location = 2) in vec2	inTexCoord;	\n"
		"layout(location = 3) in vec3	instPos1;	// instance attributes\n"
		"layout(location = 4) in vec4	instClr;   \n"
		"layout(location = 5) in vec2	instUV;    \n"
		"layout(location = 6) in vec3	instPos2;  \n"
		"uniform mat4			viewMatrix;			\n"
		"uniform mat4			projMatrix;			\n"
		"uniform mat4			modelMatrix;		\n"
		"out vec4 vworldpos;		\n"
		"out vec4 vcolor;			\n"
		"flat out vec3 vtexcoord1;	\n"
		"flat out vec3 vtexcoord2;	\n"
		"flat out mat4 vxform;		\n"
		"void main() {\n"
		"  int inst = gl_InstanceID;	\n"
		"  vxform = mat4 ( instPos2.x-instPos1.x, 0, 0, 0, \n"
		"			0, instPos2.y-instPos1.y, 0, 0,   \n"
		"			0, 0, instPos2.z-instPos1.z, 0,   \n"
		"			instPos1.x, instPos1.y, instPos1.z, 1);	\n"
		"  vworldpos = vxform * vec4(inVertex,1);\n"
		"  vcolor = instClr;\n"
	    "  gl_Position = projMatrix * viewMatrix * vworldpos;\n"
        "}\n"
	;
	glShaderSource(vs, 1, &vss, 0);
	glCompileShader(vs);
	glGetShaderInfoLog ( vs, 16384, (GLsizei*) &len, buf );
	if ( len > 0 ) dbgprintf  ( "ERROR ShaderInst vert: %s\n", buf );
	checkGL( "Compile vertex shader" );

	GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
	GLchar const * fss =
		"#version 300 es\n"
		"  precision mediump float;\n"
		"  precision mediump int;\n"					
		"in vec4 vworldpos; \n"
		"in vec4 vcolor; \n"
		"flat in mat4 vxform; \n"
		"flat in vec3 vtexcoord1; \n"
		"flat in vec3 vtexcoord2; \n"
		"out vec4 outColor;\n"
		"void main () {\n"
		"  outColor = vcolor;\n"
		"}\n"
	;
	glShaderSource(fs, 1, &fss, 0);
	glCompileShader(fs);
	glGetShaderInfoLog ( fs, 16384, (GLsizei*) &len, buf );
	if ( len > 0 ) dbgprintf  ( "ERROR ShaderInst frag: %s\n", buf );
	checkGL( "Compile fragment shader" );

	mSH[SINST] = glCreateProgram();
	glAttachShader( mSH[SINST], vs);
	glAttachShader( mSH[SINST], fs);
	checkGL( "Attach program" );
	glLinkProgram( mSH[SINST] );
	checkGL( "Link program" );
	glUseProgram( mSH[SINST] );
	checkGL( "Use program" );

	/*mProj[SINST] =	glGetProgramResourceIndex ( mSH[SINST], GL_UNIFORM, "projMatrix" );
	mModel[SINST] =	glGetProgramResourceIndex ( mSH[SINST], GL_UNIFORM, "modelMatrix" );	
	mView[SINST] =	glGetProgramResourceIndex ( mSH[SINST], GL_UNIFORM, "viewMatrix" );*/

    mProj[SINST] =	glGetUniformLocation ( mSH[SINST], "projMatrix" );
    mModel[SINST] =	glGetUniformLocation ( mSH[SINST], "modelMatrix" );
    mView[SINST] =	glGetUniformLocation ( mSH[SINST], "viewMatrix" );

	checkGL( "Get Shader Matrices" );	
}

void nvDraw::CreateS3D ()
{

	// OpenGL - Create shaders
	char buf[16384];
	int len = 0;
	checkGL( "Start shaders" );

	// OpenGL 4.2 Core
	// -- Cannot use hardware lighting pipeline (e.g. glLightfv, glMaterialfv)
	GLuint vs = glCreateShader(GL_VERTEX_SHADER);
	GLchar const * vss =
			"#version 300 es\n"
			"\n"
			"layout(location = 0) in vec3 inPosition;\n"
			"layout(location = 1) in vec4 inColor;\n"
			"layout(location = 2) in vec2 inTexCoord;\n"
			"layout(location = 3) in vec3 inNorm;\n"
			"out vec3 vpos;\n"		
			"out vec4 vcolor;\n"				
			"out vec2 vtexcoord;\n"
			"out vec3 vnorm;\n"
			"uniform mat4 modelMatrix;\n"
			"uniform mat4 viewMatrix;\n"
			"uniform mat4 projMatrix;\n"
			"\n"
			"void main()\n"
			"{\n"		
			"	 vpos = (modelMatrix * vec4(inPosition, 1.f)).xyz;\n"
			"    vcolor = inColor;\n"
			"    vtexcoord = inTexCoord;\n"
			"	 vnorm = inNorm;\n"
			"    gl_Position = projMatrix * viewMatrix * modelMatrix * vec4(inPosition, 1.f);\n"
			"}\n"
	;
	glShaderSource(vs, 1, &vss, 0);
	glCompileShader(vs);
	glGetShaderInfoLog ( vs, 16384, (GLsizei*) &len, buf );
	if ( len > 0 ) dbgprintf  ( "ERROR Shader3D vert: %s\n", buf );
	checkGL( "Compile vertex shader" );

	GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
	GLchar const * fss =
		"#version 300 es\n"
		"  precision mediump float;\n"
		"  precision mediump int;\n"
		"uniform sampler2D imgTex;\n"
		"in vec3		vpos; \n"
		"in vec4		vcolor; \n"		
		"in vec2		vtexcoord; \n"
		"in vec3		vnorm; \n"		
		"uniform vec3	lightpos; \n"
		"out vec4		outColor;\n"
		"void main () {\n"		
		"    vec4 imgclr = texture(imgTex, vtexcoord);\n"
		"    outColor = (vcolor.w > 0.f) ? vcolor * imgclr: vec4( vcolor.x, vcolor.y, vcolor.z, -vcolor.w*imgclr.x); \n"
		"}\n"
	;

	

	//"   float d = 0.1f + 0.9f * clamp( dot ( vnorm, normalize(lightpos-vpos) ), 0.f, 1.f); \n"
	// "   outColor = vec4(d, d, d, 1) * vcolor;\n"
	// "   outColor = vec4(vnorm.x, vnorm.y, vnorm.z, 1) * vcolor;\n"

	glShaderSource(fs, 1, &fss, 0);
	glCompileShader(fs);
	glGetShaderInfoLog ( fs, 16384, (GLsizei*) &len, buf );
	if ( len > 0 ) dbgprintf  ( "ERROR Shader3D frag: %s\n", buf );
	checkGL( "Compile fragment shader" );

	mSH[S3D] = glCreateProgram();
	glAttachShader( mSH[S3D], vs);
	glAttachShader( mSH[S3D], fs);
	checkGL( "Attach program" );
	glLinkProgram( mSH[S3D] );
	checkGL( "Link program" );
	glUseProgram( mSH[S3D] );
	checkGL( "Use program" );

	/*mProj[S3D] =	glGetProgramResourceIndex ( mSH[S3D], GL_UNIFORM, "projMatrix" );
	mModel[S3D] =	glGetProgramResourceIndex ( mSH[S3D], GL_UNIFORM, "modelMatrix" );	
	mView[S3D] =	glGetProgramResourceIndex ( mSH[S3D], GL_UNIFORM, "viewMatrix" );	
	mLight[S3D] =	glGetProgramResourceIndex ( mSH[S3D], GL_UNIFORM, "lightpos" );*/

    mProj[S3D] =	glGetUniformLocation ( mSH[S3D], "projMatrix" );
    mModel[S3D] =	glGetUniformLocation ( mSH[S3D], "modelMatrix" );
    mView[S3D] =	glGetUniformLocation ( mSH[S3D], "viewMatrix" );
    mLight[S3D] =	glGetUniformLocation ( mSH[S3D], "lightpos" );
	mTex[S3D] =		glGetUniformLocation ( mSH[S3D], "imgTex" );

	checkGL( "Get Shader Matrices" );	
}

void nvDraw::CreateSPnt ()
{
	// OpenGL - Create shaders
	char buf[16384];
	int len = 0;
	checkGL( "Start shaders" );

	// OpenGL 4.2 Core
	// -- Cannot use hardware lighting pipeline (e.g. glLightfv, glMaterialfv)
	GLuint vs = glCreateShader(GL_VERTEX_SHADER);
	GLchar const * vss =
			"#version 300 es\n"			
			"\n"
			"layout(location = 0) in vec3 inPos;\n"
			"layout(location = 1) in uint inClr;\n"
			"layout(location = 2) in vec3 inVel;\n"
			"out vec4 vpos;\n"		
			"out vec4 vclr;\n"
		    "out float vintes;\n"
			"uniform vec4 voffset;\n"
			"uniform vec4 eyeHi;\n"
			"uniform vec4 eyeLo;\n"
			"uniform mat4 modelMatrix;\n"
			"uniform mat4 viewMatrix;\n"
			"uniform mat4 projMatrix;\n"
			"\n"
			"vec4 CLR2VEC ( uint c ) {	return vec4( float(c & 255u)/255.0, float((c>>8u) & 255u)/255.0, float((c>>16u) & 255u)/255.0, float((c>>24u) & 255u)/255.0 ); } \n"
		    "\n"
			"void main()\n"
			"{\n"		
			"    vpos = vec4( ( inPos * voffset.w) + (voffset.xyz*voffset.w - eyeHi.xyz), 1 ) ;\n" 
			"    float v = length(inVel)*length(inVel)/20.f;\n"
			"    vclr = CLR2VEC(inClr) + vec4(v,v,v, 1);\n"
			"    gl_Position = projMatrix * viewMatrix * modelMatrix * vpos;\n"			
			"	 vpos += eyeHi;\n" 
			"}\n"
	;

	glShaderSource(vs, 1, &vss, 0);
	glCompileShader(vs);
	glGetShaderInfoLog ( vs, 16384, (GLsizei*) &len, buf );
	if ( len > 0 ) dbgprintf  ( "ERROR Shader3D vert: %s\n", buf );
	checkGL( "Compile vertex shader" );

	GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
	GLchar const * fss =
		"#version 300 es\n"
		"  precision mediump float;\n"
		"  precision mediump int;\n"
		"uniform sampler2D imgTex;\n"
		"in vec4		vpos; \n"	
		"in vec4		vclr; \n"
		"in float		vintes; \n"
		"in float	vdepth;\n"
		"uniform vec3	lightpos; \n"
		"out vec4		outColor;\n"
		"void main () {\n"										
		"    outColor = vclr;\n"
		"}\n"
	;	

	glShaderSource(fs, 1, &fss, 0);
	glCompileShader(fs);
	glGetShaderInfoLog ( fs, 16384, (GLsizei*) &len, buf );
	if ( len > 0 ) dbgprintf  ( "ERROR Shader3D frag: %s\n", buf );
	checkGL( "Compile fragment shader" );

	mSH[SPNT] = glCreateProgram();
	glAttachShader( mSH[SPNT], vs);
	glAttachShader( mSH[SPNT], fs);
	checkGL( "Attach program" );
	glLinkProgram( mSH[SPNT] );
	checkGL( "Link program" );
	glUseProgram( mSH[SPNT] );
	checkGL( "Use program" );

    mProj[SPNT] =	glGetUniformLocation ( mSH[SPNT], "projMatrix" );
    mModel[SPNT] =	glGetUniformLocation ( mSH[SPNT], "modelMatrix" );
    mView[SPNT] =	glGetUniformLocation ( mSH[SPNT], "viewMatrix" );
    mLight[SPNT] =	glGetUniformLocation ( mSH[SPNT], "lightpos" );
	mTex[SPNT] =	glGetUniformLocation ( mSH[SPNT], "imgTex" );
	checkGL( "Get tex" );	

	mOffs[SPNT] =	glGetUniformLocation ( mSH[SPNT], "voffset" );
	checkGL( "Get voff" );	
	
	mEyeHi[SPNT] =	glGetUniformLocation ( mSH[SPNT], "eyeHi" );
	mEyeLo[SPNT] =	glGetUniformLocation ( mSH[SPNT], "eyeLo" );
	checkGL( "Get hi/lo" );	

	checkGL( "Get Shader Matrices" );	
}

void nvDraw::drawGL ()
{
	glEnable ( GL_DEPTH_TEST );
	glEnable ( GL_TEXTURE_2D );	
	glEnable ( GL_BLEND );
	glDepthFunc ( GL_LESS );
	glBlendFunc ( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
	glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE );

	glBindVertexArray ( mVAO );		
	glUseProgram ( mSH[S3D] );	
	if ( m3D.size()==0 ) { dbgprintf ( "ERROR: Must have used start3D before using drawGL\n" ); }
	nvSet* s = &m3D[ 0 ];	

	/*glProgramUniformMatrix4fv ( mSH[S3D], mProj[S3D],  1, GL_FALSE, s->proj );
	glProgramUniformMatrix4fv ( mSH[S3D], mModel[S3D], 1, GL_FALSE, s->model ); 
	glProgramUniformMatrix4fv ( mSH[S3D], mView[S3D],  1, GL_FALSE, s->view );*/
	glUniformMatrix4fv ( mProj[S3D],  1, GL_FALSE, s->proj );
	glUniformMatrix4fv ( mModel[S3D], 1, GL_FALSE, s->model );
	glUniformMatrix4fv ( mView[S3D],  1, GL_FALSE, s->view );

	glActiveTexture ( GL_TEXTURE0 );		
	glBindTexture ( GL_TEXTURE_2D, mWhiteImg.getTex() );
	checkGL ( "drawGL" );
}
void nvDraw::setLight (int s, float x1, float y1, float z1 )
{
	//glProgramUniform3f ( mSH[s], mLight[s], x1, y1, z1 );
	glUniform3f ( mLight[s], x1, y1, z1 );
}
void nvDraw::setPreciseEye (int s, Camera3D* cam )
{
	Vector3DF hi,lo;
	cam->getPreciseEye ( hi, lo );
	glUniformMatrix4fv ( mView[s],  1, GL_FALSE, cam->getRotateMatrix().GetDataF() );	// use rotate instead of view matrix (no translate)
	glUniform4f ( mEyeHi[s], hi.x, hi.y, hi.z, 0 );
	glUniform4f ( mEyeLo[s], lo.x, lo.y, lo.z, 0 );
}
void nvDraw::setOffset (int s, float x, float y, float z, float sz )
{
	glUniform4f ( mOffs[s], x, y, z, sz );
}


bool nvDraw::Initialize ( const char* fontName )
{
	CreateSColor ();
	CreateSInst ();
	CreateS3D ();
	CreateSPnt ();

	if ( mVAO != 65535 ) {
		dbgprintf ( "ERROR: init2D was already called.\n" );
	}
	#ifdef USE_DX	
		// DirectX - Create model/view/proj buffers
		D3D11_BUFFER_DESC bd; 
		ZeroMemory( &bd, sizeof(bd) ); 
		bd.Usage = D3D11_USAGE_DEFAULT; 
		bd.ByteWidth = sizeof(MatrixBuffer); 
		bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER; 
		bd.CPUAccessFlags = 0;
		HRESULT hr;
		checkHR ( g_pDevice->CreateBuffer( &bd, NULL, &mpMatrixBuffer[0] ), "CreateBuffer" );
		checkHR ( g_pDevice->CreateBuffer( &bd, NULL, &mpMatrixBuffer[1] ), "CreateBuffer" );
		checkHR ( g_pDevice->CreateBuffer( &bd, NULL, &mpMatrixBuffer[2] ), "CreateBuffer" );
	#else
		// OpenGL - Create VAO
		glGenVertexArrays ( 1, &mVAO );
		checkGL ( "init, glGenVertexArrays" );
	#endif
	glBindVertexArray ( mVAO );
    checkGL ( "init, glBindVertexArray" );
	// cube edges
	memset ( &mCubeEdgeSet, 0, sizeof(nvSet) );
	mCurrSet = &mCubeEdgeSet;
	drawLine3D ( 0, 0, 0, 1, 0, 0, 1, 1, 1, 1 );
	drawLine3D ( 1, 0, 0, 1, 0, 1, 1, 1, 1, 1 );
	drawLine3D ( 1, 0, 1, 0, 0, 1, 1, 1, 1, 1 );
	drawLine3D ( 0, 0, 1, 0, 0, 0, 1, 1, 1, 1 );
	drawLine3D ( 0, 1, 0, 1, 1, 0, 1, 1, 1, 1 );
	drawLine3D ( 1, 1, 0, 1, 1, 1, 1, 1, 1, 1 );
	drawLine3D ( 1, 1, 1, 0, 1, 1, 1, 1, 1, 1 );
	drawLine3D ( 0, 1, 1, 0, 1, 0, 1, 1, 1, 1 );	
	drawLine3D ( 0, 0, 0, 0, 1, 0, 1, 1, 1, 1 );
	drawLine3D ( 1, 0, 0, 1, 1, 0, 1, 1, 1, 1 );
	drawLine3D ( 1, 0, 1, 1, 1, 1, 1, 1, 1, 1 );
	drawLine3D ( 0, 0, 1, 0, 1, 1, 1, 1, 1, 1 );
	UpdateVBOs ( mCubeEdgeSet );
	

	// cube faces
	memset(&mCubeFaceSet, 0, sizeof(nvSet));
	mCurrSet = &mCubeFaceSet;
	drawFace3D (0, 1, 0,  1, 1, 0,  1, 1, 1,  0, 1, 1,  0,  1, 0,  1, 1, 1, 1);	 // y+
	drawFace3D (0, 0, 0,  1, 0, 0,  1, 0, 1,  0, 0, 1,  0, -1, 0,  1, 1, 1, 1);  // y-

	drawFace3D (0, 0, 1,  1, 0, 1,  1, 1, 1,  0, 1, 1,  0, 0,  1,  1, 1, 1, 1);  // z+
	drawFace3D (0, 0, 0,  1, 0, 0,  1, 1, 0,  0, 1, 0,  0, 0, -1,  1, 1, 1, 1);  // z-

	drawFace3D (1, 0, 0,  1, 0, 1,  1, 1, 1,  1, 1, 0,   1, 0, 0,  1, 1, 1, 1);  // x+
	drawFace3D (0, 0, 0,  0, 0, 1,  0, 1, 1,  0, 1, 0,  -1, 0, 0,  1, 1, 1, 1);  // x-
	UpdateVBOs(mCubeFaceSet);

	mWhiteImg.Create ( 8, 8, IMG_RGBA );
	mWhiteImg.Fill ( 1,1,1,1 );

	if ( !LoadFont ( fontName ) ) return false;
	
	return true;
}

void nvDraw::UpdateVBOs ( nvSet& s )
{	
	#ifdef USE_DX	

		for (int n=0; n < GRP_MAX; n++) {
			#ifdef DEBUG_UTIL
				dbgprintf  ( "Draw::UpdateVBOs: %d of %d.\n", n, GRP_MAX );
			#endif
			if ( s.mNum[n] == 0 ) continue;

			if ( s.mVBO[n] != 0x0 ) s.mVBO[n]->Release ();

			D3D11_BUFFER_DESC bd; 
			ZeroMemory( &bd, sizeof(bd) ); 
			bd.Usage = D3D11_USAGE_DYNAMIC;
			bd.ByteWidth = s.mNum[n] * sizeof(nvVert); 
			bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
			bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
			bd.StructureByteStride = 0;				
			checkHR ( g_pDevice->CreateBuffer( &bd, 0x0, &s.mVBO[n] ), "CreateBuffer(VBO)" );				
			
			// create index buffer 
			bd.ByteWidth = s.mNumI[n] * sizeof(uint);
			bd.BindFlags = D3D11_BIND_INDEX_BUFFER;
			checkHR ( g_pDevice->CreateBuffer( &bd, 0x0, &s.mVBOI[n] ), "CreateBuffer(VBO)" );					
			
			D3D11_MAPPED_SUBRESOURCE resrc;
			ZeroMemory( &resrc, sizeof(resrc) ); 
			checkHR( g_pContext->Map ( s.mVBO[n], 0, D3D11_MAP_WRITE_DISCARD, 0, &resrc ), "Map" );
			memcpy ( resrc.pData, s.mGeom[n], s.mNum[n] * sizeof(nvVert) );
			g_pContext->Unmap ( s.mVBO[n], 0 );
			
			checkHR( g_pContext->Map ( s.mVBOI[n], 0, D3D11_MAP_WRITE_DISCARD, 0, &resrc ), "Map" );
			memcpy ( resrc.pData, s.mIdx[n], s.mNumI[n] * sizeof(uint) );		
			g_pContext->Unmap ( s.mVBOI[n], 0 );
		} 
		
	#else		

		for (int n=0; n < GRP_MAX; n++ ) {
			if ( s.mNum[n] == 0 ) continue; 
			if ( s.mVBO[n] == 0 ) {
			    glGenBuffers ( 1, &s.mVBO[n] );
                checkGL ( "UpdateVBO:glGenBuffers" );
			}
			// bind buffer /w data
			glBindBuffer ( GL_ARRAY_BUFFER, s.mVBO[n] );			
			glBufferData ( GL_ARRAY_BUFFER, s.mNum[n] * sizeof(nvVert), s.mGeom[n], GL_STATIC_DRAW );
			checkGL ( "UpdateVBO:glBufferData" );
			// bind index buffer - not necessary in GL when using glDrawElements
			
			//-- debugging
			//if ( n == GRP_IMG ) 
			//	for (int j=0; j < s.mNum[n]; j++ ) dbgprintf  ( "%d  %f,%f,%f\n", j, s.mGeom[n][j].x, s.mGeom[n][j].y, s.mGeom[n][j].z );
		}
	#endif
}

void nvDraw::drawSet2D ( nvSet& s )
{
	GLsizei sz;

	if ( s.zfactor == 1.0 ) 
		glDisable ( GL_DEPTH_TEST );		// don't preserve order
	else					
		glEnable ( GL_DEPTH_TEST );			// preserve order

	/*glProgramUniformMatrix4fv ( mSH[SCOLOR], mProj[SCOLOR],  1, GL_FALSE, s.proj );
	glProgramUniformMatrix4fv ( mSH[SCOLOR], mModel[SCOLOR], 1, GL_FALSE, s.model ); 
	glProgramUniformMatrix4fv ( mSH[SCOLOR], mView[SCOLOR],  1, GL_FALSE, s.view );*/
	glUniformMatrix4fv ( mProj[SCOLOR],  1, GL_FALSE, s.proj );
	glUniformMatrix4fv ( mModel[SCOLOR], 1, GL_FALSE, s.model );
	glUniformMatrix4fv ( mView[SCOLOR],  1, GL_FALSE, s.view );

	checkGL ( "matrices" );	

	// triangles			

	//glEnable ( GL_PRIMITIVE_RESTART );			// OpenGL 4.0 only
	//glPrimitiveRestartIndex ( IDX_NULL );

	if ( s.mVBO[GRP_TRI] !=0 && s.mNum[GRP_TRI] != 0 ) {			
		glBindTexture(GL_TEXTURE_2D, mWhiteImg.getTex());		// default texture (solid white)
		glBindBuffer ( GL_ARRAY_BUFFER, s.mVBO[ GRP_TRI ] );	
		glVertexAttribPointer( localPos, 3, GL_FLOAT, GL_FALSE, sizeof(nvVert), 0 );
		glVertexAttribPointer( localClr, 4, GL_FLOAT, GL_FALSE, sizeof(nvVert), (void*) 12 );
		glVertexAttribPointer( localUV,  2, GL_FLOAT, GL_FALSE, sizeof(nvVert), (void*) 28 );
		sz = (GLsizei) s.mNumI[GRP_TRI];
		glDrawElements ( GL_TRIANGLE_STRIP, sz, GL_UNSIGNED_INT, s.mIdx[GRP_TRI] );		
	}
	checkGL ( "triangles" );

	// images
	// * Note: Must be drawn individually unless we use bindless
	int pos=0;
	uint* img = s.mIdx[GRP_IMG];								// using index to store image GLIDs
	char* pnt;	
	for (int n=0; n < s.mNum[GRP_IMG] / 4 ; n++ ) {		
		glBindTexture ( GL_TEXTURE_2D, *img );
		glBindBuffer ( GL_ELEMENT_ARRAY_BUFFER, 0 );
		glBindBuffer ( GL_ARRAY_BUFFER, s.mVBO[ GRP_IMG ] );
		glVertexAttribPointer( localPos, 3, GL_FLOAT, GL_FALSE, sizeof(nvVert), (void*) (pos + 0) );
		glVertexAttribPointer( localClr, 4, GL_FLOAT, GL_FALSE, sizeof(nvVert), (void*) (pos + 12) );
		glVertexAttribPointer( localUV,  2, GL_FLOAT, GL_FALSE, sizeof(nvVert), (void*) (pos + 28) );

		glDrawArrays ( GL_TRIANGLE_FAN, 0, 4 );
		checkGL ( "images" );
		pos+= sizeof(nvVert)*4;
		img++;
	}

	// text
	if ( s.mVBO[GRP_TRITEX] !=0 && s.mNum[GRP_TRITEX] != 0 ) {
		glBindTexture ( GL_TEXTURE_2D, mFontImg.getTex() );			
		glBindBuffer ( GL_ARRAY_BUFFER, s.mVBO[ GRP_TRITEX ] );	
		glVertexAttribPointer( localPos, 3, GL_FLOAT, GL_FALSE, sizeof(nvVert), 0 );
		glVertexAttribPointer( localClr, 4, GL_FLOAT, GL_FALSE, sizeof(nvVert), (void*) 12 );
		glVertexAttribPointer( localUV,  2, GL_FLOAT, GL_FALSE, sizeof(nvVert), (void*) 28 );
		sz = (GLsizei) s.mNumI[GRP_TRITEX];
		glDrawElements ( GL_TRIANGLE_STRIP, sz, GL_UNSIGNED_INT, s.mIdx[GRP_TRITEX] );
		checkGL ( "text" );
	}		
	//glDisable ( GL_PRIMITIVE_RESTART );		// OpenGL 4.0 only

	// lines 			
	if ( s.mVBO[GRP_LINES] !=0 && s.mNum[GRP_LINES] != 0 ) {
		glLineWidth(2);
		glBindTexture(GL_TEXTURE_2D, mWhiteImg.getTex());			// default texture (solid white)
		glBindBuffer ( GL_ARRAY_BUFFER, s.mVBO[ GRP_LINES ] );	
		glVertexAttribPointer( localPos, 3, GL_FLOAT, GL_FALSE, sizeof(nvVert), 0 );
		glVertexAttribPointer( localClr, 4, GL_FLOAT, GL_FALSE, sizeof(nvVert), (void*) 12 );
		glVertexAttribPointer( localUV,  2, GL_FLOAT, GL_FALSE, sizeof(nvVert), (void*) 28 );
		sz = (GLsizei) s.mNum[GRP_LINES];
		glDrawArrays ( GL_LINES, 0, sz );		
	}
	checkGL ( "lines" );

	if (s.mVBO[GRP_POINTS] != 0 && s.mNum[GRP_POINTS] != 0) {
		glBindTexture(GL_TEXTURE_2D,mWhiteImg.getTex());			// default texture (solid white)
		glBindBuffer (GL_ARRAY_BUFFER,s.mVBO[GRP_POINTS]);
		glVertexAttribPointer(localPos,3,GL_FLOAT,GL_FALSE,sizeof(nvVert),0);
		glVertexAttribPointer(localClr,4,GL_FLOAT,GL_FALSE,sizeof(nvVert),(void*)12);
		glVertexAttribPointer(localUV,2,GL_FLOAT,GL_FALSE,sizeof(nvVert),(void*)28);
		sz = (GLsizei)s.mNum[GRP_POINTS];
		glDrawArrays (GL_POINTS, 0, sz);
	}
	checkGL ("points");

}

void nvDraw::selfDraw3D ( Camera3D* cam, int sh ) 
{
	glEnable ( GL_DEPTH_TEST );
	glEnable ( GL_BLEND );
	glBlendFunc ( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
	glDepthFunc ( GL_LESS );		

	glBindVertexArray(mVAO);	
	glUseProgram(mSH[sh]);	

	Matrix4F ident;
	ident.Identity();
	glUniformMatrix4fv ( mProj[sh],  1, GL_FALSE,  cam->getProjMatrix().GetDataF() );
	glUniformMatrix4fv ( mModel[sh], 1, GL_FALSE, ident.GetDataF() );
	glUniformMatrix4fv ( mView[sh],  1, GL_FALSE, cam->getViewMatrix().GetDataF() );

	glEnableVertexAttribArray(localPos);
}

void nvDraw::selfEndDraw3D ()
{
	glUseProgram ( 0 );
	glBindVertexArray ( 0 );

	checkGL ( "selfEndDraw3D" );
}


void nvDraw::drawSet3D ( nvSet& s )
{
	if (s.zfactor == 1.0)
		glEnable(GL_DEPTH_TEST);		// standard 3D		
	else
		glDisable(GL_DEPTH_TEST);		// overlay 3D

	glBindVertexArray(mVAO);
	glUseProgram(mSH[S3D]);

	/*glProgramUniformMatrix4fv ( mSH[S3D], mProj[S3D],  1, GL_FALSE, s.proj );
	glProgramUniformMatrix4fv ( mSH[S3D], mModel[S3D], 1, GL_FALSE, s.model );
	glProgramUniformMatrix4fv ( mSH[S3D], mView[S3D],  1, GL_FALSE, s.view );*/
	glUniformMatrix4fv ( mProj[S3D],  1, GL_FALSE, s.proj );
	glUniformMatrix4fv ( mModel[S3D], 1, GL_FALSE, s.model );
	glUniformMatrix4fv ( mView[S3D],  1, GL_FALSE, s.view );

	GLsizei sz;

	glEnableVertexAttribArray(localPos);
	glEnableVertexAttribArray(localClr);
	glEnableVertexAttribArray(localUV);

	// 3D images
	// * Note: Must be drawn individually unless we use bindless
	int pos=0;
	uint* img = s.mIdx[GRP_IMG];								// using index to store image GLIDs
	char* pnt;	
	for (int n=0; n < s.mNum[GRP_IMG] / 4 ; n++ ) {		
		glBindTexture ( GL_TEXTURE_2D, *img );
		glBindBuffer ( GL_ELEMENT_ARRAY_BUFFER, 0 );
		glBindBuffer ( GL_ARRAY_BUFFER, s.mVBO[ GRP_IMG ] );
		glVertexAttribPointer( localPos, 3, GL_FLOAT, GL_FALSE, sizeof(nvVert), (void*) (pos + 0) );
		glVertexAttribPointer( localClr, 4, GL_FLOAT, GL_FALSE, sizeof(nvVert), (void*) (pos + 12) );
		glVertexAttribPointer( localUV,  2, GL_FLOAT, GL_FALSE, sizeof(nvVert), (void*) (pos + 28) );

		glDrawArrays ( GL_TRIANGLE_FAN, 0, 4 );
		checkGL ( "images" );
		pos+= sizeof(nvVert)*4;
		img++;
	}
	
	// triangles			

	//-- OpenGL 4.0 only
	//glEnable(GL_PRIMITIVE_RESTART);
	//glPrimitiveRestartIndex(IDX_NULL);

	if (s.mVBO[GRP_TRI] != 0 && s.mNum[GRP_TRI] != 0) {
		glBindBuffer(GL_ARRAY_BUFFER, s.mVBO[GRP_TRI]);
		glVertexAttribPointer(localPos, 3, GL_FLOAT, GL_FALSE, sizeof(nvVert), 0);
		glVertexAttribPointer(localClr, 4, GL_FLOAT, GL_FALSE, sizeof(nvVert), (void*) 12);
		glVertexAttribPointer(localUV,  2, GL_FLOAT, GL_FALSE, sizeof(nvVert), (void*) 28);		
		glEnableVertexAttribArray(localNorm);
		glVertexAttribPointer(localNorm, 3, GL_FLOAT, GL_FALSE, sizeof(nvVert), (void*) 36);
		sz = (GLsizei)s.mNumI[GRP_TRI];
		glDrawElements(GL_TRIANGLE_STRIP, sz, GL_UNSIGNED_INT, s.mIdx[GRP_TRI]);
		glDisableVertexAttribArray(localNorm);
	}

	// box 3D (instanced)
	// ----------- NOTE: Need to make SINST3D shader for this. 
	/* if ( s.mVBO[GRP_BOX] !=0 && s.mNum[GRP_BOX] != 0 ) {						
		glUseProgram ( mSH[SINST] );							
		glProgramUniformMatrix4fv ( mSH[SINST], mProj[SINST],  1, GL_FALSE, s.proj );	
		glProgramUniformMatrix4fv ( mSH[SINST], mModel[SINST], 1, GL_FALSE, s.model ); 
		glProgramUniformMatrix4fv ( mSH[SINST], mView[SINST],  1, GL_FALSE, s.view );
		
		// instanced geometry 
		glBindBuffer ( GL_ARRAY_BUFFER, mCubeFaceSet.mVBO[GRP_TRI] );		
		glVertexAttribPointer ( localPos, 3, GL_FLOAT, GL_FALSE, sizeof(nvVert), 0 );		
		glVertexAttribPointer ( localClr, 4, GL_FLOAT, GL_FALSE, sizeof(nvVert), (void*) 12 );
		glVertexAttribPointer ( localUV,  2, GL_FLOAT, GL_FALSE, sizeof(nvVert), (void*) 28 );						
		glEnableVertexAttribArray(localNorm);
		glVertexAttribPointer (localNorm, 3, GL_FLOAT, GL_FALSE, sizeof(nvVert), (void*) 36);
			
		// bind instances						
		// a single instance is two verticies wide, so use stride of sizeof(nvVert)*2
		glBindBuffer ( GL_ARRAY_BUFFER, s.mVBO[GRP_BOX] );				
		glEnableVertexAttribArray ( attrPos );
		glVertexAttribPointer( attrPos, 3, GL_FLOAT, GL_FALSE, sizeof(nvVert)*2, 0 );
		glVertexAttribDivisor ( attrPos, 1 );
		glEnableVertexAttribArray ( attrClr );   
		glVertexAttribPointer( attrClr, 4, GL_FLOAT, GL_FALSE, sizeof(nvVert)*2, (void*) 12 );
		glVertexAttribDivisor ( attrClr, 1 );
		glEnableVertexAttribArray ( attrUV );
		glVertexAttribPointer( attrUV,   2, GL_FLOAT, GL_FALSE, sizeof(nvVert)*2, (void*) 28 );
		glVertexAttribDivisor ( attrUV, 1 );
		glEnableVertexAttribArray ( attrPos2 );
		glVertexAttribPointer( attrPos2, 3, GL_FLOAT, GL_FALSE, sizeof(nvVert)*2, (void*) 36 );			
		glVertexAttribDivisor ( attrPos2, 1 );						

		sz = (GLsizei) mCubeEdgeSet.mNum[GRP_LINES];
		glDrawArraysInstanced ( GL_LINES, 0, sz, (GLsizei) s.mNum[GRP_BOX] );			
		checkGL ( "draw3D boxes" );
			
		glDisableVertexAttribArray ( attrPos );
		glDisableVertexAttribArray ( attrClr );
		glDisableVertexAttribArray ( attrUV );
		glDisableVertexAttribArray ( attrPos2 );
		glDisableVertexAttribArray ( localNorm );
		glUseProgram ( mSH[SCOLOR] );				
	}	 */

	// text
	if ( s.mVBO[GRP_TRITEX] !=0 && s.mNum[GRP_TRITEX] != 0 ) {
		glBindTexture ( GL_TEXTURE_2D, mFontImg.getTex() );			
		glBindBuffer ( GL_ARRAY_BUFFER, s.mVBO[ GRP_TRITEX ] );	
		glVertexAttribPointer( localPos, 3, GL_FLOAT, GL_FALSE, sizeof(nvVert), 0 );
		glVertexAttribPointer( localClr, 4, GL_FLOAT, GL_FALSE, sizeof(nvVert), (void*) 12 );
		glVertexAttribPointer( localUV,  2, GL_FLOAT, GL_FALSE, sizeof(nvVert), (void*) 28 );
		sz = (GLsizei) s.mNumI[GRP_TRITEX];
		glDrawElements ( GL_TRIANGLE_STRIP, sz, GL_UNSIGNED_INT, s.mIdx[GRP_TRITEX] );
		checkGL ( "text" );
	}		

	// lines 	
	// -- draw lines last as they may overlay (alpha-blend) with polygons
	if (s.mVBO[GRP_LINES] != 0 && s.mNum[GRP_LINES] != 0) {
		glBindTexture(GL_TEXTURE_2D, mWhiteImg.getTex());
		glBindBuffer(GL_ARRAY_BUFFER, s.mVBO[GRP_LINES]);
		glVertexAttribPointer(localPos, 3, GL_FLOAT, GL_FALSE, sizeof(nvVert), 0);
		glVertexAttribPointer(localClr, 4, GL_FLOAT, GL_FALSE, sizeof(nvVert), (void*)12);
		glVertexAttribPointer(localUV, 2, GL_FLOAT, GL_FALSE, sizeof(nvVert), (void*)28);
		sz = (GLsizei)s.mNum[GRP_LINES];
		glDrawArrays(GL_LINES, 0, sz);
		checkGL("draw3D lines");
	}

	if (s.mVBO[GRP_POINTS] != 0 && s.mNum[GRP_POINTS] != 0) {		
		glPointSize (2.0);
		glBindTexture(GL_TEXTURE_2D,mWhiteImg.getTex());			// default texture (solid white)
		glBindBuffer (GL_ARRAY_BUFFER,s.mVBO[GRP_POINTS]);
		glVertexAttribPointer(localPos,3,GL_FLOAT,GL_FALSE,sizeof(nvVert),0);
		glVertexAttribPointer(localClr,4,GL_FLOAT,GL_FALSE,sizeof(nvVert),(void*)12);
		glVertexAttribPointer(localUV,2,GL_FLOAT,GL_FALSE,sizeof(nvVert),(void*)28);
		sz = (GLsizei)s.mNum[GRP_POINTS];
		glDrawArrays (GL_POINTS,0,sz);
		checkGL ("draw3D points");
	}	
}



void nvDraw::draw2D ()
{
    glEnable ( GL_BLEND );
    glBlendFunc ( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
	checkGL ( "draw2D:enable BLEND" ) ;

	//glEnable ( GL_TEXTURE_2D );
	//checkGL ( "draw2D:enable TEX2D" ) ;

	// Update VBOs for Dynamic Draw sets
	// (disable attrib arrays first)
	glDisableVertexAttribArray( 0 );
	glDisableVertexAttribArray( 1 );
	glDisableVertexAttribArray( 2 );
	glDisableVertexAttribArray( 3 );
    checkGL ( "draw2D:glDisableVertexAttribArray" );

	// Bind VAO
	glBindVertexArray(mVAO);
	checkGL ( "draw2D:bind VAO");

	// Update VBOs
	std::vector<nvSet>::iterator it;
	it = mDynamic.begin();	
	for ( int n = 0; n < mDynNum; n++ ) 
		UpdateVBOs ( (*it++) );

	// Bind Program
	glUseProgram(mSH[SCOLOR]);
	checkGL ( "draw2D:useprog SCOLOR");

	// Bind texture slot
	glUniform1i ( mTex[SCOLOR], 0 );
	glActiveTexture ( GL_TEXTURE0 );
	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );	

	// Draw Sets	
	// (attrib arrays enabled)
	glEnableVertexAttribArray( localPos );
	glEnableVertexAttribArray( localClr );
	glEnableVertexAttribArray( localUV  );
    checkGL ( "draw2D:enableVertexAttribArray" ) ;

	for ( it = mStatic.begin(); it != mStatic.end(); it++ ) 
		drawSet2D ( (*it) );	

	it = mDynamic.begin();
	for ( int n = 0; n < mDynNum; n++ ) 
		drawSet2D ( (*it++) );	

	// Delete dynamic buffers	
	nvSet* s;
	for (int n=0; n < mDynamic.size(); n++ ) {		
		s = &mDynamic[n];
		for (int grp=0; grp < GRP_MAX; grp++) {
			if ( s->mGeom[grp] != 0x0 ) delete s->mGeom[grp]; 
			if ( s->mIdx[grp] != 0x0 )  delete s->mIdx[grp];	
			s->mNum[grp] = 0;	s->mMax[grp] = 0;	s->mGeom[grp] = 0;
			s->mNumI[grp] = 0;	s->mMaxI[grp] = 0;	s->mIdx[grp] = 0;
		}		
	}
	mDynNum = 0;	// reset first dynamic buffer (reuses VBOs)	

	mCurrZ = 0;
}

void nvDraw::draw3D ()
{
	std::vector<nvSet>::iterator it;

	glEnable ( GL_DEPTH_TEST );
	glEnable ( GL_BLEND );
	glDepthFunc ( GL_LEQUAL );
	glBlendFunc ( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
	glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE );

	glBindVertexArray ( mVAO );		
	glUseProgram ( mSH[S3D] );

	// Bind texture slot
	glUniform1i ( mTex[S3D], 0 );
	glActiveTexture ( GL_TEXTURE0 );
	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );		
	checkGL ( "draw3D prep" );
	
	glDisableVertexAttribArray( 0 );
	glDisableVertexAttribArray( 1 );
	glDisableVertexAttribArray( 2 );
	glDisableVertexAttribArray( 3 );

	it = m3D.begin();
	for ( int n = 0; n < m3DNum; n++ ) 
		UpdateVBOs ( (*it++) );	

	glEnableVertexAttribArray( localPos );
	glEnableVertexAttribArray( localClr );
	glEnableVertexAttribArray( localUV  );	

	it = m3D.begin();
	for ( int n = 0; n < m3DNum; n++ ) 
		drawSet3D ( (*it++) );
		
	m3DNum = 0;			// reset dynamic buffers

	glUseProgram ( 0 );
	glBindVertexArray ( 0 );
	//checkGL ( "draw3D done" );
}

bool nvDraw::LoadFont ( const char * fontName )
{
    if (!fontName) return false;

    char fname[200], fpath[1024];
    sprintf (fname, "%s.tga", fontName);
	if ( !getFileLocation ( fname, fpath ) ) {
		dbgprintf ( "ERROR: Cannot find %s\n", fname );
		return false;
	}
	
	if (!mFontImg.LoadTga(fpath)) {
		dbgprintf("ERROR: Must build with TGA support for fonts.\n" );
		return false;
	}

	// change extension from .tga to .bin
	int l = (int) strlen(fpath);
	fpath[l-3] = 'b';	fpath[l-2] = 'i'; fpath[l-1] = 'n';	fpath[l-0] = '\0';
    FILE *fd = fopen( fpath, "rb" );
    if ( !fd ) {
		dbgprintf("ERROR: Cannot find %s.bin\n", fname);
		return false;
	}	
    int r = (int) fread(&mGlyphInfos, 1, sizeof(FileHeader), fd);
    fclose(fd);

	// fix glyph offsets (make all positive)
	float ymin = 10000000, ymax = 0;
	for (int c=0; c < 256; c++ ) {
		GlyphInfo& gly = mGlyphInfos.glyphs[c];
		if ( gly.pix.offY < ymin) ymin = gly.pix.offY;		
	}
	for (int c=0; c < 256; c++ ) {
		GlyphInfo& gly = mGlyphInfos.glyphs[c];
		gly.pix.offY -= ymin;
	}
	return true;
}

//--------------------------------------- 2D GUIs

nvGui::nvGui ()
{
	g_GuiCallback = 0;
	mActiveGui = -1;
}
nvGui::~nvGui()
{
	Clear();
}
void nvGui::Clear()
{
	// delete images
	for (int n = 0; n < mGui.size(); n++) {
		for (int j = 0; j < mGui[n].imgs.size(); j++)
			delete mGui[n].imgs[j];
		mGui[n].imgs.clear();
	}

	// clear
	mGui.clear();
}


int nvGui::AddGui ( float x, float y, float w, float h, char* name, int gtype, int dtype, void* data, float vmin, float vmax  )
{
	Gui g;

	g.name = name;
	g.x = x; g.y = y;
	g.w = w; g.h = h;
	g.gtype = gtype;
	g.dtype = dtype;
	g.data = data;
	g.vmin = vmin;
	g.vmax = vmax;
	g.items.clear ();
	g.backclr.Set ( 0.5f, 0.5f, 0.5f );
	g.backalpha = 0.7f;
	if ( gtype == GUI_ICON || gtype == GUI_TOOLBAR ) {
		g.backclr.Set ( 0, 0, 0 );
		g.backalpha = 0;
	}
	
	mGui.push_back ( g );
	return (int) mGui.size()-1;
}



int nvGui::AddItem ( char* name, char* imgname )
{
	int g = (int) mGui.size()-1;
	mGui[g].items.push_back ( name );
	nvImg* img = 0x0;
	
	if ( imgname != 0x0 ) {
		char fpath[1024];
		if ( getFileLocation ( imgname, fpath ) ) {
			img = new nvImg;
			img->LoadPng ( fpath );
		}		
	}
	mGui[g].imgs.push_back ( img );	

	return g;
}
void nvGui::SetBackclr ( float r, float g, float b, float a )
{
	int gi = (int) mGui.size()-1;
	mGui[gi].backclr = Vector3DF(r,g,b);
	mGui[gi].backalpha = a;
}

std::string nvGui::getItemName ( int g, int v )
{
	return mGui[g].items[v];
}

bool nvGui::guiChanged ( int n )
{
	if ( n < 0 || n >= mGui.size() ) return false;
	if ( mGui[n].changed ) {
		mGui[n].changed = false;
		return true;
	}
	return false;
}

void nvGui::Draw ( nvImg* chrome )
{
	char buf[1024];
	float x1, y1, x2, y2, frac, dx, dy, x3;
	float tx, ty;
	bool bval;

	start2D ();

	Vector3DF	tc ( 1, 1, 1);		// text color
	float		tw = 1;
	Vector3DF	toff ( 5, 15, 0 );		// text offset
	float		xoff = 50;				// value offset (from right)
	
	for (int n=0; n < mGui.size(); n++ ) {
		
		x1 = mGui[n].x;	y1 = mGui[n].y;
		x2 = x1 + mGui[n].w; y2 = y1 + mGui[n].h;	
		tx = x1 + toff.x; ty = y1 + toff.y;
		x3 = x2 - xoff;

		if ( chrome != 0x0 ) drawImg ( chrome->getTex(), x1, y1, x2, y2, 1, 1, 1, 1 );		
		else				 drawFill ( x1, y1, x2, y2, mGui[n].backclr.x, mGui[n].backclr.y, mGui[n].backclr.z, mGui[n].backalpha );
		
		switch ( mGui[n].gtype ) {
		case GUI_PRINT: {
			
			#ifdef DEBUG_UTIL
				dbgprintf  ( "Gui::Draw: Textbox. %d of %d\n", n, mGui.size() );
			#endif
			switch ( mGui[n].dtype ) {
			case GUI_STR:	sprintf ( buf, "%s", ((std::string*) mGui[n].data)->c_str() );	break;
			case GUI_INT:	sprintf ( buf, "%d", *(int*) mGui[n].data );	break;
			case GUI_VEC3:	{
				Vector3DF* val = (Vector3DF*) mGui[n].data;
				sprintf ( buf, "%4.0f x %4.0f x %4.0f", val->x, val->y, val->z );
				} break;
			case GUI_FLOAT: sprintf ( buf, "%.5f", *(float*) mGui[n].data );	break;
			case GUI_BOOL:  if (*(bool*) mGui[n].data) sprintf (buf, "on" ); else sprintf(buf,"off");	break;
			};
			getTextSize (buf, dx, dy );
			drawText ( x3, ty, buf, tc.x, tc.y, tc.z, tw );
			sprintf ( buf, "%s", mGui[n].name.c_str() );	
			drawText ( tx, ty, buf, tc.x, tc.y, tc.z, tw );	
			} break;
		case GUI_SLIDER: {				
			#ifdef DEBUG_UTIL
				dbgprintf  ( "Gui::Draw: Slider. %d of %d\n", n, mGui.size() );
			#endif			
			switch ( mGui[n].dtype ) {
			case GUI_INT:	frac = (float(*(int  *) mGui[n].data) - mGui[n].vmin) / (mGui[n].vmax-mGui[n].vmin); sprintf ( buf, "%d", *(int*) mGui[n].data );	break;
			case GUI_FLOAT: frac = (     (*(float*) mGui[n].data) - mGui[n].vmin) / (mGui[n].vmax-mGui[n].vmin); sprintf ( buf, "%.3f", *(float*) mGui[n].data );	break;
			};			
			drawFill ( x3, y1+2, x3+frac*(x2-x3), y2-2, .6f, 1.0f, .8f, 1.0f );		
			getTextSize (buf, dx, dy );
			drawText ( x3, ty, buf, tc.x, tc.y, tc.z, tw );
			sprintf ( buf, "%s", mGui[n].name.c_str() );	
			drawText ( tx, ty, buf, tc.x, tc.y, tc.z, tw );		
			} break;
		case GUI_CHECK: {
			#ifdef DEBUG_UTIL
				dbgprintf  ( "Gui::Draw: Checkbox. %d of %d\n", n, mGui.size() );
			#endif			
			switch ( mGui[n].dtype ) {
			case GUI_INT:		bval = (*(int*) mGui[n].data) == 0 ? false : true;	break;
			case GUI_FLOAT:		bval = (*(float*) mGui[n].data) == 0.0 ? false : true;	break;
			case GUI_BOOL:		bval = *(bool*) mGui[n].data;	break;
			};
			if ( bval ) {
				drawText ( x2-40, ty, "On", tc.x, tc.y, tc.z, tw );				
			} else {
				drawText ( x2-40, ty, "Off", tc.x, tc.y, tc.z, tw );
			}
			sprintf ( buf, "%s", mGui[n].name.c_str() );	
			drawText ( tx, ty, buf, tc.x, tc.y, tc.z, tw );		
			} break;
		case GUI_COMBO: {
			sprintf ( buf, "%s", mGui[n].name.c_str() );	
			drawText ( tx, ty, buf, tc.x, tc.y, tc.z, tw );
			
			int val = *(int*) mGui[n].data;
			if ( val >=0 && val < mGui[n].items.size() ) {
				sprintf ( buf, "%s", mGui[n].items[val].c_str() );
			} else {
				sprintf ( buf, "" );
			}
			drawText ( x3, ty, buf, tc.x, tc.y, tc.z, tw );
			} break;
		case GUI_TOOLBAR: {
			sprintf ( buf, "%s", mGui[n].name.c_str() );	
			int iw = mGui[n].imgs[0]->getWidth();
			int ih = mGui[n].imgs[0]->getHeight();
			float ix, iy;
			char msg[1024];

			int val = *(int*) mGui[n].data;
			ix = x1 + val*(iw + 25); iy = y1;
			drawFill ( ix, iy, ix+iw, iy+ih, 1.0, 1.0, 1.0, 0.25 );

			for (int j=0; j < mGui[n].items.size(); j++ ) {      // buttons
				ix = x1 + j*(iw + 25); iy = y1;
				drawImg ( mGui[n].imgs[j]->getTex(), ix, iy, ix+iw, iy+ih, 1, 1, 1, 1 );
				strcpy ( msg, mGui[n].items[j].c_str() );
				drawText ( ix, y2, msg, tc.x, tc.y, tc.z, tw );				
			}
			}; break;

		case GUI_ICON:	{
			drawImg ( mGui[n].imgs[0]->getTex(), x1, y1, x2, y2, 1, 1, 1, 1 );
			}; break;

		}
	}
	end2D ();

}

bool nvGui::MouseUp ( float x, float y )
{
	int ag = mActiveGui; 
	mActiveGui = -1;
	return (ag!=-1);
}

bool nvGui::MouseDown ( float x, float y )
{
	// GUI down - Check if GUI is hit
	float xoff = 150;
	float x1, y1, x2, x3, y2;
	for (int n=0; n < mGui.size(); n++ ) {
		x1 = mGui[n].x;			y1 = mGui[n].y;
		x2 = x1 + mGui[n].w;	y2 = y1 + mGui[n].h;		
		x3 = x2 - xoff;

		switch ( mGui[n].gtype ) {
		case GUI_SLIDER:
			if ( x > x3 && x < x2 && y > y1 && y < y2) {
				mActiveGui = n;	  	// set active gui								
				return true;
			}
			break;
		case GUI_CHECK: 
			if ( x > x1 && x < x2 && y > y1 && y < y2 ) {
				mActiveGui = -1;
				mGui[ n ].changed = true;
				int val;
				switch ( mGui[ n ].dtype ) {
				case GUI_INT:	val = ( (*(int*) mGui[n].data) == 0 ) ? 1 : 0;			*(int*) mGui[n].data = (int) val;		break;
				case GUI_FLOAT:	val = ( (*(float*) mGui[n].data) == 0.0 ) ? 1 : 0;		*(float*) mGui[n].data = (float) val;	break;
				case GUI_BOOL:	val = ( (*(bool*) mGui[n].data) == false ) ? 1 : 0;		*(bool*) mGui[n].data = (val==0) ? false : true;		break;
				};
				if ( g_GuiCallback ) g_GuiCallback( n, (float) val );
				return true;
			}
			break;
		case GUI_COMBO:
			if ( x > x2-xoff && x < x2 && y > y1 && y < y2 ) {
				mActiveGui = n;
				mGui [ n ].changed = true;								// combe box has changed
				int val = *(int*) mGui[n].data;							// get combo id
				val = (val >= mGui[n].items.size()-1 ) ? 0 : val+1;		// increment value
				*(int*) mGui[n].data = val;
				if ( g_GuiCallback ) g_GuiCallback( n, (float) val );
			} 
			break;
		case GUI_TOOLBAR: {
			int iw = mGui[n].imgs[0]->getWidth();
			int ih = mGui[n].imgs[0]->getHeight();
			float ix, iy;

			for (int j=0; j < mGui[n].items.size(); j++ ) {      // buttons
				ix = x1 + j*(iw + 25); iy = y1;
				if ( x > ix && y > iy && x < ix+iw && y < iy+ih ) {
					mActiveGui = n;
					mGui [ n ].changed = true;
					*(int*) mGui[n].data = j;
					if ( g_GuiCallback ) g_GuiCallback( n, (float) j );
				}
			}
			} break;
		default: break;


			
		};
	}
	mActiveGui = -1;
	return false;
}

bool nvGui::MouseDrag ( float x, float y )
{
	// GUI drag - Adjust value of hit gui
	float x1, y1, x2, x3, y2, val;
	float xoff = 150;
	if ( mActiveGui != -1 ) {
		x1 = mGui[ mActiveGui].x;			y1 = mGui[mActiveGui ].y;
		x2 = x1 + mGui[ mActiveGui ].w;	y2 = y1 + mGui[mActiveGui ].h;
		x3 = x2 - xoff;
		if ( x <= x3 ) {
			mGui[ mActiveGui ].changed = true;			
			val = mGui[ mActiveGui ].vmin;			
			if ( mGui[ mActiveGui ].dtype == GUI_INT ) val = (float) int(val); 
			if ( g_GuiCallback ) g_GuiCallback ( mActiveGui, val );
			return true;
		}
		if ( x >= x2 ) {
			mGui[ mActiveGui ].changed = true;
			val = mGui[ mActiveGui ].vmax;		
			if ( mGui[ mActiveGui ].dtype == GUI_INT ) val = (float) int(val); 
			if ( g_GuiCallback ) g_GuiCallback ( mActiveGui, val );
			return true;
		}
		if ( x > x3 && x < x2 ) {
			mGui[ mActiveGui ].changed = true;
			switch ( mGui[ mActiveGui ].dtype ) {
			case GUI_INT:	val = (float) int( mGui[ mActiveGui ].vmin +   (x-x3)*mGui[ mActiveGui ].vmax / (x2-x3) );	 break;
			case GUI_FLOAT:	val = (float) (mGui[ mActiveGui ].vmin + (x-x3)*mGui[ mActiveGui ].vmax / (x2-x3));	 break;						
			};
			if ( g_GuiCallback ) g_GuiCallback ( mActiveGui, val );
			return true;
		}
	}
	return false;
}


nvImg::nvImg ()
{
	mXres = 0;
	mYres = 0;
	mData = 0;
	mTex = UINT_NULL;
}
nvImg::~nvImg()
{
	if (mData != 0x0)
		free(mData);
}

void nvImg::Create ( int x, int y, int fmt )
{
	mXres = x;
	mYres = y;
	mSize = mXres * mYres;
	mFmt = fmt;

	switch ( mFmt ) {
	case IMG_RGB:		mSize *= 3;	break;
	case IMG_RGBA:		mSize *= 4; break;
	case IMG_GREY16:	mSize *= 2; break;	
	}

    if ( mData != 0x0 ) free ( mData );
    mData = (unsigned char*) malloc ( mSize );
	 
	memset ( mData, 0, mSize );
    
	UpdateTex();
}

void nvImg::Fill ( float r, float g, float b, float a )
{
	unsigned char* pix = mData;
	for (int n=0; n < mXres*mYres; n++ ) {
	  *pix++ = (unsigned char) (r*255.0f); 
	  *pix++ = (unsigned char) (g*255.0f); 
	  *pix++ = (unsigned char) (b*255.0f); 
	  *pix++ = (unsigned char) (a*255.0f);
	}
	UpdateTex ();
}
void nvImg::FlipY ()
{
	int pitch = mSize / mYres;
	unsigned char* buf = (unsigned char*) malloc ( pitch );
	for (int y=0; y < mYres/2; y++ ) {
		memcpy ( buf, mData + (y*pitch), pitch );		
		memcpy ( mData + (y*pitch), mData + ((mYres-y-1)*pitch), pitch );		
		memcpy ( mData + ((mYres-y-1)*pitch), buf, pitch );
	}
	UpdateTex ();
}


bool nvImg::LoadPng ( char* fname, bool bGrey )
{
	char fpath[1024];
	
	if ( ! getFileLocation ( fname, fpath ) ) {
		dbgprintf ( "ERROR: Unable to find png: %s.\n", fname );
		return false;
	}

	#ifdef BUILD_PNG	
		std::vector< unsigned char > out;
		unsigned int w, h; 

		unsigned error = lodepng::decode ( out, w, h, fpath, (bGrey ? LCT_GREY : LCT_RGBA), (bGrey ? 16 : 8) );
		if (error) {
			dbgprintf  ( "ERROR: Reading PNG: %s\n", lodepng_error_text(error) );
			return false;
		}	
		Create ( w, h, (bGrey ? IMG_GREY16 : IMG_RGBA) );
		int stride = mSize / mYres;

		for (int y=0; y < mYres; y++ ) 
			memcpy ( mData + y*stride, &out[ y*stride ], stride );

		//FlipY();

		dbgprintf ( "Decoded PNG: %d x %d, %s\n", mXres, mYres, fpath );

		UpdateTex ();
		return true;
	#else
		return false;
	#endif


}

void nvImg::SavePng ( char* fname )
{
	#ifdef BUILD_PNG
		dbgprintf  ( "Saving PNG: %s\n", fname );
		save_png ( fname, mData, mXres, mYres, 4 );
	#endif
}

bool nvImg::LoadTga ( char* fname )
{
	#ifdef BUILD_TGA
		dbgprintf  ( "Reading TGA: %s\n", fname );
		TGA* fontTGA = new TGA;
		TGA::TGAError err = fontTGA->load(fname);
		if (err != TGA::TGA_NO_ERROR) {
			delete fontTGA;
			return false;  
		}
	 
		mXres = fontTGA->m_nImageWidth;
		mYres = fontTGA->m_nImageHeight;
		mSize = mXres * mYres;
	 
		switch ( fontTGA->m_texFormat ) {
		case TGA::RGB:		mFmt = IMG_RGB;		mSize *= 3;	break;
		case TGA::RGBA:		mFmt = IMG_RGBA;	mSize *= 4; break;
		case TGA::ALPHA:	mFmt = IMG_GREY16;	mSize *= 2;	break;
		case -1:
			delete fontTGA;
			return false;
		}

		if ( mData != 0x0 ) free ( mData );
		mData = (unsigned char*) malloc ( mSize );
	 
		memcpy ( mData, fontTGA->m_nImageData, mSize );
    
		UpdateTex();

		delete fontTGA;

		return true;
	#else
		return false;
	#endif
}


void nvImg::BindTex ()
{
	#ifdef USE_DX
		ID3D11ShaderResourceView* vlist[1];
		vlist[0] = mTexIV;
		g_pContext->PSSetShaderResources( 0, 1, vlist );
		g_pContext->PSSetSamplers ( 0, 1, &g_pSamplerState );
	#else	
		glBindTexture ( GL_TEXTURE_2D, mTex );
	#endif
}


void nvImg::UpdateTex ()
{
	#ifdef USE_DX

		unsigned char* fixed_data = mData;

		DXGI_FORMAT fmt;		
		int size;
		switch ( mFmt ) {
		case IMG_RGB: {			
			fmt = DXGI_FORMAT_R8G8B8A8_UNORM;	
			size = 4;
			fixed_data = (unsigned char*) malloc ( mXres*mYres*size );
			unsigned char* dest = fixed_data;
			unsigned char* src = mData;
			for (int y=0; y < mYres; y++ ) {
				for (int x=0; x < mXres; x++ ) {
					*dest++ = *(src+2);
					*dest++ = *(src+1);
					*dest++ = *(src);
					*dest++ = 255;
					src+=3;
				}
			}
			
			} break;  // !!!! RGB removed in DX11
		case IMG_RGBA:	fmt = DXGI_FORMAT_R8G8B8A8_UNORM;	size = 4; break;
		case IMG_LUM:	fmt = DXGI_FORMAT_R8_UNORM;			size = 1; break;
		}

		D3D11_TEXTURE2D_DESC desc;
		ZeroMemory ( &desc, sizeof(desc) );		
		desc.Width = mXres;
		desc.Height = mYres;
		desc.MipLevels = desc.ArraySize = 1;
		desc.Format = fmt;
		desc.SampleDesc.Count = 1;
		desc.SampleDesc.Quality = 0;
		desc.Usage = D3D11_USAGE_DYNAMIC;
		desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
		desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
		desc.MiscFlags = 0;		
		g_pDevice->CreateTexture2D ( &desc, 0, &mTex );

		D3D11_MAPPED_SUBRESOURCE resrc;
		ZeroMemory( &resrc, sizeof(resrc) ); 
		checkHR( g_pContext->Map ( mTex, 0, D3D11_MAP_WRITE_DISCARD, 0, &resrc ), "Map" );
		memcpy ( resrc.pData, fixed_data, mXres*mYres * size );
		g_pContext->Unmap ( mTex, 0 );
		
		D3D11_SHADER_RESOURCE_VIEW_DESC view_desc;		
		ZeroMemory ( &view_desc, sizeof(view_desc) );		
		view_desc.Format = desc.Format;
		view_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
		view_desc.Texture2D.MipLevels = 1;
		view_desc.Texture2D.MostDetailedMip = 0;
		g_pDevice->CreateShaderResourceView ( mTex, &view_desc, &mTexIV ); 

		if ( mFmt == IMG_RGB ) {
			free ( fixed_data );
		}

	#else
		if ( mTex != UINT_NULL ) 
			glDeleteTextures ( 1, (GLuint*) &mTex );
	
		//dbgprintf  ( " Updating Texture %d x %d\n", mXres, mYres );
		glGenTextures ( 1, (GLuint*)&mTex );
		glBindTexture ( GL_TEXTURE_2D, mTex );
		checkGL ( "nvImg::UpdateTex" );
		glPixelStorei ( GL_PACK_ALIGNMENT, 1 );
		glPixelStorei ( GL_UNPACK_ALIGNMENT, 1 );
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		
		GLenum fmt;
		int size;
		switch ( mFmt ) {
		case IMG_RGB:	fmt = GL_RGB; size = 3;			break;
		case IMG_RGBA:	fmt = GL_RGBA; size = 4;		break;
		case IMG_GREY16: fmt = GL_LUMINANCE; size = 2;	break;
		}

		glTexImage2D ( GL_TEXTURE_2D, 0, fmt, mXres, mYres, 0, fmt, GL_UNSIGNED_BYTE, mData );
	#endif
}
nvMesh::nvMesh ()
{
	mVBO.clear();
}

void nvMesh::Clear ()
{
	mVertices.clear ();
	mFaceVN.clear ();
	mNumFaces = 0;
	mNumSides = 0;
}

void nvMesh::ComputeNormals ()
{    
	int v1, v2, v3;
	Vector3DF p1, p2, p3;
	Vector3DF norm, side;

    // Clear vertex normals
    for (int n=0; n < mVertices.size(); n++) {
		mVertices[n].nx = 0;
		mVertices[n].ny = 0;
		mVertices[n].nz = 0;			
    }

    // Compute normals of all faces
    int n=0;
    for (int f = 0; f < mNumFaces; f++ ) {
		v1 = mFaceVN[f*3]; v2 = mFaceVN[f*3+1]; v3 = mFaceVN[f*3+2]; 
		p1.Set ( mVertices[v1].x, mVertices[v1].y, mVertices[v1].z );
		p2.Set ( mVertices[v2].x, mVertices[v2].y, mVertices[v2].z );
		p3.Set ( mVertices[v3].x, mVertices[v3].y, mVertices[v3].z );        
        norm = p2; norm -= p1; norm.Normalize ();
        side = p3; side -= p1; side.Normalize ();
        norm.Cross ( side );
		mVertices[v1].nx += norm.x; mVertices[v1].ny += norm.y; mVertices[v1].nz += norm.z; 
		mVertices[v2].nx += norm.x; mVertices[v2].ny += norm.y; mVertices[v2].nz += norm.z; 
		mVertices[v3].nx += norm.x; mVertices[v3].ny += norm.y; mVertices[v3].nz += norm.z; 
    }

    // Normalize vertex normals
    Vector3DF vec;
    for (int n=0; n < mVertices.size(); n++) {
		p1.Set ( mVertices[n].nx, mVertices[n].ny, mVertices[n].nz );
		p1.Normalize ();
		mVertices[n].nx = p1.x; mVertices[n].ny = p1.y; mVertices[n].nz = p1.z;
	}
}

void nvMesh::AddPlyElement ( char typ, int n )
{
    dbgprintf  ( "  Element: %d, %d\n", typ, n );
    PlyElement* p = new PlyElement;
    if ( p == 0x0 ) { dbgprintf  ( "ERROR: Unable to allocate PLY element.\n" ); }
    p->num = n;
    p->type = typ;
    p->prop_list.clear ();
    m_PlyCurrElem = (int) m_Ply.size();
    m_Ply.push_back ( p );
}

void nvMesh::AddPlyProperty ( char typ, std::string name )
{
    dbgprintf  ( "  Property: %d, %s\n", typ, name.c_str() );
    PlyProperty p;
    p.name = name;
    p.type = typ;
    m_Ply [ m_PlyCurrElem ]->prop_list.push_back ( p );
}
int nvMesh::FindPlyElem ( char typ )
{
    for (int n=0; n < (int) m_Ply.size(); n++) {
        if ( m_Ply[n]->type == typ ) return n;
    }
    return -1;
}

int nvMesh::FindPlyProp ( int elem, std::string name )
{
    for (int n=0; n < (int) m_Ply[elem]->prop_list.size(); n++) {
        if ( m_Ply[elem]->prop_list[n].name.compare ( name)==0 )
            return n;
    }
    return -1;
}

bool nvMesh::LoadPly ( char* fname, float scal )
{
    FILE* fp;

    int m_PlyCnt;
    float m_PlyData[40];
    char buf[1000];
    char bword[200];
    std::string word;
    int vnum, fnum, elem, cnt;
    char typ;

    fp = fopen ( fname, "rt" );
    if ( fp == 0x0 ) { dbgprintf  ( "ERROR: Could not find mesh file: %s\n", fname ); }

    // Read header
    fgets ( buf, 1000, fp );
    readword ( buf, ' ', bword, 1000 ); word = bword;
    if ( word.compare("ply" )!=0 ) {
		dbgprintf  ( "ERROR: Not a ply file. %s\n", fname );        
    }

    m_Ply.clear ();

    dbgprintf  ( "Reading PLY mesh: %s.\n", fname );
    while ( feof( fp ) == 0 ) {
        fgets ( buf, 1000, fp );
        readword ( buf, ' ', bword, 1000 );
        word = bword;
        if ( word.compare("comment" )!=0 ) {
            if ( word.compare("end_header")==0 ) break;
            if ( word.compare("property")==0 ) {
                readword ( buf, ' ', bword, 1000 );
                word = bword;
                if ( word.compare("float")==0 )		typ = PLY_FLOAT;
                if ( word.compare("float16")==0 )	typ = PLY_FLOAT;
                if ( word.compare("float32")==0 )	typ = PLY_FLOAT;
                if ( word.compare("int8")==0 )		typ = PLY_INT;
                if ( word.compare("uint8")==0 )		typ = PLY_UINT;
                if ( word.compare("list")==0) {
                    typ = PLY_LIST;
                    readword ( buf, ' ', bword, 1000 );
                    readword ( buf, ' ', bword, 1000 );
                }
                readword ( buf, ' ', bword, 1000 );
                word = bword;
                AddPlyProperty ( typ, word );
            }
            if ( word.compare("element" )==0 ) {
                readword ( buf, ' ', bword, 1000 );    word = bword;
                if ( word.compare("vertex")==0 ) {
                    readword ( buf, ' ', bword, 1000 );
                    vnum = atoi ( bword );
                    dbgprintf  ( "  Verts: %d\n", vnum );
                    AddPlyElement ( PLY_VERTS, vnum );
                }
                if ( word.compare("face")==0 ) {
                    readword ( buf, ' ', bword, 1000 );
                    fnum = atoi ( bword );
                    dbgprintf  ( "  Faces: %d\n", fnum );
                    AddPlyElement ( PLY_FACES, fnum );
                }
            }
        }
    }

    // Read data
    int xi, yi, zi, ui, vi;
    dbgprintf  ( "Reading verts..\n" );
    elem = FindPlyElem ( PLY_VERTS );
    xi = FindPlyProp ( elem, "x" );
    yi = FindPlyProp ( elem, "y" );
    zi = FindPlyProp ( elem, "z" );
    ui = FindPlyProp ( elem, "s" );
    vi = FindPlyProp ( elem, "t" );
    if ( elem == -1 || xi == -1 || yi == -1 || zi == -1 ) {
        dbgprintf  ( "ERROR: Vertex data not found.\n" );
    }

    uxlong vert;
    for (int n=0; n < m_Ply[elem]->num; n++) {
        fgets ( buf, 1000, fp );
        for (int j=0; j < (int) m_Ply[elem]->prop_list.size(); j++) {
            readword ( buf, ' ', bword, 1000 );
            m_PlyData[ j ] = (float) atof ( bword );
        }
        vert = AddVert ( m_PlyData[xi]*scal, m_PlyData[yi]*scal, m_PlyData[zi]*scal, m_PlyData[ui], m_PlyData[vi], 0 );
    }

    dbgprintf  ( "Reading faces..\n" );
    elem = FindPlyElem ( PLY_FACES );
    xi = FindPlyProp ( elem, "vertex_indices" );
    if ( elem == -1 || xi == -1 ) {
        dbgprintf  ( "ERROR: Face data not found.\n" );
    }
    for (int n=0; n < m_Ply[elem]->num; n++) {
        fgets ( buf, 1000, fp );
        m_PlyCnt = 0;
        for (int j=0; j < (int) m_Ply[elem]->prop_list.size(); j++) {
            if ( m_Ply[elem]->prop_list[j].type == PLY_LIST ) {
                readword ( buf, ' ', bword, 1000 );
                cnt = atoi ( bword );
                m_PlyData[ m_PlyCnt++ ] = (float) cnt;
                for (int c =0; c < cnt; c++) {
                    readword ( buf, ' ', bword, 1000 );
                    m_PlyData[ m_PlyCnt++ ] = (float) atof ( bword );
                }
            } else {
                readword ( buf, ' ', bword, 1000 );
                m_PlyData[ m_PlyCnt++ ] = (float) atof ( bword );
            }
        }
        if ( m_PlyData[xi] == 3 ) {
            //debug.Printf ( "    Face: %d, %d, %d\n", (int) m_PlyData[xi+1], (int) m_PlyData[xi+2], (int) m_PlyData[xi+3] );
            AddFace ( (int) m_PlyData[xi+1], (int) m_PlyData[xi+2], (int) m_PlyData[xi+3] );
        }

        if ( m_PlyData[xi] == 4 ) {
            //debug.Printf ( "    Face: %d, %d, %d, %d\n", (int) m_PlyData[xi+1], (int) m_PlyData[xi+2], (int) m_PlyData[xi+3], (int) m_PlyData[xi+4]);
           // AddFace ( (int) m_PlyData[xi+1], (int) m_PlyData[xi+2], (int) m_PlyData[xi+3], (int) m_PlyData[xi+4] );
        }
    }
    for (int n=0; n < (int) m_Ply.size(); n++) {
        delete m_Ply[n];
    }
    m_Ply.clear ();
    m_PlyCurrElem = 0;

	dbgprintf  ( "Computing normals.\n");
	ComputeNormals ();
	dbgprintf  ( "Updating VBOs.\n");
	UpdateVBO( true );

	return 1;
}

int nvMesh::AddVert ( float x, float y, float z, float tx, float ty, float tz )
{
	Vertex v;	
	v.x = x; v.y = y; v.z = z;
	v.nx = v.x; v.ny = v.y; v.nz = v.z;
	float d = v.nx*v.nx+v.ny*v.ny+v.nz*v.nz;
	if ( d > 0 ) { d = sqrt(d); v.nx /= d; v.ny /= d; v.nz /= d; }
	
	v.tx = tx; v.ty = ty; v.tz = tz;
	
	mVertices.push_back ( v );
	return (int) mVertices.size()-1;
}
void nvMesh::SetVert ( int n, float x, float y, float z, float tx, float ty, float tz )
{
	mVertices[n].x = x;
	mVertices[n].y = y;
	mVertices[n].z = z;
	mVertices[n].tx = tx;
	mVertices[n].ty = ty;
	mVertices[n].tz = tz;
}
void nvMesh::SetNormal ( int n, float x, float y, float z )
{
	mVertices[n].nx = x;
	mVertices[n].ny = y;
	mVertices[n].nz = z;
}

int  nvMesh::AddFace ( int v0, int v1, int v2 )
{
	mFaceVN.push_back ( v0 );
	mFaceVN.push_back ( v1 );
	mFaceVN.push_back ( v2 );
	mNumFaces++;
	mNumSides = 3;
	return mNumFaces-1;
}
int nvMesh::AddFace4 ( int v0, int v1, int v2, int v3 )
{
	mFaceVN.push_back ( v0 );
	mFaceVN.push_back ( v1 );
	mFaceVN.push_back ( v2 );
	mFaceVN.push_back ( v3 );
	mNumFaces++;
	mNumSides = 4;
	return mNumFaces-1;
}

void nvMesh::UpdateVBO ( bool rebuild, int cnt )
{
	int numv = (int) mVertices.size();
	int numf = mNumFaces;

	#ifdef USE_DX
		if ( rebuild ) {
			#ifdef DEBUG_UTIL
				dbgprintf  ( "nvMesh: UpdateVBO (rebuild)\n" );
			#endif
			if ( mVBO.size() == 0 ) {
				mVBO.push_back ( 0 );		// vertices
				mVBO.push_back ( 0 );		// faces	
			} else {
				mVBO[0]->Release ();
				mVBO[1]->Release ();
			}
			D3D11_BUFFER_DESC bd; 
			ZeroMemory( &bd, sizeof(bd) ); 
			bd.Usage = D3D11_USAGE_DYNAMIC; 
			bd.ByteWidth = numv * sizeof(Vertex); 
			bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
			bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
			bd.StructureByteStride = 0;

			D3D11_SUBRESOURCE_DATA InitData; 
			ZeroMemory( &InitData, sizeof(InitData) ); InitData.pSysMem = &mVertices[0].x;			
			checkHR ( g_pDevice->CreateBuffer( &bd, &InitData, &mVBO[0] ), "CreateBuffer(VBO)" );				
			
			bd.ByteWidth = numf * mNumSides * sizeof(unsigned int);
			bd.BindFlags = D3D11_BIND_INDEX_BUFFER; 
			bd.StructureByteStride = 0; //sizeof(unsigned int);			

			ZeroMemory( &InitData, sizeof(InitData) ); InitData.pSysMem = &mFaceVN[0];
			checkHR ( g_pDevice->CreateBuffer( &bd, &InitData, &mVBO[1] ), "CreateBuffer(VBO)"  );			

		} else {
			
			D3D11_MAPPED_SUBRESOURCE resrc;
			ZeroMemory( &resrc, sizeof(resrc) ); 
			checkHR( g_pContext->Map ( mVBO[0], 0, D3D11_MAP_WRITE_DISCARD, 0, &resrc ), "Map" );

			#ifdef DEBUG_UTIL
				dbgprintf  ( "nvMesh: map\n" );
			#endif
			for (int n=0; n < cnt; n++ ) {				
				memcpy ( resrc.pData, &mVertices[0].x, numv * sizeof(Vertex) );
			}
			g_pContext->Unmap ( mVBO[0], 0 );
			#ifdef DEBUG_UTIL
				dbgprintf  ( "nvMesh: unmap\n" );
			#endif
		}

	#else		
		if ( mVBO.size()==0 ) {
			mVBO.push_back ( -1 );		// vertex buffer
			mVBO.push_back ( -1 );		// face buffer
		} else {
			glDeleteBuffers ( 1, &mVBO[0] );
			glDeleteBuffers ( 1, &mVBO[1] );
		}
		
		glGenBuffers ( 1, &mVBO[0] );
		glGenBuffers ( 1, &mVBO[1] );
		glGenVertexArrays ( 1, &mVAO );
		glBindVertexArray ( mVAO );
		glBindBuffer ( GL_ARRAY_BUFFER, mVBO[0] );
		glBufferData ( GL_ARRAY_BUFFER, mVertices.size() * sizeof(Vertex), &mVertices[0].x, GL_STATIC_DRAW );
		glVertexAttribPointer ( localPos, 3, GL_FLOAT, false, sizeof(Vertex), 0 );		
		glVertexAttribPointer ( localNorm, 3, GL_FLOAT, false, sizeof(Vertex), (void*) 12 );		
		glVertexAttribPointer ( localUV, 3, GL_FLOAT, false, sizeof(Vertex), (void*) 24 );
		glBindBuffer ( GL_ELEMENT_ARRAY_BUFFER, mVBO[1] );		
		glBufferData ( GL_ELEMENT_ARRAY_BUFFER, numf*mNumSides*sizeof(int), &mFaceVN[0], GL_STATIC_DRAW );
		glBindVertexArray ( 0 );

	#endif
}
void nvMesh::SelectVAO ()
{
	glBindVertexArray ( mVAO );
}

void nvMesh::SelectVBO ( )
{
	#ifdef USE_DX
		#ifdef DEBUG_UTIL
			dbgprintf  ( "nvMesh: SelectVBO\n" );
		#endif
		UINT stride[3];	
		UINT offset[3];
		ID3D11Buffer* vptr[3];
		vptr[0] = mVBO[0];		stride[0] = sizeof(Vertex);		offset[0] = 0;		// Pos		
		vptr[1] = mVBO[0];		stride[1] = sizeof(Vertex);		offset[1] = 12;		// Normal
		vptr[2] = mVBO[0];		stride[2] = sizeof(Vertex);		offset[2] = 24;		// UV
		g_pContext->IASetVertexBuffers( 0, 3, vptr, stride, offset ); 				
		g_pContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST ); 	
		g_pContext->IASetIndexBuffer ( mVBO[1], DXGI_FORMAT_R32_UINT, 0 );
		checkDEV( "nvMesh: SelectVBO" );
	#else
		glBindBuffer ( GL_ARRAY_BUFFER, mVBO[0] );		
		glVertexAttribPointer ( localPos, 3, GL_FLOAT, false, sizeof(Vertex), 0 );		
		glVertexAttribPointer ( localClr, 4, GL_FLOAT, false, sizeof(Vertex), 0 );		
		glVertexAttribPointer ( localUV,  2, GL_FLOAT, false, sizeof(Vertex), (void*) 24 );
		glVertexAttribPointer ( localNorm,3, GL_FLOAT, false, sizeof(Vertex), (void*) 12 );
		glEnableVertexAttribArray ( localPos );	
		glEnableVertexAttribArray ( localUV );
		glEnableVertexAttribArray ( localNorm );
		glBindBuffer ( GL_ELEMENT_ARRAY_BUFFER, mVBO[1] );			
	#endif
}
void nvMesh::Draw ( int inst )
{
	#ifdef USE_DX		
		#ifdef DEBUG_UTIL
			dbgprintf  ( "nvMesh: DrawIndexedInstanced\n" );
		#endif
		g_pContext->DrawIndexedInstanced ( mNumFaces*mNumSides, inst, 0, 0, 0 );
		checkDEV( "nvMesh: DrawIndexedInstanced" );
	#else		
		glDrawElementsInstanced ( GL_TRIANGLES, mNumFaces*mNumSides, GL_UNSIGNED_INT, 0, inst );
	#endif
	checkGL ( "nvMesh:Draw" );
}
void nvMesh::DrawPatches ( int inst )
{
	#ifdef USE_DX
    #else
		//--- OpenGL 4.0 only
		//glPatchParameteri( GL_PATCH_VERTICES, mNumSides );
		//glDrawElementsInstanced ( GL_PATCHES, mNumFaces*mNumSides, GL_UNSIGNED_INT, 0, inst );
	#endif
}

//----------------- Geometry utilities
//

#define EPS		0.00001

// Line A: p1 to p2
// Line B: p3 to p4
bool intersectLineLine(Vector3DF p1, Vector3DF p2, Vector3DF p3, Vector3DF p4, Vector3DF& pa, Vector3DF& pb, double& mua, double& mub)
{
	Vector3DF p13, p43, p21;
	double d1343, d4321, d1321, d4343, d2121;
	double numer, denom;

	p13 = p1;	p13 -= p3;
	p43 = p4;	p43 -= p3;
	if (fabs(p43.x) < EPS && fabs(p43.y) < EPS && fabs(p43.z) < EPS) return false;
	p21 = p2;	p21 -= p1;
	if (fabs(p21.x) < EPS && fabs(p21.y) < EPS && fabs(p21.z) < EPS) return false;

	d1343 = p13.Dot(p43);
	d4321 = p43.Dot(p21);
	d1321 = p13.Dot(p21);
	d4343 = p43.Dot(p43);
	d2121 = p21.Dot(p21);

	denom = d2121 * d4343 - d4321 * d4321;
	if (fabs(denom) < EPS) return false;
	numer = d1343 * d4321 - d1321 * d4343;

	mua = numer / denom;
	mub = (d1343 + d4321 * (mua)) / d4343;

	pa = p21;	pa *= (float)mua;		pa += p1;
	pb = p43;	pb *= (float)mub;		pb += p3;

	return true;
}

Vector3DF intersectLineLine(Vector3DF p1, Vector3DF p2, Vector3DF p3, Vector3DF p4)
{
	Vector3DF pa, pb;
	double ma, mb;
	if (intersectLineLine(p1, p2, p3, p4, pa, pb, ma, mb)) {
		return pa;
	}
	return p2;
}

float intersectLineBox(Vector3DF p1, Vector3DF p2, Vector3DF bmin, Vector3DF bmax)
{
	Vector3DF p;
	Vector3DF nearp, farp;
	float t[6];
	Vector3DF dir;
	dir = p2; dir -= p1;

	int bst1 = -1, bst2 = -1;		// bst1 = front face hit, bst2 = back face hit

	t[0] = (bmax.y - p1.y) / dir.y;			// 0 = max y
	t[1] = (bmin.x - p1.x) / dir.x;			// 1 = min x
	t[2] = (bmin.z - p1.z) / dir.z;			// 2 = min z
	t[3] = (bmax.x - p1.x) / dir.x;			// 3 = max x
	t[4] = (bmax.z - p1.z) / dir.z;			// 4 = max z
	t[5] = (bmin.y - p1.y) / dir.y;			// 5 = min y

	p = dir * t[0]; p += p1;    if (p.x < bmin.x || p.x > bmax.x || p.z < bmin.z || p.z > bmax.z) t[0] = -1;
	p = dir * t[1]; p += p1;    if (p.y < bmin.y || p.y > bmax.y || p.z < bmin.z || p.z > bmax.z) t[1] = -1;
	p = dir * t[2]; p += p1;    if (p.x < bmin.x || p.x > bmax.x || p.y < bmin.y || p.y > bmax.y) t[2] = -1;
	p = dir * t[3]; p += p1;    if (p.y < bmin.y || p.y > bmax.y || p.z < bmin.z || p.z > bmax.z) t[3] = -1;
	p = dir * t[4]; p += p1;    if (p.x < bmin.x || p.x > bmax.x || p.y < bmin.y || p.y > bmax.y) t[4] = -1;
	p = dir * t[5]; p += p1;    if (p.x < bmin.x || p.x > bmax.x || p.z < bmin.z || p.z > bmax.z) t[5] = -1;

	for (int j = 0; j < 6; j++)
		if (t[j] > 0.0 && (t[j] < t[bst1] || bst1 == -1)) bst1 = j;
	for (int j = 0; j < 6; j++)
		if (t[j] > 0.0 && (t[j] < t[bst2] || bst2 == -1) && j != bst1) bst2 = j;

	if (bst1 == -1)
		return 0.0f;

	if (p1.x >= bmin.x && p1.y >= bmin.y && p1.z >= bmin.z && p1.x <= bmax.x && p1.y <= bmax.y && p1.z <= bmax.z) {
		return t[bst2];
	}
	else {
		return t[bst1];
	}
}

Vector3DF intersectLinePlane(Vector3DF p1, Vector3DF p2, Vector3DF p0, Vector3DF pnorm)
{
	Vector3DF u, w;
	u = p2;	u -= p1;					// ray direction
	w = p1;	w -= p0;

	float dval = pnorm.Dot(u);
	float nval = -pnorm.Dot(w);

	if (fabs(dval) < EPS) {			// segment is parallel to plane
		if (nval == 0) return p1;       // segment lies in plane
		else			return p1;      // no intersection
	}
	// they are not parallel, compute intersection
	float t = nval / dval;
	u *= t;
	u += p1;
	return u;
}

Vector3DF projectPointLine(Vector3DF p, Vector3DF p0, Vector3DF p1 )
{
	Vector3DF dir = p1-p0;
	return p0 + dir * float( (p-p0).Dot(dir) / dir.Dot(dir));
}
Vector3DF projectPointLine(Vector3DF p, Vector3DF dir, float& t )
{
	t = float(p.Dot(dir) / dir.Dot(dir));
	return dir * t;
}

bool checkHit3D(Camera3D* cam, int x, int y, Vector3DF target, float radius)
{
	#ifdef USE_CAMERA
		Vector3DF dir = cam->inverseRay(x, y);
		Vector3DF bmin = target - Vector3DF(radius*0.5f, radius*0.5, radius*0.5);
		Vector3DF bmax = target + Vector3DF(radius*0.5f, radius*0.5, radius*0.5);
		float t = intersectLineBox(cam->getPos(), cam->getPos() + dir, bmin, bmax);
		return (t != 0);
	#else
		return false;
	#endif
}

Vector3DF moveHit3D(Camera3D* cam, int x, int y, Vector3DF target, Vector3DF plane_norm)
{
	#ifdef USE_CAMERA
		Vector3DF dir = cam->inverseRay(x, y);
		Vector3DF hit = intersectLinePlane(cam->getPos(), cam->getPos() + dir, target, plane_norm);
		return hit;
	#else
		return Vector3DF(0,0,0);
	#endif
	
}
