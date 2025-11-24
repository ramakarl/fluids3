//----------------------------------------------------------------------------------
// File:   app_util.cpp
// Author: Rama Hoetzlein
// Email:  rhoetzlein@nvidia.com
// 
// Copyright (c) 2013 NVIDIA Corporation. All rights reserved.

// PLY Format, Copyright (c) 2005-2013, Lode Vandevenne, ZLib
// Vector/Matrix/Camera, Copyright (c) 2005-2011, Rama Hoetzlein, ZLib
//
// TO  THE MAXIMUM  EXTENT PERMITTED  BY APPLICABLE  LAW, THIS SOFTWARE  IS PROVIDED
// *AS IS*  AND NVIDIA AND  ITS SUPPLIERS DISCLAIM  ALL WARRANTIES,  EITHER  EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED  TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL  NVIDIA OR ITS SUPPLIERS
// BE  LIABLE  FOR  ANY  SPECIAL,  INCIDENTAL,  INDIRECT,  OR  CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION,  DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS)
// ARISING OUT OF THE  USE OF OR INABILITY  TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
//
// Z-Lib License
// This software is provided 'as-is', without any express or implied
// warranty.  In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgment in the product documentation would be
//    appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.
//
//----------------------------------------------------------------------------------

/*!
 * This file provides utility classes including meshes, images, 2D drawing, and GUIs,
*  for making a complete demo application without any external lib dependencies. 
 * Demos should include app_direct.h or app_opengl.h to select a graphics API
 * and then include app_util.h for additional functionality here.
 *
 * Functionality in this file:
 *  - String manipulation: strParse, strSplit, strReplace, strTrim, strExtract
 *  - nvMesh: Construct, load, and render meshes. PLY format supported
 *  - nvImg: Cosntruct, load, and render images. PNG and TGA format supported
 *  - nvDraw: A lightweight, efficient, 2D drawing API. Uses VBOs to render
 *     lines, circles, triangles, and text. Allows for both static and dynamic 
 *     groups (rewrite per frame), and immediate style (out-of-order) calling.
 *  - nvGui: A lightweight class for creating on-screen GUIs. Currently only checkboxes
 *    or sliders supported. Relies on nvDraw to render.
 *  - Vector/Matrix: Classes for vector and matrix math
 *  - Camera: Class for 3D camera projection, matching OpenGL matricies
 *
 * Useage: 
 *    1. Main programs implement the functions required by app_opengl/directx.
 *    2. During display(), first do any rendering desired by your demo or application.
 *    3. Then call drawGui to render GUI items to the 2D layer.
 *    4. Then call draw2D to actually render all 2D objects (gui and user-specified)
 *    5. Finally call SwapBuffers 
  *
 * Example main.cpp:
 *    #include "app_opengl.h"
 *    #include "app_util.h"
 *    void initialize() {}
 *    void display () {
 *        drawGui();				// drawGui asks that all GUIs are rendered in 2D
 *        draw2D();					// draw2D asks that all nvDraw items are rendered
 *        SwapBuffers ( g_hDC );	// g_hDC comes from app_opengl.h
 *    }
 *
 */

#include "app_util.h"

// Globals
nvDraw	g_2D;
nvGui   g_Gui;

struct MatrixBuffer 
{
    float m[16];    
}; 

// Utility functions
void init2D ( const char* fontName )		{ g_2D.Initialize( fontName ); }
void start2D ()		{ g_2D.start2D(); }
void start2D (bool bStatic)		{ g_2D.start2D(bStatic); }
void updatestatic2D ( int n )	{ g_2D.updateStatic2D (n); }
void static2D ()	{ g_2D.start2D(true); }
void end2D ()		{ g_2D.end2D(); }
void draw2D ()		{ g_2D.Draw(); }
void setview2D (float w, float h)			{ g_2D.setView2D(w,h); }
void setview2D ( float* model, float* view, float* proj )		{ g_2D.setView2D( model, view, proj ); }
void setorder2D ( bool zt, float zfactor )		{ g_2D.setOrder2D( zt, zfactor ); }
void setText ( float scale, float kern )	{ g_2D.setText(scale,kern); }
void drawLine ( float x1, float y1, float x2, float y2, float r, float g, float b, float a )	{ g_2D.drawLine(x1,y1,x2,y2,r,g,b,a); }
void drawRect ( float x1, float y1, float x2, float y2, float r, float g, float b, float a )	{ g_2D.drawRect(x1,y1,x2,y2,r,g,b,a); }
void drawFill ( float x1, float y1, float x2, float y2, float r, float g, float b, float a )	{ g_2D.drawFill(x1,y1,x2,y2,r,g,b,a); }
void drawTri  ( float x1, float y1, float x2, float y2, float x3, float y3, float r, float g, float b, float a )	{ g_2D.drawTri(x1,y1,x2,y2,x3,y3,r,g,b,a); }
void drawCircle ( float x1, float y1, float radius, float r, float g, float b, float a )		{ g_2D.drawCircle(x1,y1,radius,r,g,b,a); }
void drawCircleDash ( float x1, float y1, float radius, float r, float g, float b, float a )	{ g_2D.drawCircleDash(x1,y1,radius,r,g,b,a); }
void drawCircleFill ( float x1, float y1, float radius, float r, float g, float b, float a )	{ g_2D.drawCircleFill(x1,y1,radius,r,g,b,a); }
void drawText ( float x1, float y1, char* msg, float r, float g, float b, float a )				{ g_2D.drawText(x1,y1,msg,r,g,b,a); }
void drawText ( float x1, float y1, char* msg  )				{ g_2D.drawText(x1,y1,msg, 1,1,1,1 ); }
float getTextX ( char* msg )	{ return g_2D.getTextX(msg); }
float getTextY ( char* msg )	{ return g_2D.getTextY(msg); }

void drawGui ()		{ g_Gui.Draw(); }
int  addGui ( float x, float y, float w, float h, char* name, int gtype, int dtype, void* data, float vmin, float vmax ) { return g_Gui.AddGui ( x, y, w, h, name, gtype, dtype, data, vmin, vmax ); }
bool guiChanged ( int n )  { return g_Gui.guiChanged(n); }
bool guiMouseDown ( float x, float y )	{ return g_Gui.MouseDown(x,y); }
bool guiMouseDrag ( float x, float y )	{ return g_Gui.MouseDrag(x,y); }



#include <sstream>

//------------------------------------------------------ UTILITIES
int strToI (std::string s) {
	//return ::atoi ( s.c_str() );
	std::istringstream str_stream ( s ); 
	int x; 
	if (str_stream >> x) return x;		// this is the correct way to convert std::string to int, do not use atoi
	return 0;
};
float strToF (std::string s) {
	//return ::atof ( s.c_str() );
	std::istringstream str_stream ( s ); 
	float x; 
	if (str_stream >> x) return x;		// this is the correct way to convert std::string to float, do not use atof
	return 0;
};
unsigned char strToC ( std::string s ) {
	char c;
	memcpy ( &c, s.c_str(), 1 );		// cannot use atoi here. atoi only returns numbers for strings containing ascii numbers.
	return c;
};

std::string strParse ( std::string& str, std::string lsep, std::string rsep )
{
	std::string result;
	size_t lfound, rfound;

	lfound = str.find_first_of ( lsep );
	if ( lfound != std::string::npos) {
		rfound = str.find_first_of ( rsep, lfound+1 );
		if ( rfound != std::string::npos ) {
			result = str.substr ( lfound+1, rfound-lfound-1 );					// return string strickly between lsep and rsep
			str = str.substr ( 0, lfound ) + str.substr ( rfound+1 );
			return result;
		} 
	}
	return "";
}
bool strGet ( std::string str, std::string& result, std::string lsep, std::string rsep )
{	
	size_t lfound, rfound;

	lfound = str.find_first_of ( lsep );
	if ( lfound != std::string::npos) {
		rfound = str.find_first_of ( rsep, lfound+1 );
		if ( rfound != std::string::npos ) {
			result = str.substr ( lfound+1, rfound-lfound-1 );					// return string strickly between lsep and rsep			
			return true;
		} 
	}
	return false;
}

std::string strSplit ( std::string& str, std::string sep )
{
	std::string result;
	size_t found;

	found = str.find_first_of ( sep );
	if ( found != std::string::npos) {
		result = str.substr ( 0, found );
		str = str.substr ( found+1 );
	} else {
		result = str;
		str = "";
	}
	return result;
}
std::string strReplace ( std::string str, std::string delim, std::string ins )
{
	size_t found = str.find_first_of ( delim );
	while ( found != std::string::npos ) {
		str = str.substr ( 0, found ) + ins + str.substr ( found+1 );
		found = str.find_first_of ( delim );
	}
	return str;
}

bool strSub ( std::string str, int first, int cnt, std::string cmp )
{
	if ( str.substr ( first, cnt ).compare ( cmp ) == 0 ) return true;
	return false;
}

// trim from start
#include <algorithm>
#include <cctype>
std::string strLTrim(std::string str) {
        str.erase(str.begin(), std::find_if(str.begin(), str.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
        return str;
}

// trim from end
std::string strRTrim(std::string str) {
        str.erase(std::find_if(str.rbegin(), str.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), str.end());
        return str;
}

// trim from both ends
std::string strTrim(std::string str) {
        return strLTrim(strRTrim(str));
}

std::string strLeft ( std::string str, int n )
{
	return str.substr ( 0, n );
}
int strExtract ( std::string& str, std::vector<std::string>& list )
{
	size_t found ;
	for (int n=0; n < list.size(); n++) {
		found = str.find ( list[n] );
		if ( found != std::string::npos ) {
			str = str.substr ( 0, found ) + str.substr ( found + list[n].length() );
			return n;
		}
	}
	return -1;
}

unsigned long getFileSize ( char* fname )
{
	FILE* fp = fopen ( fname, "rb" );
	fseek ( fp, 0, SEEK_END );
	return ftell ( fp );
}
unsigned long getFilePos ( FILE* fp )
{
	return ftell ( fp );
}



nvDraw::nvDraw ()
{
	mCurrZ = 0;
	localPos = 0;
	localClr = 1;
	localUV = 2;
	mDynNum = 0;
	mTextScale = 1;
	mTextKern = 0;
	mWidth = 0;
	mHeight = 0;	
}
void nvDraw::Initialize ( const char* fontName )
{
	MakeShaders2D ();

	LoadFont ( fontName );

	mWhiteImg.Create ( 1, 1, IMG_RGBA );
	mWhiteImg.Fill ( 1,1,1,1 );

	#ifdef USE_DX	
		// DirectX - Create model/view/proj buffers
		D3D11_BUFFER_DESC bd; 
		ZeroMemory( &bd, sizeof(bd) ); 
		bd.Usage = D3D11_USAGE_DEFAULT; 
		bd.ByteWidth = sizeof(MatrixBuffer); 
		bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER; 
		bd.CPUAccessFlags = 0;
		HRESULT hr;
		checkHR ( g_pDevice->CreateBuffer( &bd, NULL, &mpMatrixBuffer[0] ) );
		checkHR ( g_pDevice->CreateBuffer( &bd, NULL, &mpMatrixBuffer[1] ) );
		checkHR ( g_pDevice->CreateBuffer( &bd, NULL, &mpMatrixBuffer[2] ) );
	#else
		// OpenGL - Create VAO
		glGenVertexArrays ( 1, &mVAO );	
	#endif
}

void nvDraw::setView2D ( float w, float h )
{
	mWidth = w; mHeight = h;		
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
		SetMatrixView ( mStatic[n], mModelMtx, mViewMtx, mProjMtx, mZFactor );
	} else {
		SetDefaultView ( mStatic[n], mWidth, mHeight, mZFactor );
	}
}

int nvDraw::start2D ( bool bStatic )
{
	Set2D new_set;
	Set2D* s;
	
	if ( bStatic ) {		
		mStatic.push_back ( new_set );		
		s = &mStatic[ mStatic.size()-1 ]; 		
		mCurrSet = s;		
	} else {
		int curr = mDynNum;		
		if ( mDynNum >= mDynamic.size() ) {
			mDynamic.push_back ( new_set );						
			mDynNum = (int) mDynamic.size();
		} else {		
			mDynNum++;
		}		
		s = &mDynamic[curr];		
		mCurrSet = s;
	}

	for (int n=0; n < GRP_NUM; n++ ) {
		s->mNum[n] = 0;		s->mMax[n] = 0;		s->mGeom[n] = 0;
		s->mNumI[n] = 0;	s->mMaxI[n] = 0;	s->mIdx[n] = 0;		
	}
	if ( mWidth==-1 ) {
		SetMatrixView ( *s, mModelMtx, mViewMtx, mProjMtx, mZFactor );
	} else {
		SetDefaultView ( *s, mWidth, mHeight, mZFactor );
	}

	return mCurr;
}

void nvDraw::setOrder2D ( bool zt, float zfactor )
{	
	if ( zt == false ) zfactor = 1;
	mZFactor = zfactor;
}

void nvDraw::SetDefaultView ( Set2D& s, float w, float h, float zf )
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

void nvDraw::SetMatrixView ( Set2D& s, float* model, float* view, float* proj, float zf )
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

Vert2D* nvDraw::allocGeom ( int cnt, int grp, Set2D* s, int& ndx )
{
	if ( s->mNum[grp] + cnt >= s->mMax[grp] ) {		
		unsigned long new_max = s->mMax[grp] * 8 + cnt;		
		//	app_printf ( "allocGeom: expand, %lu\n", new_max );
		Vert2D* new_data = (Vert2D*) malloc ( new_max*sizeof(Vert2D) );
		if ( s->mGeom[grp] != 0x0 ) {
			memcpy ( new_data, s->mGeom[grp], s->mNum[grp]*sizeof(Vert2D) );
			delete s->mGeom[grp];
		}
		s->mGeom[grp] = new_data;
		s->mMax[grp] = new_max;
	}
	Vert2D* start = s->mGeom[grp] + s->mNum[grp];
	if ( s->zfactor != 1.0 ) {
		for (int j=0; j < cnt; j++ ) (start+j)->z = (float) mCurrZ;		
		mCurrZ++;
	} else {
		for (int j=0; j < cnt; j++ ) (start+j)->z = 0.0;		
	}
	ndx = s->mNum[grp];
	s->mNum[grp] += cnt;	
	return start;
}

uint* nvDraw::allocIdx ( int cnt, int grp, Set2D* s )
{
	if ( s->mNumI[grp] + cnt >= s->mMaxI[grp] ) {		
		unsigned long new_max = s->mMaxI[grp] * 8 + cnt;
		// app_printf ( "allocIdx: expand, %lu\n", new_max );
		uint* new_data = (uint*) malloc ( new_max*sizeof(uint) );
		if ( s->mIdx[grp] != 0x0 ) {
			memcpy ( new_data, s->mIdx[grp], s->mNumI[grp]*sizeof(uint) );
			delete s->mIdx[grp];
		}
		s->mIdx[grp] = new_data;
		s->mMaxI[grp] = new_max;
	}
	uint* start = s->mIdx[grp] + s->mNumI[grp];		
	s->mNumI[grp] += cnt;
	return start;
}

void nvDraw::remove2D ( int id )
{


}
void nvDraw::drawLine ( float x1, float y1, float x2, float y2, float r, float g, float b, float a )
{
	int ndx;
	Vert2D* v = allocGeom ( 2, GRP_LINES, mCurrSet, ndx );

	v->x = x1; v->y = y1; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;
	v->x = x2; v->y = y2; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	
}
void nvDraw::drawRect ( float x1, float y1, float x2, float y2, float r, float g, float b, float a )
{
	int ndx;
	Vert2D* v = allocGeom ( 8, GRP_LINES, mCurrSet, ndx );

	v->x = x1; v->y = y1; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0; v++;
	v->x = x2; v->y = y1; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;
	v->x = x2; v->y = y1; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0; v++;
	v->x = x2; v->y = y2; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;
	v->x = x2; v->y = y2; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0; v++;
	v->x = x1; v->y = y2; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;
	v->x = x1; v->y = y2; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0; v++;
	v->x = x1; v->y = y1; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	
}
void nvDraw::drawFill ( float x1, float y1, float x2, float y2, float r, float g, float b, float a )
{
	int ndx;
	Vert2D* v = allocGeom ( 4, GRP_TRI, mCurrSet, ndx );
	uint* i = allocIdx ( 5, GRP_TRI, mCurrSet );

	v->x = x1; v->y = y1; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;
	v->x = x2; v->y = y1; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 1; v->ty = 0;	v++;
	v->x = x1; v->y = y2; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 1;	v++;
	v->x = x2; v->y = y2; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 1; v->ty = 1;
	*i++ = ndx++; *i++ = ndx++; *i++ = ndx++; *i++ = ndx++; *i++ = IDX_NULL;
}
void nvDraw::drawTri ( float x1, float y1, float x2, float y2, float x3, float y3, float r, float g, float b, float a )
{
	int ndx;
	Vert2D* v = allocGeom ( 3, GRP_TRI, mCurrSet, ndx );
	uint* i = allocIdx ( 4, GRP_TRI, mCurrSet );

	v->x = x1; v->y = y1; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;
	v->x = x2; v->y = y2; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;
	v->x = x3; v->y = y3; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;		
	*i++ = ndx++; *i++ = ndx++; *i++ = ndx++; *i++ = IDX_NULL;
}

void nvDraw::drawCircle ( float x1, float y1, float radius, float r, float g, float b, float a )
{
	int ndx;
	Vert2D* v = allocGeom ( 62, GRP_LINES, mCurrSet, ndx );	
	
	float dx, dy, dxl, dyl;	
	dxl = (float) cos( (0/31.0)*3.141592*2.0 )*radius;
	dyl = (float) sin( (0/31.0)*3.141592*2.0 )*radius;
	for (int n=1; n < 32; n++ ) {
		dx = (float) cos( (n/31.0)*3.141592*2.0 )*radius;
		dy = (float) sin( (n/31.0)*3.141592*2.0 )*radius;
		v->x = x1+dxl; v->y = y1+dyl; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;
		v->x = x1+dx; v->y = y1+dy; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;
		dxl = dx; dyl = dy;
	}		
}
void nvDraw::drawCircleDash ( float x1, float y1, float radius, float r, float g, float b, float a )
{
	int ndx;
	Vert2D* v = allocGeom ( 32, GRP_LINES, mCurrSet, ndx );	
	
	float dx, dy;		
	for (int n=0; n < 32; n++ ) {
		dx = (float) cos( (n/31.0)*3.141592*2.0 )*radius;
		dy = (float) sin( (n/31.0)*3.141592*2.0 )*radius;		
		v->x = x1+dx; v->y = y1+dy; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;		
	}		
}
void nvDraw::drawCircleFill ( float x1, float y1, float radius, float r, float g, float b, float a )
{
	int ndx;
	Vert2D* v = allocGeom ( 64, GRP_TRI, mCurrSet, ndx );
	uint* i = allocIdx ( 65, GRP_TRI, mCurrSet );
	
	float dx, dy;
	for (int n=0; n < 32; n++ ) {
		dx = (float) cos( (n/31.0)*3.141592*2.0 )*radius;
		dy = (float) sin( (n/31.0)*3.141592*2.0 )*radius;
		v->x = x1; v->y = y1; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;
		*i++ = ndx++;
		v->x = x1+dx; v->y = y1+dy; v->r = r; v->g = g; v->b = b; v->a = a; v->tx = 0; v->ty = 0;	v++;
		*i++ = ndx++;
	}	
	*i++ = IDX_NULL;
}

// from Tristan Lorach, OpenGLText
void nvDraw::drawText ( float x1, float y1, char* msg, float r, float g, float b, float a )
{
	int len = (int) strlen ( msg );
	int ndx;
	Vert2D* v = allocGeom ( len*4, GRP_TRITEX, mCurrSet, ndx );
	uint* i = allocIdx ( len*5, GRP_TRITEX, mCurrSet );

	int h = mGlyphInfos.pix.ascent + mGlyphInfos.pix.descent + mGlyphInfos.pix.linegap;
	float lPosX = x1+1;
	float lPosY = y1;

	float lLinePosX = lPosX;
	float lLinePosY = lPosY;	
	const char* c = msg;
	
	while (*c != '\0' ) {
		if ( *c == '\n' ) {
			lPosX = lLinePosX;
			lLinePosY += h;
			lPosY = lLinePosY;
		} else if ( *c >=0 && *c <= 128 ) {
			GlyphInfo& gly = mGlyphInfos.glyphs[*c];
			float pX = lPosX + gly.pix.offX;
			float pY = lPosY + gly.pix.height + gly.pix.offY;
			
			v->x = pX; v->y = pY; v->r = r; v->g = g; v->b = b; v->a = a; 
			v->tx = gly.norm.u; v->ty = gly.norm.v;	v++;

			v->x = pX; v->y = pY - gly.pix.height*mTextScale; v->r = r; v->g = g; v->b = b; v->a = a; 
			v->tx = gly.norm.u; v->ty = gly.norm.v + gly.norm.height;	v++;
			
			v->x = pX + gly.pix.width*mTextScale; v->y = pY; v->r = r; v->g = g; v->b = b; v->a = a; 
			v->tx = gly.norm.u + gly.norm.width; v->ty = gly.norm.v;	v++;
			
			v->x = pX + gly.pix.width*mTextScale; v->y = pY - gly.pix.height*mTextScale; v->r = r; v->g = g; v->b = b; v->a = a; 
			v->tx = gly.norm.u + gly.norm.width; v->ty = gly.norm.v + gly.norm.height;	v++;

			*i++ = ndx++; *i++ = ndx++;	*i++ = ndx++; *i++ = ndx++; *i++ = IDX_NULL;
			
			lPosX += gly.pix.advance* mTextScale + mTextKern;
            lPosY += 0;
		}
		c++;
	}
}

float nvDraw::getTextX ( char* msg )
{
	int len = (int) strlen ( msg );
	int h = mGlyphInfos.pix.ascent + mGlyphInfos.pix.descent + mGlyphInfos.pix.linegap;
	float lPosX = 0;
	float lPosY = 0;
	float lLinePosX = lPosX;
	float lLinePosY = lPosY;	
	const char* c = msg;	
	while (*c != '\0' ) {
		if ( *c == '\n' ) {
			lPosX = lLinePosX;
			lLinePosY += h;
			lPosY = lLinePosY;
		} else if ( *c >=0 && *c <= 128 ) {
			GlyphInfo& gly = mGlyphInfos.glyphs[*c];
			float pX = lPosX + gly.pix.offX;
			float pY = lPosY + gly.pix.height + gly.pix.offY;			
			lPosX += gly.pix.advance* mTextScale + mTextKern;
            lPosY += 0;
		}
		c++;
	}
	return lPosX;
}
float nvDraw::getTextY ( char* msg )
{
	int len = (int) strlen ( msg );
	int h = mGlyphInfos.pix.ascent + mGlyphInfos.pix.descent + mGlyphInfos.pix.linegap;
	float lPosX = 0;
	float lPosY = 0;
	float lLinePosX = lPosX;
	float lLinePosY = lPosY;	
	const char* c = msg;	
	while (*c != '\0' ) {
		if ( *c == '\n' ) {
			lPosX = lLinePosX;
			lLinePosY += h;
			lPosY = lLinePosY;
		} else if ( *c >=0 && *c <= 128 ) {
			GlyphInfo& gly = mGlyphInfos.glyphs[*c];
			float pX = lPosX + gly.pix.offX;
			float pY = lPosY + gly.pix.height + gly.pix.offY;			
			lPosX += gly.pix.advance* mTextScale + mTextKern;
            lPosY += 0;
		}
		c++;
	}
	return lPosY;
}

void nvDraw::UpdateVBOs ( Set2D& s )
{
	
	#ifdef USE_DX	
		
		for (int n=0; n < GRP_NUM; n++) {
			if ( s.mNum[n] == 0 ) continue;
			if ( s.mVBO[n] == 0x0 ) {
				// create new buffers				
				D3D11_BUFFER_DESC bd; 
				ZeroMemory( &bd, sizeof(bd) ); 
				bd.Usage = D3D11_USAGE_DYNAMIC;
				bd.ByteWidth = s.mNum[n] * sizeof(Vert2D); 
				bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
				bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
				bd.StructureByteStride = 0;				
				checkHR ( g_pDevice->CreateBuffer( &bd, 0x0, &s.mVBO[n] ) );				
				// create index buffer 			
				bd.BindFlags = D3D11_BIND_INDEX_BUFFER;
				checkHR ( g_pDevice->CreateBuffer( &bd, 0x0, &s.mVBOI[n] ) );					
			}
			D3D11_MAPPED_SUBRESOURCE resrc;
			ZeroMemory( &resrc, sizeof(resrc) ); 
			checkHR( g_pContext->Map ( s.mVBO[n], 0, D3D11_MAP_WRITE_DISCARD, 0, &resrc ) );
			memcpy ( resrc.pData, s.mGeom[n], s.mNum[n] * sizeof(Vert2D) );
			g_pContext->Unmap ( s.mVBO[n], 0 );
			checkHR( g_pContext->Map ( s.mVBOI[n], 0, D3D11_MAP_WRITE_DISCARD, 0, &resrc ) );
			memcpy ( resrc.pData, s.mIdx[n], s.mNumI[n] * sizeof(uint) );
			g_pContext->Unmap ( s.mVBOI[n], 0 );			
		}
		
	#else
		glBindVertexArray ( mVAO );

		for (int n=0; n < GRP_NUM; n++ ) {
			if ( s.mNum[n] == 0 ) continue; 
			if ( s.mVBO[n] == 0 ) glGenBuffers ( 1, &s.mVBO[n] );		
			// bind buffer /w data
			glBindBuffer ( GL_ARRAY_BUFFER, s.mVBO[n] );
			glBufferData ( GL_ARRAY_BUFFER, s.mNum[n] * sizeof(Vert2D), s.mGeom[n], GL_DYNAMIC_DRAW_ARB);		
			// bind index buffer - not necessary in GL when using glDrawElements
			
			//-- debugging
			//for (int j=0; j < s.mNum[n]; j++ ) app_printf ( "%d  %f,%f,%f\n", j, s.mGeom[n][j].x, s.mGeom[n][j].y, s.mGeom[n][j].z );
		}
	#endif
}

void nvDraw::Draw ( Set2D& s )
{
	#ifdef USE_DX

		g_pContext->UpdateSubresource ( mpMatrixBuffer[0], 0, NULL, &s.model, sizeof(float)*16, sizeof(float)*16 );
		g_pContext->UpdateSubresource ( mpMatrixBuffer[1], 0, NULL, &s.view, sizeof(float)*16, sizeof(float)*16 );
		g_pContext->UpdateSubresource ( mpMatrixBuffer[2], 0, NULL, &s.proj, sizeof(float)*16, sizeof(float)*16 );
		g_pContext->VSSetConstantBuffers( 0, 3, &mpMatrixBuffer[0] );	

		UINT stride[3];	
		UINT offset[3];
		ID3D11Buffer* vptr[3];			
		
		mWhiteImg.BindTex ();
		vptr[0] = s.mVBO[GRP_LINES];	stride[0] = sizeof(Vert2D);		offset[0] = 0;		// Pos
		vptr[1] = s.mVBO[GRP_LINES];	stride[1] = sizeof(Vert2D);		offset[1] = 12;		// Color
		vptr[2] = s.mVBO[GRP_LINES];	stride[2] = sizeof(Vert2D);		offset[2] = 28;		// Tex Coord
		g_pContext->IASetVertexBuffers( 0, 3, vptr, stride, offset ); 
		g_pContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_LINELIST );
		g_pContext->Draw ( s.mNum[GRP_LINES], 0 );

		vptr[0] = s.mVBO[GRP_TRI];		stride[0] = sizeof(Vert2D);		offset[0] = 0;		// Pos
		vptr[1] = s.mVBO[GRP_TRI];		stride[1] = sizeof(Vert2D);		offset[1] = 12;		// Color
		vptr[2] = s.mVBO[GRP_TRI];		stride[2] = sizeof(Vert2D);		offset[2] = 28;		// Tex Coord
		g_pContext->IASetVertexBuffers( 0, 3, vptr, stride, offset ); 
		g_pContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP );
		g_pContext->IASetIndexBuffer ( s.mVBOI[GRP_TRI], DXGI_FORMAT_R32_UINT, 0 );
		g_pContext->DrawIndexed ( s.mNumI[GRP_TRI], 0, 0 );

		mFontImg.BindTex ();
		vptr[0] = s.mVBO[GRP_TRITEX];	stride[0] = sizeof(Vert2D);		offset[0] = 0;		// Pos
		vptr[1] = s.mVBO[GRP_TRITEX];	stride[1] = sizeof(Vert2D);		offset[1] = 12;		// Color
		vptr[2] = s.mVBO[GRP_TRITEX];	stride[2] = sizeof(Vert2D);		offset[2] = 28;		// Tex Coord
		g_pContext->IASetVertexBuffers( 0, 3, vptr, stride, offset ); 
		g_pContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP );
		g_pContext->IASetIndexBuffer ( s.mVBOI[GRP_TRITEX], DXGI_FORMAT_R32_UINT, 0 );
		g_pContext->DrawIndexed ( s.mNumI[GRP_TRITEX], 0, 0 ); 

	#else
		if ( s.zfactor == 1.0 ) 
			glDisable ( GL_DEPTH_TEST );		// don't preserve order
		else					
			glEnable ( GL_DEPTH_TEST );			// preserve order

		glEnable ( GL_TEXTURE_2D );
		glActiveTexture ( GL_TEXTURE1 );
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE );
		glBindTexture ( GL_TEXTURE_2D, mWhiteImg.getTex() );
		glUniformMatrix4fv ( mProj,  1, GL_FALSE, s.proj );	
		glUniformMatrix4fv ( mModel, 1, GL_FALSE, s.model ); 
		glUniformMatrix4fv ( mView,  1, GL_FALSE, s.view );	

		if ( s.mNum[GRP_LINES] > 0 ) {			
			glBindBuffer ( GL_ARRAY_BUFFER, s.mVBO[ GRP_LINES ] );	
			glVertexAttribPointer( localPos, 3, GL_FLOAT, GL_FALSE, sizeof(Vert2D), 0 );
			glVertexAttribPointer( localClr, 4, GL_FLOAT, GL_FALSE, sizeof(Vert2D), (void*) 12 );
			glVertexAttribPointer( localUV,  2, GL_FLOAT, GL_FALSE, sizeof(Vert2D), (void*) 28 );
			glDrawArrays ( GL_LINES, 0, s.mNum[GRP_LINES] );
		}

		if ( s.mNumI[GRP_TRI] > 0 ) {
			glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
			glPrimitiveRestartIndex ( IDX_NULL );				
			glEnable ( GL_PRIMITIVE_RESTART );    
			glBindBuffer ( GL_ARRAY_BUFFER, s.mVBO[ GRP_TRI ] );	
			glVertexAttribPointer( localPos, 3, GL_FLOAT, GL_FALSE, sizeof(Vert2D), 0 );
			glVertexAttribPointer( localClr, 4, GL_FLOAT, GL_FALSE, sizeof(Vert2D), (void*) 12 );
			glVertexAttribPointer( localUV,  2, GL_FLOAT, GL_FALSE, sizeof(Vert2D), (void*) 28 );
			glDrawElements ( GL_TRIANGLE_STRIP, s.mNumI[GRP_TRI], GL_UNSIGNED_INT, s.mIdx[GRP_TRI] );
			glDisable ( GL_PRIMITIVE_RESTART ); 
		}

		if ( s.mNumI[GRP_TRITEX] > 0 ) {
			glBindTexture ( GL_TEXTURE_2D, mFontImg.getTex() );
			glUniform1i ( mFont, 1 );
			glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
			glPrimitiveRestartIndex ( IDX_NULL );				
			glEnable ( GL_PRIMITIVE_RESTART );    
			glBindBuffer ( GL_ARRAY_BUFFER, s.mVBO[ GRP_TRITEX ] );	
			glVertexAttribPointer( localPos, 3, GL_FLOAT, GL_FALSE, sizeof(Vert2D), 0 );
			glVertexAttribPointer( localClr, 4, GL_FLOAT, GL_FALSE, sizeof(Vert2D), (void*) 12 );
			glVertexAttribPointer( localUV,  2, GL_FLOAT, GL_FALSE, sizeof(Vert2D), (void*) 28 );
			glDrawElements ( GL_TRIANGLE_STRIP, s.mNumI[GRP_TRITEX], GL_UNSIGNED_INT, s.mIdx[GRP_TRITEX] );
		}
		glDisable ( GL_PRIMITIVE_RESTART );
		glDisable ( GL_TEXTURE_2D );
	#endif
}

void nvDraw::Draw ()
{
	#ifdef USE_DX
		// Set 2D shader
		g_pContext->VSSetShader ( mVS, 0, 0);
		g_pContext->PSSetShader ( mPS, 0, 0);
		g_pContext->IASetInputLayout( mLO );		
		g_pContext->OMSetDepthStencilState( g_pDepthOffState, 1 );

	#else
		glEnable ( GL_BLEND );
		glUseProgram ( mSH2D );
		glBindVertexArray ( mVAO );
		glEnableVertexAttribArray( localPos );
		glEnableVertexAttribArray( localClr );
		glEnableVertexAttribArray( localUV  );
	#endif

	std::vector<Set2D>::iterator it;
	
	for ( it = mStatic.begin(); it != mStatic.end(); it++ ) {
		UpdateVBOs ( (*it) );
		Draw ( (*it) );	
	}
	for ( it = mDynamic.begin(); it != mDynamic.end(); it++ ) {
		UpdateVBOs ( (*it) );
		Draw ( (*it) );
	} 

	// delete dynamic buffers	
	Set2D* s;
	for (int n=0; n < mDynamic.size(); n++ ) {		
		s = &mDynamic[n];
		for (int grp=0; grp < GRP_NUM; grp++) {
			if ( s->mGeom[grp] != 0x0 ) delete s->mGeom[grp]; 
			if ( s->mIdx[grp] != 0x0 )  delete s->mIdx[grp];	
			s->mNum[grp] = 0;	s->mMax[grp] = 0;	s->mGeom[grp] = 0;
			s->mNumI[grp] = 0;	s->mMaxI[grp] = 0;	s->mIdx[grp] = 0;
		}		
	}
	mDynNum = 0;	// reset first dynamic buffer (reuses VBOs)	

	mCurrZ = 0;

	#ifdef USE_DX
		g_pContext->OMSetDepthStencilState( g_pDepthStencilState, 1 );
	#else
		glUseProgram ( 0 );
		glBindVertexArray ( 0 );
		glEnable ( GL_DEPTH_TEST );
	#endif
}

void nvDraw::MakeShaders2D ()
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
		checkHR ( g_pDevice->CreateVertexShader( pBlobVS->GetBufferPointer(), pBlobVS->GetBufferSize(), NULL, &mVS ) );
    	
		// Create pixel shader
		hr = D3DCompile( g_strPS, lstrlenA( g_strPS ) + 1, "PS", NULL, NULL, "PS", "ps_4_0", dwShaderFlags, 0, &pBlobPS, &pBlobError ) ;		
		checkSHADER ( hr, pBlobError );
		checkHR ( g_pDevice->CreatePixelShader( pBlobPS->GetBufferPointer(), pBlobPS->GetBufferSize(), NULL, &mPS ) );
		
		// Create input-assembler layout
		D3D11_INPUT_ELEMENT_DESC vs_layout[] =
		{
			{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT,		0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
			{ "COLOR",    0, DXGI_FORMAT_R32G32B32A32_FLOAT,	1, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
			{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,			2, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
		};
		UINT numElements = sizeof( vs_layout ) / sizeof( vs_layout[0] );		
		checkHR ( g_pDevice->CreateInputLayout( vs_layout, numElements, pBlobVS->GetBufferPointer(), pBlobVS->GetBufferSize(), &mLO ) );

	#else
		// OpenGL - Create shaders
		char buf[16384];
		int len = 0;
		checkGL( "Start shaders" );

		// OpenGL 4.2 Core
		// -- Cannot use hardware lighting pipeline (e.g. glLightfv, glMaterialfv)
		GLuint vs = glCreateShader(GL_VERTEX_SHADER);
		GLchar const * vss =
			"#version 420\n"
			"\n"
			"layout(location = 0) in vec4 inPosition;\n"
			"layout(location = 1) in vec4 inColor;\n"
			"layout(location = 2) in vec2 inTexCoord;\n"
			"out vec4 position;\n"		
			"out vec4 color;\n"				
			"out vec2 texcoord;\n"
			"uniform mat4 modelMatrix;\n"
			"uniform mat4 viewMatrix;\n"
			"uniform mat4 projMatrix;\n"		
			"out gl_PerVertex {\n"
			"   vec4 gl_Position;\n"		
			"};\n"
			"\n"
			"void main()\n"
			"{\n"		
			"	 position = modelMatrix * inPosition;\n"
			"    color = inColor;\n"
			"    texcoord = inTexCoord;\n"
			"    gl_Position = projMatrix * viewMatrix * modelMatrix * inPosition;\n"				
			"}\n"
		;
		glShaderSource(vs, 1, &vss, 0);
		glCompileShader(vs);
		glGetShaderInfoLog ( vs, 16384, (GLsizei*) &len, buf );
		app_printf ( "%s\n", buf );
		checkGL( "Compile vertex shader" );

		GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
		GLchar const * fss =
			"#version 420\n"
			"\n"		
			"uniform sampler2D fontTex;\n"
			"in vec4 position;\n"		
			"in vec4 color;\n"
			"in vec2 texcoord;\n"		
			"layout(location = 0) out vec4 outColor;\n"
			"\n"
			"void main()\n"
			"{\n"							
			"    outColor = color * vec4(1,1,1,texture(fontTex, texcoord).x); \n"
			"}\n"
		;
		glShaderSource(fs, 1, &fss, 0);
		glCompileShader(fs);
		glGetShaderInfoLog ( fs, 16384, (GLsizei*) &len, buf );
		app_printf ( "%s\n", buf );
		checkGL( "Compile fragment shader" );

		mSH2D = glCreateProgram();
		glAttachShader( mSH2D, vs);
		glAttachShader( mSH2D, fs);
		checkGL( "Attach program" );
		glLinkProgram( mSH2D );
		checkGL( "Link program" );
		glUseProgram( mSH2D );
		checkGL( "Use program" );

		mProj = glGetUniformLocation( mSH2D, "projMatrix");
		mModel = glGetUniformLocation( mSH2D, "modelMatrix");
		mView = glGetUniformLocation( mSH2D, "viewMatrix");		
		mFont = glGetUniformLocation( mSH2D, "fontTex");
		checkGL( "Get Shader Matrices" );
	#endif
}

/*
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	*/

bool nvDraw::LoadFont ( const char * fontName )
{
    if (!fontName) return false;

    char fname[200];
    sprintf (fname, "%s.tga", fontName);
	if ( !mFontImg.LoadTga ( fname ) ) return false;

	sprintf(fname, "%s.bin", fontName);
    FILE *fd = fopen(fname, "rb");
    if ( !fd ) return false;

    int r = (int)fread(&mGlyphInfos, 1, sizeof(FileHeader), fd);
    fclose(fd);

	return true;
}

//--------------------------------------- 2D GUIs

nvGui::nvGui ()
{
	mActiveGui = -1;
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
	
	mGui.push_back ( g );
	return mGui.size()-1;
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

inline float log2 ( double x ) {
	return log(x)/log(2.0);
}

void nvGui::Draw ()
{
	char buf[1024];
	float x1, y1, x2, y2, frac, dx;
	bool bval;

	start2D ();
	
	for (int n=0; n < mGui.size(); n++ ) {
		
		x1 = mGui[n].x;	y1 = mGui[n].y;
		x2 = x1 + mGui[n].w; y2 = y1 + mGui[n].h;	
		drawFill ( x1-2, y1-2, x2+2, y2+2, 0.3, 0.3, 0.5, 0.6 );

		switch ( mGui[n].gtype ) {
		case GUI_PRINT: {
			switch ( mGui[n].dtype ) {
			case GUI_INT:		sprintf ( buf, "%d", *(int*) mGui[n].data );	break;
			case GUI_INTLOG:	sprintf ( buf, "%d", *(int*) mGui[n].data );	break;
			case GUI_FLOAT:		sprintf ( buf, "%.5f", *(float*) mGui[n].data );	break;
			case GUI_BOOL:		if (*(bool*) mGui[n].data) sprintf (buf, "on" ); else sprintf(buf,"off");	break;
			};
			dx = getTextX (buf);
			drawText ( x2 - dx, y1+10, buf, .9f, .9f, .9f, 1.0f );
			sprintf ( buf, "%s", mGui[n].name.c_str() );	
			drawText ( x1, y1+10, buf, 1,1,1,1 );	
			} break;
		case GUI_SLIDER: {				
			drawFill ( x1, y1+12, x2, y2, .3f, .3f, .3f, 1.0f );	
			switch ( mGui[n].dtype ) {
			case GUI_INT:		frac = (float(*(int  *) mGui[n].data) - mGui[n].vmin) / (mGui[n].vmax-mGui[n].vmin); sprintf ( buf, "%d", *(int*) mGui[n].data );	break;
			case GUI_INTLOG:	frac = ( log2(float(*(int  *) mGui[n].data) - mGui[n].vmin) ) / log2(mGui[n].vmax); sprintf ( buf, "%d", *(int*) mGui[n].data );	break;
			case GUI_FLOAT:		frac = (     (*(float*) mGui[n].data) - mGui[n].vmin) / (mGui[n].vmax-mGui[n].vmin); sprintf ( buf, "%.3f", *(float*) mGui[n].data );	break;
			};
			drawFill ( x1, y1+12, x1+frac*(x2-x1), y2, .6f, 1.0f, .8f, 1.0f );		
			dx = getTextX (buf);
			drawText ( x2 - dx, y1+10, buf, 1.0f, 1.0f, 1.0f, 1.0f );
			sprintf ( buf, "%s", mGui[n].name.c_str() );	
			drawText ( x1, y1+10, buf, 1,1,1,1 );		
			} break;
		case GUI_CHECK: {
			drawRect ( x2-10, y1, x2, y1+10, .8f, .8f, .8f, 1.0f);
			switch ( mGui[n].dtype ) {
			case GUI_INT:		bval = (*(int*) mGui[n].data) == 0 ? false : true;	break;
			case GUI_INTLOG:	bval = (*(int*) mGui[n].data) == 0 ? false : true;	break;
			case GUI_FLOAT:		bval = (*(float*) mGui[n].data) == 0.0 ? false : true;	break;
			case GUI_BOOL:		bval = *(bool*) mGui[n].data;	break;
			};
			if ( bval ) {
				drawText ( x2-40, y1+10, "on", .6f, 1.0f, .8f, 1.0f );
				drawLine ( x2-10, y1, x2, y1+10, .6f, 1.0f, .8f, 1.0f );
				drawLine ( x2-10, y1+10, x2, y1, .6f, 1.0f, .8f, 1.0f );
			} else {
				drawText ( x2-40, y1+10, "off", .8f, .8f, .8f, 1.0f );
			}
			sprintf ( buf, "%s", mGui[n].name.c_str() );	
			drawText ( x1, y1+10, buf, 1.0f, 1.0f, 1.0f, 1.0f );		
			} break;
		}
	}
	end2D ();

}

bool nvGui::MouseDown ( float x, float y )
{
	// GUI down - Check if GUI is hit
	float x1, y1, x2, y2;
	for (int n=0; n < mGui.size(); n++ ) {
		x1 = mGui[n].x;			y1 = mGui[n].y;
		x2 = x1 + mGui[n].w;	y2 = y1 + mGui[n].h;
		switch ( mGui[n].gtype ) {
		case GUI_SLIDER:
			if ( x > x1 && x < x2 && y > y1 && y < y2) {
				mActiveGui = n;	  	// set active gui				
				return true;
			}
			break;
		case GUI_CHECK: 
			if ( x > x2-10 && x < x2 && y > y1 && y < y1+10 ) {
				mActiveGui = -1;
				mGui[ n ].changed = true;
				switch ( mGui[ n ].dtype ) {
				case GUI_INT:	*(int*) mGui[n].data = ( (*(int*) mGui[n].data) == 0 ) ? 1 : 0;			break;
				case GUI_FLOAT:	*(float*) mGui[n].data = ( (*(int*) mGui[n].data) == 0.0 ) ? 1.0 : 0.0;	break;
				case GUI_BOOL:	*(bool*) mGui[n].data = ! (*(bool*) mGui[n].data);		break;
				};
				return true;
			}
			break;
		default: break;
		};
	}
	mActiveGui = -1;
	return false;
}

bool nvGui::MouseDrag ( float x, float y )
{
	// GUI drag - Adjust value of hit gui
	float x1, y1, x2, y2;
	if ( mActiveGui != -1 ) {
		x1 = mGui[ mActiveGui].x;			y1 = mGui[mActiveGui ].y;
		x2 = x1 + mGui[ mActiveGui ].w;	y2 = y1 + mGui[mActiveGui ].h;
		if ( x <= x1 ) {
			mGui[ mActiveGui ].changed = true;
			switch ( mGui[ mActiveGui ].dtype ) {
			case GUI_INT:		*(int*) mGui[ mActiveGui ].data = (int) mGui[ mActiveGui ].vmin;		break;
			case GUI_INTLOG:	*(int*) mGui[ mActiveGui ].data = (int) mGui[ mActiveGui ].vmin;		break;
			case GUI_FLOAT:		*(float*) mGui[ mActiveGui ].data = (float) mGui[ mActiveGui ].vmin;	break;						
			};
			return true;
		}
		if ( x >= x2 ) {
			mGui[ mActiveGui ].changed = true;
			switch ( mGui[ mActiveGui ].dtype ) {
			case GUI_INT:	*(int*) mGui[ mActiveGui ].data = (int) mGui[ mActiveGui ].vmax;		break;
			case GUI_INTLOG:	*(int*) mGui[ mActiveGui ].data = (int) mGui[ mActiveGui ].vmax;		break;
			case GUI_FLOAT:	*(float*) mGui[ mActiveGui ].data = (float) mGui[ mActiveGui ].vmax;	break;						
			};
			return true;
		}
		if ( x > x1 && x < x2 ) {
			mGui[ mActiveGui ].changed = true;
			switch ( mGui[ mActiveGui ].dtype ) {
			case GUI_INT:	*(int*) mGui[ mActiveGui ].data = (int) (mGui[ mActiveGui ].vmin + (x-x1)*mGui[ mActiveGui ].vmax / (x2-x1));		break;
			case GUI_INTLOG:
				*(int*) mGui[ mActiveGui ].data = mGui[ mActiveGui ].vmin + (int) pow ( 2.0, double(x-x1) * log2(mGui[ mActiveGui ].vmax) / (x2-x1) );		break;
				break;
			case GUI_FLOAT:	*(float*) mGui[ mActiveGui ].data = (float) (mGui[ mActiveGui ].vmin + (x-x1)*mGui[ mActiveGui ].vmax / (x2-x1));	break;						
			};
			return true;
		}
	}
	return false;
}


bool readword ( char *line, char *word, char delim )
{
    int max_size = 200;
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

    if (*buf_pos=='\n' || *buf_pos=='\0') {    // buf_pos now points to the end of buffer
        //strcpy_s (word, max_size, start_pos);            // copy word to output string
        strncpy (word, start_pos, max_size);
        if ( *buf_pos=='\n') *(word + strlen(word)-1) = '\0';
        *line = '\0';                        // clear input buffer
    } else {
                                            // buf_pos now points to the delimiter after word
        *buf_pos++ = '\0';                    // replace delimiter with end-of-word marker
        //strcpy_s (word, max_size, start_pos);
        strncpy (word, start_pos, buf_pos-line );    // copy word(s) string to output string
                                            // move start_pos to beginning of entire buffer
        strcpy ( start_pos, buf_pos );        // copy remainder of buffer to beginning of buffer
    }
    return true;                        // return word(s) copied
}



/*void save_png ( char* fname, unsigned char* img, int w, int h )
{
	unsigned error = lodepng::encode ( "test.png", img, w, h );	  
	if (error) printf ( "png write error: %s\n", lodepng_error_text(error) );
}*/

void nvImg::Create ( int x, int y, int fmt )
{
	mXres = x;
	mYres = y;
	mSize = mXres * mYres;
	mFmt = fmt;

	switch ( mFmt ) {
	case IMG_RGB:  mSize *= 3;	break;
	case IMG_RGBA: mSize *= 4; break;
	case IMG_LUM:  break;
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
		*pix++ = r*255.0f; *pix++ = g*255.0f; *pix++ = b*255.0f; *pix++ = a*255.0f;
	}
	UpdateTex ();
}

bool nvImg::LoadPng ( char* fname )
{
	std::vector<unsigned char> out;
	unsigned int w, h; 

	app_printf ( "Reading PNG: %s\n", fname );
	unsigned error = lodepng::decode ( out, w, h, fname );
	if (error) {
		app_printf ( "png read error: %s\n", lodepng_error_text(error) );
		return false;
	}	
	mXres = w;
	mYres = h;
	mSize = w*h*4;
	mFmt = IMG_RGBA;
	if ( mData != 0x0 )  free ( mData );
	mData = (unsigned char*) malloc ( mSize );

	memcpy ( mData, &out[0], mSize );
	//memset ( mData, 128, mSize );

	UpdateTex ();

	return true;
}

bool nvImg::LoadTga ( char* fname )
{
	app_printf ( "Reading TGA: %s\n", fname );
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
	case TGA::ALPHA:	mFmt = IMG_LUM;					break;
	}

    if ( mData != 0x0 ) free ( mData );
    mData = (unsigned char*) malloc ( mSize );
	 
	memcpy ( mData, fontTGA->m_nImageData, mSize );
    
	UpdateTex();

	delete fontTGA;

	return true;
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
		checkHR( g_pContext->Map ( mTex, 0, D3D11_MAP_WRITE_DISCARD, 0, &resrc ) );
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
		if ( mTex != -1 ) glDeleteTextures ( 1, (GLuint*) &mTex );
	
		app_printf ( " Updating Texture %d x %d\n", mXres, mYres );
		glGenTextures(1, (GLuint*)&mTex);
		glBindTexture(GL_TEXTURE_2D, mTex);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		
		GLenum fmt;
		int size;
		switch ( mFmt ) {
		case IMG_RGB:	fmt = GL_RGB; size = 3;			break;
		case IMG_RGBA:	fmt = GL_RGBA; size = 4;		break;
		case IMG_LUM:	fmt = GL_LUMINANCE; size = 1;	break;
		}

		glTexImage2D ( GL_TEXTURE_2D, 0, fmt, mXres, mYres, 0, fmt, GL_UNSIGNED_BYTE, mData );
	#endif
}
nvMesh::nvMesh ()
{
	localPos = 0;
	localNorm = 1;
	localUV = 2;
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
    app_printf ( "  Element: %d, %d\n", typ, n );
    PlyElement* p = new PlyElement;
    if ( p == 0x0 ) { app_printf ( "ERROR: Unable to allocate PLY element.\n" ); }
    p->num = n;
    p->type = typ;
    p->prop_list.clear ();
    m_PlyCurrElem = (int) m_Ply.size();
    m_Ply.push_back ( p );
}

void nvMesh::AddPlyProperty ( char typ, std::string name )
{
    app_printf ( "  Property: %d, %s\n", typ, name.c_str() );
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
    if ( fp == 0x0 ) { app_printf ( "ERROR: Could not find mesh file: %s\n", fname ); }

    // Read header
    fgets ( buf, 1000, fp );
    readword ( buf, bword, ' ' ); word = bword;
    if ( word.compare("ply" )!=0 ) {
		app_printf ( "ERROR: Not a ply file. %s\n", fname );        
    }

    m_Ply.clear ();

    app_printf ( "Reading PLY mesh: %s.\n", fname );
    while ( feof( fp ) == 0 ) {
        fgets ( buf, 1000, fp );
        readword ( buf, bword, ' ' );
        word = bword;
        if ( word.compare("comment" )!=0 ) {
            if ( word.compare("end_header")==0 ) break;
            if ( word.compare("property")==0 ) {
                readword ( buf, bword, ' ' );
                word = bword;
                if ( word.compare("float")==0 )		typ = PLY_FLOAT;
                if ( word.compare("float16")==0 )	typ = PLY_FLOAT;
                if ( word.compare("float32")==0 )	typ = PLY_FLOAT;
                if ( word.compare("int8")==0 )		typ = PLY_INT;
                if ( word.compare("uint8")==0 )		typ = PLY_UINT;
                if ( word.compare("list")==0) {
                    typ = PLY_LIST;
                    readword ( buf, bword, ' ' );
                    readword ( buf, bword, ' ' );
                }
                readword ( buf, bword, ' ' );
                word = bword;
                AddPlyProperty ( typ, word );
            }
            if ( word.compare("element" )==0 ) {
                readword ( buf, bword, ' ' );    word = bword;
                if ( word.compare("vertex")==0 ) {
                    readword ( buf, bword, ' ' );
                    vnum = atoi ( bword );
                    app_printf ( "  Verts: %d\n", vnum );
                    AddPlyElement ( PLY_VERTS, vnum );
                }
                if ( word.compare("face")==0 ) {
                    readword ( buf, bword, ' ' );
                    fnum = atoi ( bword );
                    app_printf ( "  Faces: %d\n", fnum );
                    AddPlyElement ( PLY_FACES, fnum );
                }
            }
        }
    }

    // Read data
    int xi, yi, zi, ui, vi;
    app_printf ( "Reading verts..\n" );
    elem = FindPlyElem ( PLY_VERTS );
    xi = FindPlyProp ( elem, "x" );
    yi = FindPlyProp ( elem, "y" );
    zi = FindPlyProp ( elem, "z" );
    ui = FindPlyProp ( elem, "s" );
    vi = FindPlyProp ( elem, "t" );
    if ( elem == -1 || xi == -1 || yi == -1 || zi == -1 ) {
        app_printf ( "ERROR: Vertex data not found.\n" );
    }

    xref vert;
    for (int n=0; n < m_Ply[elem]->num; n++) {
        fgets ( buf, 1000, fp );
        for (int j=0; j < (int) m_Ply[elem]->prop_list.size(); j++) {
            readword ( buf, bword, ' ' );
            m_PlyData[ j ] = (float) atof ( bword );
        }
        vert = AddVert ( m_PlyData[xi]*scal, m_PlyData[yi]*scal, m_PlyData[zi]*scal, m_PlyData[ui], m_PlyData[vi] );
    }

    app_printf ( "Reading faces..\n" );
    elem = FindPlyElem ( PLY_FACES );
    xi = FindPlyProp ( elem, "vertex_indices" );
    if ( elem == -1 || xi == -1 ) {
        app_printf ( "ERROR: Face data not found.\n" );
    }
    for (int n=0; n < m_Ply[elem]->num; n++) {
        fgets ( buf, 1000, fp );
        m_PlyCnt = 0;
        for (int j=0; j < (int) m_Ply[elem]->prop_list.size(); j++) {
            if ( m_Ply[elem]->prop_list[j].type == PLY_LIST ) {
                readword ( buf, bword, ' ' );
                cnt = atoi ( bword );
                m_PlyData[ m_PlyCnt++ ] = (float) cnt;
                for (int c =0; c < cnt; c++) {
                    readword ( buf, bword, ' ' );
                    m_PlyData[ m_PlyCnt++ ] = (float) atof ( bword );
                }
            } else {
                readword ( buf, bword, ' ' );
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

	app_printf ( "Computing normals.\n");
	ComputeNormals ();
	app_printf ( "Updating VBOs.\n");
	UpdateVBO();

	return 1;
}

int nvMesh::AddVert ( float x, float y, float z, float tx, float ty )
{
	Vertex v;	
	v.x = x; v.y = y; v.z = z;
	v.nx = v.x; v.ny = v.y; v.nz = v.z;
	float d = v.nx*v.nx+v.ny*v.ny+v.nz*v.nz;
	if ( d > 0 ) { d = sqrt(d); v.nx /= d; v.ny /= d; v.nz /= d; }
	
	v.tx = tx; v.ty = ty; 
	
	mVertices.push_back ( v );
	return mVertices.size()-1;
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

void nvMesh::UpdateVBO ()
{
	int numv = mVertices.size();
	int numf = mNumFaces;

	#ifdef USE_DX
		if ( mVBO.size() == 0 ) {
			mVBO.push_back ( 0 );		// vertices
			mVBO.push_back ( 0 );		// faces	
		} else {
			mVBO[0]->Release ();		
			mVBO[1]->Release ();		
		}
		D3D11_BUFFER_DESC bd; 
		ZeroMemory( &bd, sizeof(bd) ); 
		bd.Usage = D3D11_USAGE_DEFAULT; 
		bd.ByteWidth = numv * sizeof(Vertex); 
		bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
		bd.StructureByteStride = 0;
		D3D11_SUBRESOURCE_DATA InitData; 
		ZeroMemory( &InitData, sizeof(InitData) ); InitData.pSysMem = &mVertices[0].x;
		checkHR ( g_pDevice->CreateBuffer( &bd, &InitData, &mVBO[0] ) );				
		bd.ByteWidth = numf * mNumSides * sizeof(unsigned int);
		bd.BindFlags = D3D11_BIND_INDEX_BUFFER; 
		bd.StructureByteStride = 0; //sizeof(unsigned int);
		ZeroMemory( &InitData, sizeof(InitData) ); InitData.pSysMem = &mFaceVN[0];
		checkHR ( g_pDevice->CreateBuffer( &bd, &InitData, &mVBO[1] ) );

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
		glBufferData ( GL_ARRAY_BUFFER, numv * sizeof(Vertex), &mVertices[0].x, GL_STATIC_DRAW_ARB);
		glEnableVertexAttribArray ( 0 );	
		glVertexAttribPointer ( localPos, 3, GL_FLOAT, false, sizeof(Vertex), 0 );
		glEnableVertexAttribArray ( 1 );		
		glVertexAttribPointer ( localNorm, 3, GL_FLOAT, false, sizeof(Vertex), (void*) 12 );
		glEnableVertexAttribArray ( 2 );		
		glVertexAttribPointer ( localUV, 2, GL_FLOAT, false, sizeof(Vertex), (void*) 24 );
		glBindBuffer ( GL_ELEMENT_ARRAY_BUFFER, mVBO[1] );		
		glBufferData ( GL_ELEMENT_ARRAY_BUFFER, numf*mNumSides*sizeof(int), &mFaceVN[0], GL_STATIC_DRAW_ARB);

		glBindVertexArray ( 0 );

	#endif
}

void nvMesh::SelectVBO ( )
{
	#ifdef USE_DX
		UINT stride[3];	
		UINT offset[3];
		ID3D11Buffer* vptr[3];
		vptr[0] = mVBO[0];		stride[0] = sizeof(Vertex);		offset[0] = 0;		// Pos		
		vptr[1] = mVBO[0];		stride[1] = sizeof(Vertex);		offset[1] = 12;		// Normal
		vptr[2] = mVBO[0];		stride[2] = sizeof(Vertex);		offset[2] = 24;		// UV
		g_pContext->IASetVertexBuffers( 0, 3, vptr, stride, offset ); 				
		g_pContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST ); 	
		g_pContext->IASetIndexBuffer ( mVBO[1], DXGI_FORMAT_R32_UINT, 0 );
	#else
		glBindVertexArray ( mVAO );
		glBindBuffer ( GL_ARRAY_BUFFER, mVBO[0] );	
		glEnableVertexAttribArray ( 0 );	
		glEnableVertexAttribArray ( 1 );
		glEnableVertexAttribArray ( 2 );
		glBindBuffer ( GL_ELEMENT_ARRAY_BUFFER, mVBO[1] );
	#endif
}
void nvMesh::Draw ( int inst )
{
	#ifdef USE_DX		
		g_pContext->DrawIndexedInstanced ( mNumFaces*mNumSides, inst, 0, 0, 0 );
	#else
		glDrawElementsInstanced ( GL_TRIANGLES, mNumFaces*mNumSides, GL_UNSIGNED_INT, 0, inst );
	#endif
}
void nvMesh::DrawPatches ( int inst )
{
	#ifdef USE_DX
	#else
		glPatchParameteri( GL_PATCH_VERTICES, mNumSides );
		glDrawElementsInstanced ( GL_PATCHES, mNumFaces*mNumSides, GL_UNSIGNED_INT, 0, inst );
	#endif
}



/*
LodePNG version 20130128

Copyright (c) 2005-2013 Lode Vandevenne

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

    1. The origin of this software must not be misrepresented; you must not
    claim that you wrote the original software. If you use this software
    in a product, an acknowledgment in the product documentation would be
    appreciated but is not required.

    2. Altered source versions must be plainly marked as such, and must not be
    misrepresented as being the original software.

    3. This notice may not be removed or altered from any source
    distribution.
*/

/*
The manual and changelog are in the header file "lodepng.h"
Rename this file to lodepng.cpp to use it for C++, or to lodepng.c to use it for C.
*/

#include <stdio.h>
#include <stdlib.h>

#ifdef LODEPNG_COMPILE_CPP
#include <fstream>
#endif /*LODEPNG_COMPILE_CPP*/

#define VERSION_STRING "20130128"

/*
This source file is built up in the following large parts. The code sections
with the "LODEPNG_COMPILE_" #defines divide this up further in an intermixed way.
-Tools for C and common code for PNG and Zlib
-C Code for Zlib (huffman, deflate, ...)
-C Code for PNG (file format chunks, adam7, PNG filters, color conversions, ...)
-The C++ wrapper around all of the above
*/

/*The malloc, realloc and free functions defined here with "my" in front of the
name, so that you can easily change them to others related to your platform in
this one location if needed. Everything else in the code calls these.*/

static void* mymalloc(size_t size)
{
  return malloc(size);
}

static void* myrealloc(void* ptr, size_t new_size)
{
  return realloc(ptr, new_size);
}

static void myfree(void* ptr)
{
  free(ptr);
}

/* ////////////////////////////////////////////////////////////////////////// */
/* ////////////////////////////////////////////////////////////////////////// */
/* // Tools for C, and common code for PNG and Zlib.                       // */
/* ////////////////////////////////////////////////////////////////////////// */
/* ////////////////////////////////////////////////////////////////////////// */

/*
Often in case of an error a value is assigned to a variable and then it breaks
out of a loop (to go to the cleanup phase of a function). This macro does that.
It makes the error handling code shorter and more readable.

Example: if(!uivector_resizev(&frequencies_ll, 286, 0)) ERROR_BREAK(83);
*/
#define CERROR_BREAK(errorvar, code)\
{\
  errorvar = code;\
  break;\
}

/*version of CERROR_BREAK that assumes the common case where the error variable is named "error"*/
#define ERROR_BREAK(code) CERROR_BREAK(error, code)

/*Set error var to the error code, and return it.*/
#define CERROR_RETURN_ERROR(errorvar, code)\
{\
  errorvar = code;\
  return code;\
}

/*Try the code, if it returns error, also return the error.*/
#define CERROR_TRY_RETURN(call)\
{\
  unsigned error = call;\
  if(error) return error;\
}

/*
About uivector, ucvector and string:
-All of them wrap dynamic arrays or text strings in a similar way.
-LodePNG was originally written in C++. The vectors replace the std::vectors that were used in the C++ version.
-The string tools are made to avoid problems with compilers that declare things like strncat as deprecated.
-They're not used in the interface, only internally in this file as static functions.
-As with many other structs in this file, the init and cleanup functions serve as ctor and dtor.
*/

#ifdef LODEPNG_COMPILE_ZLIB
/*dynamic vector of unsigned ints*/
typedef struct uivector
{
  unsigned* data;
  size_t size; /*size in number of unsigned longs*/
  size_t allocsize; /*allocated size in bytes*/
} uivector;

static void uivector_cleanup(void* p)
{
  ((uivector*)p)->size = ((uivector*)p)->allocsize = 0;
  myfree(((uivector*)p)->data);
  ((uivector*)p)->data = NULL;
}

/*returns 1 if success, 0 if failure ==> nothing done*/
static unsigned uivector_resize(uivector* p, size_t size)
{
  if(size * sizeof(unsigned) > p->allocsize)
  {
    size_t newsize = size * sizeof(unsigned) * 2;
    void* data = myrealloc(p->data, newsize);
    if(data)
    {
      p->allocsize = newsize;
      p->data = (unsigned*)data;
      p->size = size;
    }
    else return 0;
  }
  else p->size = size;
  return 1;
}

/*resize and give all new elements the value*/
static unsigned uivector_resizev(uivector* p, size_t size, unsigned value)
{
  size_t oldsize = p->size, i;
  if(!uivector_resize(p, size)) return 0;
  for(i = oldsize; i < size; i++) p->data[i] = value;
  return 1;
}

static void uivector_init(uivector* p)
{
  p->data = NULL;
  p->size = p->allocsize = 0;
}

#ifdef LODEPNG_COMPILE_ENCODER
/*returns 1 if success, 0 if failure ==> nothing done*/
static unsigned uivector_push_back(uivector* p, unsigned c)
{
  if(!uivector_resize(p, p->size + 1)) return 0;
  p->data[p->size - 1] = c;
  return 1;
}

/*copy q to p, returns 1 if success, 0 if failure ==> nothing done*/
static unsigned uivector_copy(uivector* p, const uivector* q)
{
  size_t i;
  if(!uivector_resize(p, q->size)) return 0;
  for(i = 0; i < q->size; i++) p->data[i] = q->data[i];
  return 1;
}

static void uivector_swap(uivector* p, uivector* q)
{
  size_t tmp;
  unsigned* tmpp;
  tmp = p->size; p->size = q->size; q->size = tmp;
  tmp = p->allocsize; p->allocsize = q->allocsize; q->allocsize = tmp;
  tmpp = p->data; p->data = q->data; q->data = tmpp;
}
#endif /*LODEPNG_COMPILE_ENCODER*/
#endif /*LODEPNG_COMPILE_ZLIB*/

/* /////////////////////////////////////////////////////////////////////////// */

/*dynamic vector of unsigned chars*/
typedef struct ucvector
{
  unsigned char* data;
  size_t size; /*used size*/
  size_t allocsize; /*allocated size*/
} ucvector;

/*returns 1 if success, 0 if failure ==> nothing done*/
static unsigned ucvector_resize(ucvector* p, size_t size)
{
  if(size * sizeof(unsigned char) > p->allocsize)
  {
    size_t newsize = size * sizeof(unsigned char) * 2;
    void* data = myrealloc(p->data, newsize);
    if(data)
    {
      p->allocsize = newsize;
      p->data = (unsigned char*)data;
      p->size = size;
    }
    else return 0; /*error: not enough memory*/
  }
  else p->size = size;
  return 1;
}

#ifdef LODEPNG_COMPILE_PNG

static void ucvector_cleanup(void* p)
{
  ((ucvector*)p)->size = ((ucvector*)p)->allocsize = 0;
  myfree(((ucvector*)p)->data);
  ((ucvector*)p)->data = NULL;
}

static void ucvector_init(ucvector* p)
{
  p->data = NULL;
  p->size = p->allocsize = 0;
}

#ifdef LODEPNG_COMPILE_DECODER
/*resize and give all new elements the value*/
static unsigned ucvector_resizev(ucvector* p, size_t size, unsigned char value)
{
  size_t oldsize = p->size, i;
  if(!ucvector_resize(p, size)) return 0;
  for(i = oldsize; i < size; i++) p->data[i] = value;
  return 1;
}
#endif /*LODEPNG_COMPILE_DECODER*/
#endif /*LODEPNG_COMPILE_PNG*/

#ifdef LODEPNG_COMPILE_ZLIB
/*you can both convert from vector to buffer&size and vica versa. If you use
init_buffer to take over a buffer and size, it is not needed to use cleanup*/
static void ucvector_init_buffer(ucvector* p, unsigned char* buffer, size_t size)
{
  p->data = buffer;
  p->allocsize = p->size = size;
}
#endif /*LODEPNG_COMPILE_ZLIB*/

#if (defined(LODEPNG_COMPILE_PNG) && defined(LODEPNG_COMPILE_ANCILLARY_CHUNKS)) || defined(LODEPNG_COMPILE_ENCODER)
/*returns 1 if success, 0 if failure ==> nothing done*/
static unsigned ucvector_push_back(ucvector* p, unsigned char c)
{
  if(!ucvector_resize(p, p->size + 1)) return 0;
  p->data[p->size - 1] = c;
  return 1;
}
#endif /*defined(LODEPNG_COMPILE_PNG) || defined(LODEPNG_COMPILE_ENCODER)*/


/* ////////////////////////////////////////////////////////////////////////// */

#ifdef LODEPNG_COMPILE_PNG
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
/*returns 1 if success, 0 if failure ==> nothing done*/
static unsigned string_resize(char** out, size_t size)
{
  char* data = (char*)myrealloc(*out, size + 1);
  if(data)
  {
    data[size] = 0; /*null termination char*/
    *out = data;
  }
  return data != 0;
}

/*init a {char*, size_t} pair for use as string*/
static void string_init(char** out)
{
  *out = NULL;
  string_resize(out, 0);
}

/*free the above pair again*/
static void string_cleanup(char** out)
{
  myfree(*out);
  *out = NULL;
}

static void string_set(char** out, const char* in)
{
  size_t insize = strlen(in), i = 0;
  if(string_resize(out, insize))
  {
    for(i = 0; i < insize; i++)
    {
      (*out)[i] = in[i];
    }
  }
}
#endif /*LODEPNG_COMPILE_ANCILLARY_CHUNKS*/
#endif /*LODEPNG_COMPILE_PNG*/

/* ////////////////////////////////////////////////////////////////////////// */

unsigned lodepng_read32bitInt(const unsigned char* buffer)
{
  return (buffer[0] << 24) | (buffer[1] << 16) | (buffer[2] << 8) | buffer[3];
}

#if defined(LODEPNG_COMPILE_PNG) || defined(LODEPNG_COMPILE_ENCODER)
/*buffer must have at least 4 allocated bytes available*/
static void lodepng_set32bitInt(unsigned char* buffer, unsigned value)
{
  buffer[0] = (unsigned char)((value >> 24) & 0xff);
  buffer[1] = (unsigned char)((value >> 16) & 0xff);
  buffer[2] = (unsigned char)((value >>  8) & 0xff);
  buffer[3] = (unsigned char)((value      ) & 0xff);
}
#endif /*defined(LODEPNG_COMPILE_PNG) || defined(LODEPNG_COMPILE_ENCODER)*/

#ifdef LODEPNG_COMPILE_ENCODER
static void lodepng_add32bitInt(ucvector* buffer, unsigned value)
{
  ucvector_resize(buffer, buffer->size + 4); /*todo: give error if resize failed*/
  lodepng_set32bitInt(&buffer->data[buffer->size - 4], value);
}
#endif /*LODEPNG_COMPILE_ENCODER*/

/* ////////////////////////////////////////////////////////////////////////// */
/* / File IO                                                                / */
/* ////////////////////////////////////////////////////////////////////////// */

#ifdef LODEPNG_COMPILE_DISK

unsigned lodepng_load_file(unsigned char** out, size_t* outsize, const char* filename)
{
  FILE* file;
  long size;

  /*provide some proper output values if error will happen*/
  *out = 0;
  *outsize = 0;

  file = fopen(filename, "rb");
  if(!file) return 78;

  /*get filesize:*/
  fseek(file , 0 , SEEK_END);
  size = ftell(file);
  rewind(file);

  /*read contents of the file into the vector*/
  *outsize = 0;
  *out = (unsigned char*)mymalloc((size_t)size);
  if(size && (*out)) (*outsize) = fread(*out, 1, (size_t)size, file);

  fclose(file);
  if(!(*out) && size) return 83; /*the above malloc failed*/
  return 0;
}

/*write given buffer to the file, overwriting the file, it doesn't append to it.*/
unsigned lodepng_save_file(const unsigned char* buffer, size_t buffersize, const char* filename)
{
  FILE* file;
  file = fopen(filename, "wb" );
  if(!file) return 79;
  fwrite((char*)buffer , 1 , buffersize, file);
  fclose(file);
  return 0;
}

#endif /*LODEPNG_COMPILE_DISK*/

/* ////////////////////////////////////////////////////////////////////////// */
/* ////////////////////////////////////////////////////////////////////////// */
/* // End of common code and tools. Begin of Zlib related code.            // */
/* ////////////////////////////////////////////////////////////////////////// */
/* ////////////////////////////////////////////////////////////////////////// */

#ifdef LODEPNG_COMPILE_ZLIB
#ifdef LODEPNG_COMPILE_ENCODER
/*TODO: this ignores potential out of memory errors*/
static void addBitToStream(size_t* bitpointer, ucvector* bitstream, unsigned char bit)
{
  /*add a new byte at the end*/
  if((*bitpointer) % 8 == 0) ucvector_push_back(bitstream, (unsigned char)0);
  /*earlier bit of huffman code is in a lesser significant bit of an earlier byte*/
  (bitstream->data[bitstream->size - 1]) |= (bit << ((*bitpointer) & 0x7));
  (*bitpointer)++;
}

static void addBitsToStream(size_t* bitpointer, ucvector* bitstream, unsigned value, size_t nbits)
{
  size_t i;
  for(i = 0; i < nbits; i++) addBitToStream(bitpointer, bitstream, (unsigned char)((value >> i) & 1));
}

static void addBitsToStreamReversed(size_t* bitpointer, ucvector* bitstream, unsigned value, size_t nbits)
{
  size_t i;
  for(i = 0; i < nbits; i++) addBitToStream(bitpointer, bitstream, (unsigned char)((value >> (nbits - 1 - i)) & 1));
}
#endif /*LODEPNG_COMPILE_ENCODER*/

#ifdef LODEPNG_COMPILE_DECODER

#define READBIT(bitpointer, bitstream) ((bitstream[bitpointer >> 3] >> (bitpointer & 0x7)) & (unsigned char)1)

static unsigned char readBitFromStream(size_t* bitpointer, const unsigned char* bitstream)
{
  unsigned char result = (unsigned char)(READBIT(*bitpointer, bitstream));
  (*bitpointer)++;
  return result;
}

static unsigned readBitsFromStream(size_t* bitpointer, const unsigned char* bitstream, size_t nbits)
{
  unsigned result = 0, i;
  for(i = 0; i < nbits; i++)
  {
    result += ((unsigned)READBIT(*bitpointer, bitstream)) << i;
    (*bitpointer)++;
  }
  return result;
}
#endif /*LODEPNG_COMPILE_DECODER*/

/* ////////////////////////////////////////////////////////////////////////// */
/* / Deflate - Huffman                                                      / */
/* ////////////////////////////////////////////////////////////////////////// */

#define FIRST_LENGTH_CODE_INDEX 257
#define LAST_LENGTH_CODE_INDEX 285
/*256 literals, the end code, some length codes, and 2 unused codes*/
#define NUM_DEFLATE_CODE_SYMBOLS 288
/*the distance codes have their own symbols, 30 used, 2 unused*/
#define NUM_DISTANCE_SYMBOLS 32
/*the code length codes. 0-15: code lengths, 16: copy previous 3-6 times, 17: 3-10 zeros, 18: 11-138 zeros*/
#define NUM_CODE_LENGTH_CODES 19

/*the base lengths represented by codes 257-285*/
static const unsigned LENGTHBASE[29]
  = {3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59,
     67, 83, 99, 115, 131, 163, 195, 227, 258};

/*the extra bits used by codes 257-285 (added to base length)*/
static const unsigned LENGTHEXTRA[29]
  = {0, 0, 0, 0, 0, 0, 0,  0,  1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3,
      4,  4,  4,   4,   5,   5,   5,   5,   0};

/*the base backwards distances (the bits of distance codes appear after length codes and use their own huffman tree)*/
static const unsigned DISTANCEBASE[30]
  = {1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513,
     769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};

/*the extra bits of backwards distances (added to base)*/
static const unsigned DISTANCEEXTRA[30]
  = {0, 0, 0, 0, 1, 1, 2,  2,  3,  3,  4,  4,  5,  5,   6,   6,   7,   7,   8,
       8,    9,    9,   10,   10,   11,   11,   12,    12,    13,    13};

/*the order in which "code length alphabet code lengths" are stored, out of this
the huffman tree of the dynamic huffman tree lengths is generated*/
static const unsigned CLCL_ORDER[NUM_CODE_LENGTH_CODES]
  = {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};

/* ////////////////////////////////////////////////////////////////////////// */

/*
Huffman tree struct, containing multiple representations of the tree
*/
typedef struct HuffmanTree
{
  unsigned* tree2d;
  unsigned* tree1d;
  unsigned* lengths; /*the lengths of the codes of the 1d-tree*/
  unsigned maxbitlen; /*maximum number of bits a single code can get*/
  unsigned numcodes; /*number of symbols in the alphabet = number of codes*/
} HuffmanTree;

/*function used for debug purposes to draw the tree in ascii art with C++*/
/*#include <iostream>
static void HuffmanTree_draw(HuffmanTree* tree)
{
  std::cout << "tree. length: " << tree->numcodes << " maxbitlen: " << tree->maxbitlen << std::endl;
  for(size_t i = 0; i < tree->tree1d.size; i++)
  {
    if(tree->lengths.data[i])
      std::cout << i << " " << tree->tree1d.data[i] << " " << tree->lengths.data[i] << std::endl;
  }
  std::cout << std::endl;
}*/

static void HuffmanTree_init(HuffmanTree* tree)
{
  tree->tree2d = 0;
  tree->tree1d = 0;
  tree->lengths = 0;
}

static void HuffmanTree_cleanup(HuffmanTree* tree)
{
  myfree(tree->tree2d);
  myfree(tree->tree1d);
  myfree(tree->lengths);
}

/*the tree representation used by the decoder. return value is error*/
static unsigned HuffmanTree_make2DTree(HuffmanTree* tree)
{
  unsigned nodefilled = 0; /*up to which node it is filled*/
  unsigned treepos = 0; /*position in the tree (1 of the numcodes columns)*/
  unsigned n, i;

  tree->tree2d = (unsigned*)mymalloc(tree->numcodes * 2 * sizeof(unsigned));
  if(!tree->tree2d) return 83; /*alloc fail*/

  /*
  convert tree1d[] to tree2d[][]. In the 2D array, a value of 32767 means
  uninited, a value >= numcodes is an address to another bit, a value < numcodes
  is a code. The 2 rows are the 2 possible bit values (0 or 1), there are as
  many columns as codes - 1.
  A good huffmann tree has N * 2 - 1 nodes, of which N - 1 are internal nodes.
  Here, the internal nodes are stored (what their 0 and 1 option point to).
  There is only memory for such good tree currently, if there are more nodes
  (due to too long length codes), error 55 will happen
  */
  for(n = 0; n < tree->numcodes * 2; n++)
  {
    tree->tree2d[n] = 32767; /*32767 here means the tree2d isn't filled there yet*/
  }

  for(n = 0; n < tree->numcodes; n++) /*the codes*/
  {
    for(i = 0; i < tree->lengths[n]; i++) /*the bits for this code*/
    {
      unsigned char bit = (unsigned char)((tree->tree1d[n] >> (tree->lengths[n] - i - 1)) & 1);
      if(treepos > tree->numcodes - 2) return 55; /*oversubscribed, see comment in lodepng_error_text*/
      if(tree->tree2d[2 * treepos + bit] == 32767) /*not yet filled in*/
      {
        if(i + 1 == tree->lengths[n]) /*last bit*/
        {
          tree->tree2d[2 * treepos + bit] = n; /*put the current code in it*/
          treepos = 0;
        }
        else
        {
          /*put address of the next step in here, first that address has to be found of course
          (it's just nodefilled + 1)...*/
          nodefilled++;
          /*addresses encoded with numcodes added to it*/
          tree->tree2d[2 * treepos + bit] = nodefilled + tree->numcodes;
          treepos = nodefilled;
        }
      }
      else treepos = tree->tree2d[2 * treepos + bit] - tree->numcodes;
    }
  }

  for(n = 0;  n < tree->numcodes * 2; n++)
  {
    if(tree->tree2d[n] == 32767) tree->tree2d[n] = 0; /*remove possible remaining 32767's*/
  }

  return 0;
}

/*
Second step for the ...makeFromLengths and ...makeFromFrequencies functions.
numcodes, lengths and maxbitlen must already be filled in correctly. return
value is error.
*/
static unsigned HuffmanTree_makeFromLengths2(HuffmanTree* tree)
{
  uivector blcount;
  uivector nextcode;
  unsigned bits, n, error = 0;

  uivector_init(&blcount);
  uivector_init(&nextcode);

  tree->tree1d = (unsigned*)mymalloc(tree->numcodes * sizeof(unsigned));
  if(!tree->tree1d) error = 83; /*alloc fail*/

  if(!uivector_resizev(&blcount, tree->maxbitlen + 1, 0)
  || !uivector_resizev(&nextcode, tree->maxbitlen + 1, 0))
    error = 83; /*alloc fail*/

  if(!error)
  {
    /*step 1: count number of instances of each code length*/
    for(bits = 0; bits < tree->numcodes; bits++) blcount.data[tree->lengths[bits]]++;
    /*step 2: generate the nextcode values*/
    for(bits = 1; bits <= tree->maxbitlen; bits++)
    {
      nextcode.data[bits] = (nextcode.data[bits - 1] + blcount.data[bits - 1]) << 1;
    }
    /*step 3: generate all the codes*/
    for(n = 0; n < tree->numcodes; n++)
    {
      if(tree->lengths[n] != 0) tree->tree1d[n] = nextcode.data[tree->lengths[n]]++;
    }
  }

  uivector_cleanup(&blcount);
  uivector_cleanup(&nextcode);

  if(!error) return HuffmanTree_make2DTree(tree);
  else return error;
}

/*
given the code lengths (as stored in the PNG file), generate the tree as defined
by Deflate. maxbitlen is the maximum bits that a code in the tree can have.
return value is error.
*/
static unsigned HuffmanTree_makeFromLengths(HuffmanTree* tree, const unsigned* bitlen,
                                            size_t numcodes, unsigned maxbitlen)
{
  unsigned i;
  tree->lengths = (unsigned*)mymalloc(numcodes * sizeof(unsigned));
  if(!tree->lengths) return 83; /*alloc fail*/
  for(i = 0; i < numcodes; i++) tree->lengths[i] = bitlen[i];
  tree->numcodes = (unsigned)numcodes; /*number of symbols*/
  tree->maxbitlen = maxbitlen;
  return HuffmanTree_makeFromLengths2(tree);
}

#ifdef LODEPNG_COMPILE_ENCODER

/*
A coin, this is the terminology used for the package-merge algorithm and the
coin collector's problem. This is used to generate the huffman tree.
A coin can be multiple coins (when they're merged)
*/
typedef struct Coin
{
  uivector symbols;
  float weight; /*the sum of all weights in this coin*/
} Coin;

static void coin_init(Coin* c)
{
  uivector_init(&c->symbols);
}

/*argument c is void* so that this dtor can be given as function pointer to the vector resize function*/
static void coin_cleanup(void* c)
{
  uivector_cleanup(&((Coin*)c)->symbols);
}

static void coin_copy(Coin* c1, const Coin* c2)
{
  c1->weight = c2->weight;
  uivector_copy(&c1->symbols, &c2->symbols);
}

static void add_coins(Coin* c1, const Coin* c2)
{
  size_t i;
  for(i = 0; i < c2->symbols.size; i++) uivector_push_back(&c1->symbols, c2->symbols.data[i]);
  c1->weight += c2->weight;
}

static void init_coins(Coin* coins, size_t num)
{
  size_t i;
  for(i = 0; i < num; i++) coin_init(&coins[i]);
}

static void cleanup_coins(Coin* coins, size_t num)
{
  size_t i;
  for(i = 0; i < num; i++) coin_cleanup(&coins[i]);
}

/*
This uses a simple combsort to sort the data. This function is not critical for
overall encoding speed and the data amount isn't that large.
*/
static void sort_coins(Coin* data, size_t amount)
{
  size_t gap = amount;
  unsigned char swapped = 0;
  while((gap > 1) || swapped)
  {
    size_t i;
    gap = (gap * 10) / 13; /*shrink factor 1.3*/
    if(gap == 9 || gap == 10) gap = 11; /*combsort11*/
    if(gap < 1) gap = 1;
    swapped = 0;
    for(i = 0; i < amount - gap; i++)
    {
      size_t j = i + gap;
      if(data[j].weight < data[i].weight)
      {
        float temp = data[j].weight; data[j].weight = data[i].weight; data[i].weight = temp;
        uivector_swap(&data[i].symbols, &data[j].symbols);
        swapped = 1;
      }
    }
  }
}

static unsigned append_symbol_coins(Coin* coins, const unsigned* frequencies, unsigned numcodes, size_t sum)
{
  unsigned i;
  unsigned j = 0; /*index of present symbols*/
  for(i = 0; i < numcodes; i++)
  {
    if(frequencies[i] != 0) /*only include symbols that are present*/
    {
      coins[j].weight = frequencies[i] / (float)sum;
      uivector_push_back(&coins[j].symbols, i);
      j++;
    }
  }
  return 0;
}

unsigned lodepng_huffman_code_lengths(unsigned* lengths, const unsigned* frequencies,
                                      size_t numcodes, unsigned maxbitlen)
{
  unsigned i, j;
  size_t sum = 0, numpresent = 0;
  unsigned error = 0;
  Coin* coins; /*the coins of the currently calculated row*/
  Coin* prev_row; /*the previous row of coins*/
  unsigned numcoins;
  unsigned coinmem;

  if(numcodes == 0) return 80; /*error: a tree of 0 symbols is not supposed to be made*/

  for(i = 0; i < numcodes; i++)
  {
    if(frequencies[i] > 0)
    {
      numpresent++;
      sum += frequencies[i];
    }
  }

  for(i = 0; i < numcodes; i++) lengths[i] = 0;

  /*ensure at least two present symbols. There should be at least one symbol
  according to RFC 1951 section 3.2.7. To decoders incorrectly require two. To
  make these work as well ensure there are at least two symbols. The
  Package-Merge code below also doesn't work correctly if there's only one
  symbol, it'd give it the theoritical 0 bits but in practice zlib wants 1 bit*/
  if(numpresent == 0)
  {
    lengths[0] = lengths[1] = 1; /*note that for RFC 1951 section 3.2.7, only lengths[0] = 1 is needed*/
  }
  else if(numpresent == 1)
  {
    for(i = 0; i < numcodes; i++)
    {
      if(frequencies[i])
      {
        lengths[i] = 1;
        lengths[i == 0 ? 1 : 0] = 1;
        break;
      }
    }
  }
  else
  {
    /*Package-Merge algorithm represented by coin collector's problem
    For every symbol, maxbitlen coins will be created*/

    coinmem = numpresent * 2; /*max amount of coins needed with the current algo*/
    coins = (Coin*)mymalloc(sizeof(Coin) * coinmem);
    prev_row = (Coin*)mymalloc(sizeof(Coin) * coinmem);
    if(!coins || !prev_row) return 83; /*alloc fail*/
    init_coins(coins, coinmem);
    init_coins(prev_row, coinmem);

    /*first row, lowest denominator*/
    error = append_symbol_coins(coins, frequencies, numcodes, sum);
    numcoins = numpresent;
    sort_coins(coins, numcoins);
    if(!error)
    {
      unsigned numprev = 0;
      for(j = 1; j <= maxbitlen && !error; j++) /*each of the remaining rows*/
      {
        unsigned tempnum;
        Coin* tempcoins;
        /*swap prev_row and coins, and their amounts*/
        tempcoins = prev_row; prev_row = coins; coins = tempcoins;
        tempnum = numprev; numprev = numcoins; numcoins = tempnum;

        cleanup_coins(coins, numcoins);
        init_coins(coins, numcoins);

        numcoins = 0;

        /*fill in the merged coins of the previous row*/
        for(i = 0; i + 1 < numprev; i += 2)
        {
          /*merge prev_row[i] and prev_row[i + 1] into new coin*/
          Coin* coin = &coins[numcoins++];
          coin_copy(coin, &prev_row[i]);
          add_coins(coin, &prev_row[i + 1]);
        }
        /*fill in all the original symbols again*/
        if(j < maxbitlen)
        {
          error = append_symbol_coins(coins + numcoins, frequencies, numcodes, sum);
          numcoins += numpresent;
        }
        sort_coins(coins, numcoins);
      }
    }

    if(!error)
    {
      /*calculate the lenghts of each symbol, as the amount of times a coin of each symbol is used*/
      for(i = 0; i < numpresent - 1; i++)
      {
        Coin* coin = &coins[i];
        for(j = 0; j < coin->symbols.size; j++) lengths[coin->symbols.data[j]]++;
      }
    }

    cleanup_coins(coins, coinmem);
    myfree(coins);
    cleanup_coins(prev_row, coinmem);
    myfree(prev_row);
  }

  return error;
}

/*Create the Huffman tree given the symbol frequencies*/
static unsigned HuffmanTree_makeFromFrequencies(HuffmanTree* tree, const unsigned* frequencies,
                                                size_t mincodes, size_t numcodes, unsigned maxbitlen)
{
  unsigned error = 0;
  while(!frequencies[numcodes - 1] && numcodes > mincodes) numcodes--; /*trim zeroes*/
  tree->maxbitlen = maxbitlen;
  tree->numcodes = (unsigned)numcodes; /*number of symbols*/
  tree->lengths = (unsigned*)myrealloc(tree->lengths, numcodes * sizeof(unsigned));
  if(!tree->lengths) return 83; /*alloc fail*/
  /*initialize all lengths to 0*/
  memset(tree->lengths, 0, numcodes * sizeof(unsigned));

  error = lodepng_huffman_code_lengths(tree->lengths, frequencies, numcodes, maxbitlen);
  if(!error) error = HuffmanTree_makeFromLengths2(tree);
  return error;
}

static unsigned HuffmanTree_getCode(const HuffmanTree* tree, unsigned index)
{
  return tree->tree1d[index];
}

static unsigned HuffmanTree_getLength(const HuffmanTree* tree, unsigned index)
{
  return tree->lengths[index];
}
#endif /*LODEPNG_COMPILE_ENCODER*/

/*get the literal and length code tree of a deflated block with fixed tree, as per the deflate specification*/
static unsigned generateFixedLitLenTree(HuffmanTree* tree)
{
  unsigned i, error = 0;
  unsigned* bitlen = (unsigned*)mymalloc(NUM_DEFLATE_CODE_SYMBOLS * sizeof(unsigned));
  if(!bitlen) return 83; /*alloc fail*/

  /*288 possible codes: 0-255=literals, 256=endcode, 257-285=lengthcodes, 286-287=unused*/
  for(i =   0; i <= 143; i++) bitlen[i] = 8;
  for(i = 144; i <= 255; i++) bitlen[i] = 9;
  for(i = 256; i <= 279; i++) bitlen[i] = 7;
  for(i = 280; i <= 287; i++) bitlen[i] = 8;

  error = HuffmanTree_makeFromLengths(tree, bitlen, NUM_DEFLATE_CODE_SYMBOLS, 15);

  myfree(bitlen);
  return error;
}

/*get the distance code tree of a deflated block with fixed tree, as specified in the deflate specification*/
static unsigned generateFixedDistanceTree(HuffmanTree* tree)
{
  unsigned i, error = 0;
  unsigned* bitlen = (unsigned*)mymalloc(NUM_DISTANCE_SYMBOLS * sizeof(unsigned));
  if(!bitlen) return 83; /*alloc fail*/

  /*there are 32 distance codes, but 30-31 are unused*/
  for(i = 0; i < NUM_DISTANCE_SYMBOLS; i++) bitlen[i] = 5;
  error = HuffmanTree_makeFromLengths(tree, bitlen, NUM_DISTANCE_SYMBOLS, 15);

  myfree(bitlen);
  return error;
}

#ifdef LODEPNG_COMPILE_DECODER

/*
returns the code, or (unsigned)(-1) if error happened
inbitlength is the length of the complete buffer, in bits (so its byte length times 8)
*/
static unsigned huffmanDecodeSymbol(const unsigned char* in, size_t* bp,
                                    const HuffmanTree* codetree, size_t inbitlength)
{
  unsigned treepos = 0, ct;
  for(;;)
  {
    if(*bp >= inbitlength) return (unsigned)(-1); /*error: end of input memory reached without endcode*/
    /*
    decode the symbol from the tree. The "readBitFromStream" code is inlined in
    the expression below because this is the biggest bottleneck while decoding
    */
    ct = codetree->tree2d[(treepos << 1) + READBIT(*bp, in)];
    (*bp)++;
    if(ct < codetree->numcodes) return ct; /*the symbol is decoded, return it*/
    else treepos = ct - codetree->numcodes; /*symbol not yet decoded, instead move tree position*/

    if(treepos >= codetree->numcodes) return (unsigned)(-1); /*error: it appeared outside the codetree*/
  }
}
#endif /*LODEPNG_COMPILE_DECODER*/

#ifdef LODEPNG_COMPILE_DECODER

/* ////////////////////////////////////////////////////////////////////////// */
/* / Inflator (Decompressor)                                                / */
/* ////////////////////////////////////////////////////////////////////////// */

/*get the tree of a deflated block with fixed tree, as specified in the deflate specification*/
static void getTreeInflateFixed(HuffmanTree* tree_ll, HuffmanTree* tree_d)
{
  /*TODO: check for out of memory errors*/
  generateFixedLitLenTree(tree_ll);
  generateFixedDistanceTree(tree_d);
}

/*get the tree of a deflated block with dynamic tree, the tree itself is also Huffman compressed with a known tree*/
static unsigned getTreeInflateDynamic(HuffmanTree* tree_ll, HuffmanTree* tree_d,
                                      const unsigned char* in, size_t* bp, size_t inlength)
{
  /*make sure that length values that aren't filled in will be 0, or a wrong tree will be generated*/
  unsigned error = 0;
  unsigned n, HLIT, HDIST, HCLEN, i;
  size_t inbitlength = inlength * 8;

  /*see comments in deflateDynamic for explanation of the context and these variables, it is analogous*/
  unsigned* bitlen_ll = 0; /*lit,len code lengths*/
  unsigned* bitlen_d = 0; /*dist code lengths*/
  /*code length code lengths ("clcl"), the bit lengths of the huffman tree used to compress bitlen_ll and bitlen_d*/
  unsigned* bitlen_cl = 0;
  HuffmanTree tree_cl; /*the code tree for code length codes (the huffman tree for compressed huffman trees)*/

  if((*bp) >> 3 >= inlength - 2) return 49; /*error: the bit pointer is or will go past the memory*/

  /*number of literal/length codes + 257. Unlike the spec, the value 257 is added to it here already*/
  HLIT =  readBitsFromStream(bp, in, 5) + 257;
  /*number of distance codes. Unlike the spec, the value 1 is added to it here already*/
  HDIST = readBitsFromStream(bp, in, 5) + 1;
  /*number of code length codes. Unlike the spec, the value 4 is added to it here already*/
  HCLEN = readBitsFromStream(bp, in, 4) + 4;

  HuffmanTree_init(&tree_cl);

  while(!error)
  {
    /*read the code length codes out of 3 * (amount of code length codes) bits*/

    bitlen_cl = (unsigned*)mymalloc(NUM_CODE_LENGTH_CODES * sizeof(unsigned));
    if(!bitlen_cl) ERROR_BREAK(83 /*alloc fail*/);

    for(i = 0; i < NUM_CODE_LENGTH_CODES; i++)
    {
      if(i < HCLEN) bitlen_cl[CLCL_ORDER[i]] = readBitsFromStream(bp, in, 3);
      else bitlen_cl[CLCL_ORDER[i]] = 0; /*if not, it must stay 0*/
    }

    error = HuffmanTree_makeFromLengths(&tree_cl, bitlen_cl, NUM_CODE_LENGTH_CODES, 7);
    if(error) break;

    /*now we can use this tree to read the lengths for the tree that this function will return*/
    bitlen_ll = (unsigned*)mymalloc(NUM_DEFLATE_CODE_SYMBOLS * sizeof(unsigned));
    bitlen_d = (unsigned*)mymalloc(NUM_DISTANCE_SYMBOLS * sizeof(unsigned));
    if(!bitlen_ll || !bitlen_d) ERROR_BREAK(83 /*alloc fail*/);
    for(i = 0; i < NUM_DEFLATE_CODE_SYMBOLS; i++) bitlen_ll[i] = 0;
    for(i = 0; i < NUM_DISTANCE_SYMBOLS; i++) bitlen_d[i] = 0;

    /*i is the current symbol we're reading in the part that contains the code lengths of lit/len and dist codes*/
    i = 0;
    while(i < HLIT + HDIST)
    {
      unsigned code = huffmanDecodeSymbol(in, bp, &tree_cl, inbitlength);
      if(code <= 15) /*a length code*/
      {
        if(i < HLIT) bitlen_ll[i] = code;
        else bitlen_d[i - HLIT] = code;
        i++;
      }
      else if(code == 16) /*repeat previous*/
      {
        unsigned replength = 3; /*read in the 2 bits that indicate repeat length (3-6)*/
        unsigned value; /*set value to the previous code*/

        if(*bp >= inbitlength) ERROR_BREAK(50); /*error, bit pointer jumps past memory*/
        if (i == 0) ERROR_BREAK(54); /*can't repeat previous if i is 0*/

        replength += readBitsFromStream(bp, in, 2);

        if(i < HLIT + 1) value = bitlen_ll[i - 1];
        else value = bitlen_d[i - HLIT - 1];
        /*repeat this value in the next lengths*/
        for(n = 0; n < replength; n++)
        {
          if(i >= HLIT + HDIST) ERROR_BREAK(13); /*error: i is larger than the amount of codes*/
          if(i < HLIT) bitlen_ll[i] = value;
          else bitlen_d[i - HLIT] = value;
          i++;
        }
      }
      else if(code == 17) /*repeat "0" 3-10 times*/
      {
        unsigned replength = 3; /*read in the bits that indicate repeat length*/
        if(*bp >= inbitlength) ERROR_BREAK(50); /*error, bit pointer jumps past memory*/

        replength += readBitsFromStream(bp, in, 3);

        /*repeat this value in the next lengths*/
        for(n = 0; n < replength; n++)
        {
          if(i >= HLIT + HDIST) ERROR_BREAK(14); /*error: i is larger than the amount of codes*/

          if(i < HLIT) bitlen_ll[i] = 0;
          else bitlen_d[i - HLIT] = 0;
          i++;
        }
      }
      else if(code == 18) /*repeat "0" 11-138 times*/
      {
        unsigned replength = 11; /*read in the bits that indicate repeat length*/
        if(*bp >= inbitlength) ERROR_BREAK(50); /*error, bit pointer jumps past memory*/

        replength += readBitsFromStream(bp, in, 7);

        /*repeat this value in the next lengths*/
        for(n = 0; n < replength; n++)
        {
          if(i >= HLIT + HDIST) ERROR_BREAK(15); /*error: i is larger than the amount of codes*/

          if(i < HLIT) bitlen_ll[i] = 0;
          else bitlen_d[i - HLIT] = 0;
          i++;
        }
      }
      else /*if(code == (unsigned)(-1))*/ /*huffmanDecodeSymbol returns (unsigned)(-1) in case of error*/
      {
        if(code == (unsigned)(-1))
        {
          /*return error code 10 or 11 depending on the situation that happened in huffmanDecodeSymbol
          (10=no endcode, 11=wrong jump outside of tree)*/
          error = (*bp) > inbitlength ? 10 : 11;
        }
        else error = 16; /*unexisting code, this can never happen*/
        break;
      }
    }
    if(error) break;

    if(bitlen_ll[256] == 0) ERROR_BREAK(64); /*the length of the end code 256 must be larger than 0*/

    /*now we've finally got HLIT and HDIST, so generate the code trees, and the function is done*/
    error = HuffmanTree_makeFromLengths(tree_ll, bitlen_ll, NUM_DEFLATE_CODE_SYMBOLS, 15);
    if(error) break;
    error = HuffmanTree_makeFromLengths(tree_d, bitlen_d, NUM_DISTANCE_SYMBOLS, 15);

    break; /*end of error-while*/
  }

  myfree(bitlen_cl);
  myfree(bitlen_ll);
  myfree(bitlen_d);
  HuffmanTree_cleanup(&tree_cl);

  return error;
}

/*inflate a block with dynamic of fixed Huffman tree*/
static unsigned inflateHuffmanBlock(ucvector* out, const unsigned char* in, size_t* bp,
                                    size_t* pos, size_t inlength, unsigned btype)
{
  unsigned error = 0;
  HuffmanTree tree_ll; /*the huffman tree for literal and length codes*/
  HuffmanTree tree_d; /*the huffman tree for distance codes*/
  size_t inbitlength = inlength * 8;

  HuffmanTree_init(&tree_ll);
  HuffmanTree_init(&tree_d);

  if(btype == 1) getTreeInflateFixed(&tree_ll, &tree_d);
  else if(btype == 2) error = getTreeInflateDynamic(&tree_ll, &tree_d, in, bp, inlength);

  while(!error) /*decode all symbols until end reached, breaks at end code*/
  {
    /*code_ll is literal, length or end code*/
    unsigned code_ll = huffmanDecodeSymbol(in, bp, &tree_ll, inbitlength);
    if(code_ll <= 255) /*literal symbol*/
    {
      if((*pos) >= out->size)
      {
        /*reserve more room at once*/
        if(!ucvector_resize(out, ((*pos) + 1) * 2)) ERROR_BREAK(83 /*alloc fail*/);
      }
      out->data[(*pos)] = (unsigned char)(code_ll);
      (*pos)++;
    }
    else if(code_ll >= FIRST_LENGTH_CODE_INDEX && code_ll <= LAST_LENGTH_CODE_INDEX) /*length code*/
    {
      unsigned code_d, distance;
      unsigned numextrabits_l, numextrabits_d; /*extra bits for length and distance*/
      size_t start, forward, backward, length;

      /*part 1: get length base*/
      length = LENGTHBASE[code_ll - FIRST_LENGTH_CODE_INDEX];

      /*part 2: get extra bits and add the value of that to length*/
      numextrabits_l = LENGTHEXTRA[code_ll - FIRST_LENGTH_CODE_INDEX];
      if(*bp >= inbitlength) ERROR_BREAK(51); /*error, bit pointer will jump past memory*/
      length += readBitsFromStream(bp, in, numextrabits_l);

      /*part 3: get distance code*/
      code_d = huffmanDecodeSymbol(in, bp, &tree_d, inbitlength);
      if(code_d > 29)
      {
        if(code_ll == (unsigned)(-1)) /*huffmanDecodeSymbol returns (unsigned)(-1) in case of error*/
        {
          /*return error code 10 or 11 depending on the situation that happened in huffmanDecodeSymbol
          (10=no endcode, 11=wrong jump outside of tree)*/
          error = (*bp) > inlength * 8 ? 10 : 11;
        }
        else error = 18; /*error: invalid distance code (30-31 are never used)*/
        break;
      }
      distance = DISTANCEBASE[code_d];

      /*part 4: get extra bits from distance*/
      numextrabits_d = DISTANCEEXTRA[code_d];
      if(*bp >= inbitlength) ERROR_BREAK(51); /*error, bit pointer will jump past memory*/

      distance += readBitsFromStream(bp, in, numextrabits_d);

      /*part 5: fill in all the out[n] values based on the length and dist*/
      start = (*pos);
      if(distance > start) ERROR_BREAK(52); /*too long backward distance*/
      backward = start - distance;
      if((*pos) + length >= out->size)
      {
        /*reserve more room at once*/
        if(!ucvector_resize(out, ((*pos) + length) * 2)) ERROR_BREAK(83 /*alloc fail*/);
      }

      for(forward = 0; forward < length; forward++)
      {
        out->data[(*pos)] = out->data[backward];
        (*pos)++;
        backward++;
        if(backward >= start) backward = start - distance;
      }
    }
    else if(code_ll == 256)
    {
      break; /*end code, break the loop*/
    }
    else /*if(code == (unsigned)(-1))*/ /*huffmanDecodeSymbol returns (unsigned)(-1) in case of error*/
    {
      /*return error code 10 or 11 depending on the situation that happened in huffmanDecodeSymbol
      (10=no endcode, 11=wrong jump outside of tree)*/
      error = (*bp) > inlength * 8 ? 10 : 11;
      break;
    }
  }

  HuffmanTree_cleanup(&tree_ll);
  HuffmanTree_cleanup(&tree_d);

  return error;
}

static unsigned inflateNoCompression(ucvector* out, const unsigned char* in, size_t* bp, size_t* pos, size_t inlength)
{
  /*go to first boundary of byte*/
  size_t p;
  unsigned LEN, NLEN, n, error = 0;
  while(((*bp) & 0x7) != 0) (*bp)++;
  p = (*bp) / 8; /*byte position*/

  /*read LEN (2 bytes) and NLEN (2 bytes)*/
  if(p >= inlength - 4) return 52; /*error, bit pointer will jump past memory*/
  LEN = in[p] + 256 * in[p + 1]; p += 2;
  NLEN = in[p] + 256 * in[p + 1]; p += 2;

  /*check if 16-bit NLEN is really the one's complement of LEN*/
  if(LEN + NLEN != 65535) return 21; /*error: NLEN is not one's complement of LEN*/

  if((*pos) + LEN >= out->size)
  {
    if(!ucvector_resize(out, (*pos) + LEN)) return 83; /*alloc fail*/
  }

  /*read the literal data: LEN bytes are now stored in the out buffer*/
  if(p + LEN > inlength) return 23; /*error: reading outside of in buffer*/
  for(n = 0; n < LEN; n++) out->data[(*pos)++] = in[p++];

  (*bp) = p * 8;

  return error;
}

static unsigned lodepng_inflatev(ucvector* out,
                                 const unsigned char* in, size_t insize,
                                 const LodePNGDecompressSettings* settings)
{
  /*bit pointer in the "in" data, current byte is bp >> 3, current bit is bp & 0x7 (from lsb to msb of the byte)*/
  size_t bp = 0;
  unsigned BFINAL = 0;
  size_t pos = 0; /*byte position in the out buffer*/

  unsigned error = 0;

  (void)settings;

  while(!BFINAL)
  {
    unsigned BTYPE;
    if(bp + 2 >= insize * 8) return 52; /*error, bit pointer will jump past memory*/
    BFINAL = readBitFromStream(&bp, in);
    BTYPE = 1 * readBitFromStream(&bp, in);
    BTYPE += 2 * readBitFromStream(&bp, in);

    if(BTYPE == 3) return 20; /*error: invalid BTYPE*/
    else if(BTYPE == 0) error = inflateNoCompression(out, in, &bp, &pos, insize); /*no compression*/
    else error = inflateHuffmanBlock(out, in, &bp, &pos, insize, BTYPE); /*compression, BTYPE 01 or 10*/

    if(error) return error;
  }

  /*Only now we know the true size of out, resize it to that*/
  if(!ucvector_resize(out, pos)) error = 83; /*alloc fail*/

  return error;
}

unsigned lodepng_inflate(unsigned char** out, size_t* outsize,
                         const unsigned char* in, size_t insize,
                         const LodePNGDecompressSettings* settings)
{
  unsigned error;
  ucvector v;
  ucvector_init_buffer(&v, *out, *outsize);
  error = lodepng_inflatev(&v, in, insize, settings);
  *out = v.data;
  *outsize = v.size;
  return error;
}

static unsigned inflate(unsigned char** out, size_t* outsize,
                        const unsigned char* in, size_t insize,
                        const LodePNGDecompressSettings* settings)
{
  if(settings->custom_inflate)
  {
    return settings->custom_inflate(out, outsize, in, insize, settings);
  }
  else
  {
    return lodepng_inflate(out, outsize, in, insize, settings);
  }
}

#endif /*LODEPNG_COMPILE_DECODER*/

#ifdef LODEPNG_COMPILE_ENCODER

/* ////////////////////////////////////////////////////////////////////////// */
/* / Deflator (Compressor)                                                  / */
/* ////////////////////////////////////////////////////////////////////////// */

static const size_t MAX_SUPPORTED_DEFLATE_LENGTH = 258;

/*bitlen is the size in bits of the code*/
static void addHuffmanSymbol(size_t* bp, ucvector* compressed, unsigned code, unsigned bitlen)
{
  addBitsToStreamReversed(bp, compressed, code, bitlen);
}

/*search the index in the array, that has the largest value smaller than or equal to the given value,
given array must be sorted (if no value is smaller, it returns the size of the given array)*/
static size_t searchCodeIndex(const unsigned* array, size_t array_size, size_t value)
{
  /*linear search implementation*/
  /*for(size_t i = 1; i < array_size; i++) if(array[i] > value) return i - 1;
  return array_size - 1;*/

  /*binary search implementation (not that much faster) (precondition: array_size > 0)*/
  size_t left  = 1;
  size_t right = array_size - 1;
  while(left <= right)
  {
    size_t mid = (left + right) / 2;
    if(array[mid] <= value) left = mid + 1; /*the value to find is more to the right*/
    else if(array[mid - 1] > value) right = mid - 1; /*the value to find is more to the left*/
    else return mid - 1;
  }
  return array_size - 1;
}

static void addLengthDistance(uivector* values, size_t length, size_t distance)
{
  /*values in encoded vector are those used by deflate:
  0-255: literal bytes
  256: end
  257-285: length/distance pair (length code, followed by extra length bits, distance code, extra distance bits)
  286-287: invalid*/

  unsigned length_code = (unsigned)searchCodeIndex(LENGTHBASE, 29, length);
  unsigned extra_length = (unsigned)(length - LENGTHBASE[length_code]);
  unsigned dist_code = (unsigned)searchCodeIndex(DISTANCEBASE, 30, distance);
  unsigned extra_distance = (unsigned)(distance - DISTANCEBASE[dist_code]);

  uivector_push_back(values, length_code + FIRST_LENGTH_CODE_INDEX);
  uivector_push_back(values, extra_length);
  uivector_push_back(values, dist_code);
  uivector_push_back(values, extra_distance);
}

static const unsigned HASH_NUM_VALUES = 65536;
static const unsigned HASH_NUM_CHARACTERS = 3;
static const unsigned HASH_SHIFT = 2;
/*
The HASH_NUM_CHARACTERS value is used to make encoding faster by using longer
sequences to generate a hash value from the stream bytes. Setting it to 3
gives exactly the same compression as the brute force method, since deflate's
run length encoding starts with lengths of 3. Setting it to higher values,
like 6, can make the encoding faster (not always though!), but will cause the
encoding to miss any length between 3 and this value, so that the compression
may be worse (but this can vary too depending on the image, sometimes it is
even a bit better instead).
The HASH_NUM_VALUES is the amount of unique possible hash values that
combinations of bytes can give, the higher it is the more memory is needed, but
if it's too low the advantage of hashing is gone.
*/

typedef struct Hash
{
  int* head; /*hash value to head circular pos*/
  int* val; /*circular pos to hash value*/
  /*circular pos to prev circular pos*/
  unsigned short* chain;
  unsigned short* zeros;
} Hash;

static unsigned hash_init(Hash* hash, unsigned windowsize)
{
  unsigned i;
  hash->head = (int*)mymalloc(sizeof(int) * HASH_NUM_VALUES);
  hash->val = (int*)mymalloc(sizeof(int) * windowsize);
  hash->chain = (unsigned short*)mymalloc(sizeof(unsigned short) * windowsize);
  hash->zeros = (unsigned short*)mymalloc(sizeof(unsigned short) * windowsize);

  if(!hash->head || !hash->val || !hash->chain || !hash->zeros) return 83; /*alloc fail*/

  /*initialize hash table*/
  for(i = 0; i < HASH_NUM_VALUES; i++) hash->head[i] = -1;
  for(i = 0; i < windowsize; i++) hash->val[i] = -1;
  for(i = 0; i < windowsize; i++) hash->chain[i] = i; /*same value as index indicates uninitialized*/

  return 0;
}

static void hash_cleanup(Hash* hash)
{
  myfree(hash->head);
  myfree(hash->val);
  myfree(hash->chain);
  myfree(hash->zeros);
}

static unsigned getHash(const unsigned char* data, size_t size, size_t pos)
{
  unsigned result = 0;
  size_t amount, i;
  if(pos >= size) return 0;
  amount = HASH_NUM_CHARACTERS;
  if(pos + amount >= size) amount = size - pos;
  for(i = 0; i < amount; i++) result ^= (data[pos + i] << (i * HASH_SHIFT));
  return result % HASH_NUM_VALUES;
}

static unsigned countZeros(const unsigned char* data, size_t size, size_t pos)
{
  const unsigned char* start = data + pos;
  const unsigned char* end = start + MAX_SUPPORTED_DEFLATE_LENGTH;
  if(end > data + size) end = data + size;
  data = start;
  while (data != end && *data == 0) data++;
  /*subtracting two addresses returned as 32-bit number (max value is MAX_SUPPORTED_DEFLATE_LENGTH)*/
  return (unsigned)(data - start);
}

static void updateHashChain(Hash* hash, size_t pos, int hashval, unsigned windowsize)
{
  unsigned wpos = pos % windowsize;
  hash->val[wpos] = hashval;
  if(hash->head[hashval] != -1) hash->chain[wpos] = hash->head[hashval];
  hash->head[hashval] = wpos;
}

/*
LZ77-encode the data. Return value is error code. The input are raw bytes, the output
is in the form of unsigned integers with codes representing for example literal bytes, or
length/distance pairs.
It uses a hash table technique to let it encode faster. When doing LZ77 encoding, a
sliding window (of windowsize) is used, and all past bytes in that window can be used as
the "dictionary". A brute force search through all possible distances would be slow, and
this hash technique is one out of several ways to speed this up.
*/
static unsigned encodeLZ77(uivector* out, Hash* hash,
                           const unsigned char* in, size_t inpos, size_t insize, unsigned windowsize,
                           unsigned minmatch, unsigned nicematch, unsigned lazymatching)
{
  unsigned short numzeros = 0;
  int usezeros = windowsize >= 8192; /*for small window size, the 'max chain length' optimization does a better job*/
  unsigned pos, i, error = 0;
  /*for large window lengths, assume the user wants no compression loss. Otherwise, max hash chain length speedup.*/
  unsigned maxchainlength = windowsize >= 8192 ? windowsize : windowsize / 8;
  unsigned maxlazymatch = windowsize >= 8192 ? MAX_SUPPORTED_DEFLATE_LENGTH : 64;

  if(!error)
  {
    unsigned offset; /*the offset represents the distance in LZ77 terminology*/
    unsigned length;
    unsigned lazy = 0;
    unsigned lazylength = 0, lazyoffset = 0;
    unsigned hashval;
    unsigned current_offset, current_length;
    const unsigned char *lastptr, *foreptr, *backptr;
    unsigned short hashpos, prevpos;

    for(pos = inpos; pos < insize; pos++)
    {
      size_t wpos = pos % windowsize; /*position for in 'circular' hash buffers*/

      hashval = getHash(in, insize, pos);
      updateHashChain(hash, pos, hashval, windowsize);

      if(usezeros && hashval == 0)
      {
        numzeros = countZeros(in, insize, pos);
        hash->zeros[wpos] = numzeros;
      }

      /*the length and offset found for the current position*/
      length = 0;
      offset = 0;

      prevpos = hash->head[hashval];
      hashpos = hash->chain[prevpos];

      lastptr = &in[insize < pos + MAX_SUPPORTED_DEFLATE_LENGTH ? insize : pos + MAX_SUPPORTED_DEFLATE_LENGTH];

      /*search for the longest string*/
      if(hash->val[wpos] == (int)hashval)
      {
        unsigned chainlength = 0;
        for(;;)
        {
          /*stop when went completely around the circular buffer*/
          if(prevpos < wpos && hashpos > prevpos && hashpos <= wpos) break;
          if(prevpos > wpos && (hashpos <= wpos || hashpos > prevpos)) break;
          if(chainlength++ >= maxchainlength) break;

          current_offset = hashpos <= wpos ? wpos - hashpos : wpos - hashpos + windowsize;
          if(current_offset > 0)
          {
            /*test the next characters*/
            foreptr = &in[pos];
            backptr = &in[pos - current_offset];

            /*common case in PNGs is lots of zeros. Quickly skip over them as a speedup*/
            if(usezeros && hashval == 0 && hash->val[hashpos] == 0 /*hashval[hashpos] may be out of date*/)
            {
              unsigned short skip = hash->zeros[hashpos];
              if(skip > numzeros) skip = numzeros;
              backptr += skip;
              foreptr += skip;
            }

            /* multiple checks at once per array bounds check */
            while(foreptr != lastptr && *backptr == *foreptr) /*maximum supported length by deflate is max length*/
            {
              ++backptr;
              ++foreptr;
            }
            current_length = (unsigned)(foreptr - &in[pos]);

            if(current_length > length)
            {
              length = current_length; /*the longest length*/
              offset = current_offset; /*the offset that is related to this longest length*/
              /*jump out once a length of max length is found (speed gain)*/
              if(current_length >= nicematch || current_length == MAX_SUPPORTED_DEFLATE_LENGTH) break;
            }
          }

          if(hashpos == hash->chain[hashpos]) break;

          prevpos = hashpos;
          hashpos = hash->chain[hashpos];
        }
      }

      if(lazymatching)
      {
        if(!lazy && length >= 3 && length <= maxlazymatch && length < MAX_SUPPORTED_DEFLATE_LENGTH)
        {
          lazy = 1;
          lazylength = length;
          lazyoffset = offset;
          continue; /*try the next byte*/
        }
        if(lazy)
        {
          lazy = 0;
          if(pos == 0) ERROR_BREAK(81);
          if(length > lazylength + 1)
          {
            /*push the previous character as literal*/
            if(!uivector_push_back(out, in[pos - 1])) ERROR_BREAK(83 /*alloc fail*/);
          }
          else
          {
            length = lazylength;
            offset = lazyoffset;
            hash->head[hashval] = -1; /*the same hashchain update will be done, this ensures no wrong alteration*/
            pos--;
          }
        }
      }
      if(length >= 3 && offset > windowsize) ERROR_BREAK(86 /*too big (or overflown negative) offset*/);

      /**encode it as length/distance pair or literal value**/
      if(length < 3) /*only lengths of 3 or higher are supported as length/distance pair*/
      {
        if(!uivector_push_back(out, in[pos])) ERROR_BREAK(83 /*alloc fail*/);
      }
      else if(length < minmatch || (length == 3 && offset > 4096))
      {
        /*compensate for the fact that longer offsets have more extra bits, a
        length of only 3 may be not worth it then*/
        if(!uivector_push_back(out, in[pos])) ERROR_BREAK(83 /*alloc fail*/);
      }
      else
      {
        addLengthDistance(out, length, offset);
        for(i = 1; i < length; i++)
        {
          pos++;
          hashval = getHash(in, insize, pos);
          updateHashChain(hash, pos, hashval, windowsize);
          if(usezeros && hashval == 0)
          {
            hash->zeros[pos % windowsize] = countZeros(in, insize, pos);
          }
        }
      }

    } /*end of the loop through each character of input*/
  } /*end of "if(!error)"*/

  return error;
}

/* /////////////////////////////////////////////////////////////////////////// */

static unsigned deflateNoCompression(ucvector* out, const unsigned char* data, size_t datasize)
{
  /*non compressed deflate block data: 1 bit BFINAL,2 bits BTYPE,(5 bits): it jumps to start of next byte,
  2 bytes LEN, 2 bytes NLEN, LEN bytes literal DATA*/

  size_t i, j, numdeflateblocks = (datasize + 65534) / 65535;
  unsigned datapos = 0;
  for(i = 0; i < numdeflateblocks; i++)
  {
    unsigned BFINAL, BTYPE, LEN, NLEN;
    unsigned char firstbyte;

    BFINAL = (i == numdeflateblocks - 1);
    BTYPE = 0;

    firstbyte = (unsigned char)(BFINAL + ((BTYPE & 1) << 1) + ((BTYPE & 2) << 1));
    ucvector_push_back(out, firstbyte);

    LEN = 65535;
    if(datasize - datapos < 65535) LEN = (unsigned)datasize - datapos;
    NLEN = 65535 - LEN;

    ucvector_push_back(out, (unsigned char)(LEN % 256));
    ucvector_push_back(out, (unsigned char)(LEN / 256));
    ucvector_push_back(out, (unsigned char)(NLEN % 256));
    ucvector_push_back(out, (unsigned char)(NLEN / 256));

    /*Decompressed data*/
    for(j = 0; j < 65535 && datapos < datasize; j++)
    {
      ucvector_push_back(out, data[datapos++]);
    }
  }

  return 0;
}

/*
write the lz77-encoded data, which has lit, len and dist codes, to compressed stream using huffman trees.
tree_ll: the tree for lit and len codes.
tree_d: the tree for distance codes.
*/
static void writeLZ77data(size_t* bp, ucvector* out, const uivector* lz77_encoded,
                          const HuffmanTree* tree_ll, const HuffmanTree* tree_d)
{
  size_t i = 0;
  for(i = 0; i < lz77_encoded->size; i++)
  {
    unsigned val = lz77_encoded->data[i];
    addHuffmanSymbol(bp, out, HuffmanTree_getCode(tree_ll, val), HuffmanTree_getLength(tree_ll, val));
    if(val > 256) /*for a length code, 3 more things have to be added*/
    {
      unsigned length_index = val - FIRST_LENGTH_CODE_INDEX;
      unsigned n_length_extra_bits = LENGTHEXTRA[length_index];
      unsigned length_extra_bits = lz77_encoded->data[++i];

      unsigned distance_code = lz77_encoded->data[++i];

      unsigned distance_index = distance_code;
      unsigned n_distance_extra_bits = DISTANCEEXTRA[distance_index];
      unsigned distance_extra_bits = lz77_encoded->data[++i];

      addBitsToStream(bp, out, length_extra_bits, n_length_extra_bits);
      addHuffmanSymbol(bp, out, HuffmanTree_getCode(tree_d, distance_code),
                       HuffmanTree_getLength(tree_d, distance_code));
      addBitsToStream(bp, out, distance_extra_bits, n_distance_extra_bits);
    }
  }
}

/*Deflate for a block of type "dynamic", that is, with freely, optimally, created huffman trees*/
static unsigned deflateDynamic(ucvector* out, size_t* bp, Hash* hash,
                               const unsigned char* data, size_t datapos, size_t dataend,
                               const LodePNGCompressSettings* settings, int final)
{
  unsigned error = 0;

  /*
  A block is compressed as follows: The PNG data is lz77 encoded, resulting in
  literal bytes and length/distance pairs. This is then huffman compressed with
  two huffman trees. One huffman tree is used for the lit and len values ("ll"),
  another huffman tree is used for the dist values ("d"). These two trees are
  stored using their code lengths, and to compress even more these code lengths
  are also run-length encoded and huffman compressed. This gives a huffman tree
  of code lengths "cl". The code lenghts used to describe this third tree are
  the code length code lengths ("clcl").
  */

  /*The lz77 encoded data, represented with integers since there will also be length and distance codes in it*/
  uivector lz77_encoded;
  HuffmanTree tree_ll; /*tree for lit,len values*/
  HuffmanTree tree_d; /*tree for distance codes*/
  HuffmanTree tree_cl; /*tree for encoding the code lengths representing tree_ll and tree_d*/
  uivector frequencies_ll; /*frequency of lit,len codes*/
  uivector frequencies_d; /*frequency of dist codes*/
  uivector frequencies_cl; /*frequency of code length codes*/
  uivector bitlen_lld; /*lit,len,dist code lenghts (int bits), literally (without repeat codes).*/
  uivector bitlen_lld_e; /*bitlen_lld encoded with repeat codes (this is a rudemtary run length compression)*/
  /*bitlen_cl is the code length code lengths ("clcl"). The bit lengths of codes to represent tree_cl
  (these are written as is in the file, it would be crazy to compress these using yet another huffman
  tree that needs to be represented by yet another set of code lengths)*/
  uivector bitlen_cl;
  size_t datasize = dataend - datapos;

  /*
  Due to the huffman compression of huffman tree representations ("two levels"), there are some anologies:
  bitlen_lld is to tree_cl what data is to tree_ll and tree_d.
  bitlen_lld_e is to bitlen_lld what lz77_encoded is to data.
  bitlen_cl is to bitlen_lld_e what bitlen_lld is to lz77_encoded.
  */

  unsigned BFINAL = final;
  size_t numcodes_ll, numcodes_d, i;
  unsigned HLIT, HDIST, HCLEN;

  uivector_init(&lz77_encoded);
  HuffmanTree_init(&tree_ll);
  HuffmanTree_init(&tree_d);
  HuffmanTree_init(&tree_cl);
  uivector_init(&frequencies_ll);
  uivector_init(&frequencies_d);
  uivector_init(&frequencies_cl);
  uivector_init(&bitlen_lld);
  uivector_init(&bitlen_lld_e);
  uivector_init(&bitlen_cl);

  /*This while loop never loops due to a break at the end, it is here to
  allow breaking out of it to the cleanup phase on error conditions.*/
  while(!error)
  {
    if(settings->use_lz77)
    {
      error = encodeLZ77(&lz77_encoded, hash, data, datapos, dataend, settings->windowsize,
                         settings->minmatch, settings->nicematch, settings->lazymatching);
      if(error) break;
    }
    else
    {
      if(!uivector_resize(&lz77_encoded, datasize)) ERROR_BREAK(83 /*alloc fail*/);
      for(i = datapos; i < dataend; i++) lz77_encoded.data[i] = data[i]; /*no LZ77, but still will be Huffman compressed*/
    }

    if(!uivector_resizev(&frequencies_ll, 286, 0)) ERROR_BREAK(83 /*alloc fail*/);
    if(!uivector_resizev(&frequencies_d, 30, 0)) ERROR_BREAK(83 /*alloc fail*/);

    /*Count the frequencies of lit, len and dist codes*/
    for(i = 0; i < lz77_encoded.size; i++)
    {
      unsigned symbol = lz77_encoded.data[i];
      frequencies_ll.data[symbol]++;
      if(symbol > 256)
      {
        unsigned dist = lz77_encoded.data[i + 2];
        frequencies_d.data[dist]++;
        i += 3;
      }
    }
    frequencies_ll.data[256] = 1; /*there will be exactly 1 end code, at the end of the block*/

    /*Make both huffman trees, one for the lit and len codes, one for the dist codes*/
    error = HuffmanTree_makeFromFrequencies(&tree_ll, frequencies_ll.data, 257, frequencies_ll.size, 15);
    if(error) break;
    /*2, not 1, is chosen for mincodes: some buggy PNG decoders require at least 2 symbols in the dist tree*/
    error = HuffmanTree_makeFromFrequencies(&tree_d, frequencies_d.data, 2, frequencies_d.size, 15);
    if(error) break;

    numcodes_ll = tree_ll.numcodes; if(numcodes_ll > 286) numcodes_ll = 286;
    numcodes_d = tree_d.numcodes; if(numcodes_d > 30) numcodes_d = 30;
    /*store the code lengths of both generated trees in bitlen_lld*/
    for(i = 0; i < numcodes_ll; i++) uivector_push_back(&bitlen_lld, HuffmanTree_getLength(&tree_ll, (unsigned)i));
    for(i = 0; i < numcodes_d; i++) uivector_push_back(&bitlen_lld, HuffmanTree_getLength(&tree_d, (unsigned)i));

    /*run-length compress bitlen_ldd into bitlen_lld_e by using repeat codes 16 (copy length 3-6 times),
    17 (3-10 zeroes), 18 (11-138 zeroes)*/
    for(i = 0; i < (unsigned)bitlen_lld.size; i++)
    {
      unsigned j = 0; /*amount of repititions*/
      while(i + j + 1 < (unsigned)bitlen_lld.size && bitlen_lld.data[i + j + 1] == bitlen_lld.data[i]) j++;

      if(bitlen_lld.data[i] == 0 && j >= 2) /*repeat code for zeroes*/
      {
        j++; /*include the first zero*/
        if(j <= 10) /*repeat code 17 supports max 10 zeroes*/
        {
          uivector_push_back(&bitlen_lld_e, 17);
          uivector_push_back(&bitlen_lld_e, j - 3);
        }
        else /*repeat code 18 supports max 138 zeroes*/
        {
          if(j > 138) j = 138;
          uivector_push_back(&bitlen_lld_e, 18);
          uivector_push_back(&bitlen_lld_e, j - 11);
        }
        i += (j - 1);
      }
      else if(j >= 3) /*repeat code for value other than zero*/
      {
        size_t k;
        unsigned num = j / 6, rest = j % 6;
        uivector_push_back(&bitlen_lld_e, bitlen_lld.data[i]);
        for(k = 0; k < num; k++)
        {
          uivector_push_back(&bitlen_lld_e, 16);
          uivector_push_back(&bitlen_lld_e, 6 - 3);
        }
        if(rest >= 3)
        {
          uivector_push_back(&bitlen_lld_e, 16);
          uivector_push_back(&bitlen_lld_e, rest - 3);
        }
        else j -= rest;
        i += j;
      }
      else /*too short to benefit from repeat code*/
      {
        uivector_push_back(&bitlen_lld_e, bitlen_lld.data[i]);
      }
    }

    /*generate tree_cl, the huffmantree of huffmantrees*/

    if(!uivector_resizev(&frequencies_cl, NUM_CODE_LENGTH_CODES, 0)) ERROR_BREAK(83 /*alloc fail*/);
    for(i = 0; i < bitlen_lld_e.size; i++)
    {
      frequencies_cl.data[bitlen_lld_e.data[i]]++;
      /*after a repeat code come the bits that specify the number of repetitions,
      those don't need to be in the frequencies_cl calculation*/
      if(bitlen_lld_e.data[i] >= 16) i++;
    }

    error = HuffmanTree_makeFromFrequencies(&tree_cl, frequencies_cl.data,
                                            frequencies_cl.size, frequencies_cl.size, 7);
    if(error) break;

    if(!uivector_resize(&bitlen_cl, tree_cl.numcodes)) ERROR_BREAK(83 /*alloc fail*/);
    for(i = 0; i < tree_cl.numcodes; i++)
    {
      /*lenghts of code length tree is in the order as specified by deflate*/
      bitlen_cl.data[i] = HuffmanTree_getLength(&tree_cl, CLCL_ORDER[i]);
    }
    while(bitlen_cl.data[bitlen_cl.size - 1] == 0 && bitlen_cl.size > 4)
    {
      /*remove zeros at the end, but minimum size must be 4*/
      if(!uivector_resize(&bitlen_cl, bitlen_cl.size - 1)) ERROR_BREAK(83 /*alloc fail*/);
    }
    if(error) break;

    /*
    Write everything into the output

    After the BFINAL and BTYPE, the dynamic block consists out of the following:
    - 5 bits HLIT, 5 bits HDIST, 4 bits HCLEN
    - (HCLEN+4)*3 bits code lengths of code length alphabet
    - HLIT + 257 code lenghts of lit/length alphabet (encoded using the code length
      alphabet, + possible repetition codes 16, 17, 18)
    - HDIST + 1 code lengths of distance alphabet (encoded using the code length
      alphabet, + possible repetition codes 16, 17, 18)
    - compressed data
    - 256 (end code)
    */

    /*Write block type*/
    addBitToStream(bp, out, BFINAL);
    addBitToStream(bp, out, 0); /*first bit of BTYPE "dynamic"*/
    addBitToStream(bp, out, 1); /*second bit of BTYPE "dynamic"*/

    /*write the HLIT, HDIST and HCLEN values*/
    HLIT = (unsigned)(numcodes_ll - 257);
    HDIST = (unsigned)(numcodes_d - 1);
    HCLEN = (unsigned)bitlen_cl.size - 4;
    /*trim zeroes for HCLEN. HLIT and HDIST were already trimmed at tree creation*/
    while(!bitlen_cl.data[HCLEN + 4 - 1] && HCLEN > 0) HCLEN--;
    addBitsToStream(bp, out, HLIT, 5);
    addBitsToStream(bp, out, HDIST, 5);
    addBitsToStream(bp, out, HCLEN, 4);

    /*write the code lenghts of the code length alphabet*/
    for(i = 0; i < HCLEN + 4; i++) addBitsToStream(bp, out, bitlen_cl.data[i], 3);

    /*write the lenghts of the lit/len AND the dist alphabet*/
    for(i = 0; i < bitlen_lld_e.size; i++)
    {
      addHuffmanSymbol(bp, out, HuffmanTree_getCode(&tree_cl, bitlen_lld_e.data[i]),
                       HuffmanTree_getLength(&tree_cl, bitlen_lld_e.data[i]));
      /*extra bits of repeat codes*/
      if(bitlen_lld_e.data[i] == 16) addBitsToStream(bp, out, bitlen_lld_e.data[++i], 2);
      else if(bitlen_lld_e.data[i] == 17) addBitsToStream(bp, out, bitlen_lld_e.data[++i], 3);
      else if(bitlen_lld_e.data[i] == 18) addBitsToStream(bp, out, bitlen_lld_e.data[++i], 7);
    }

    /*write the compressed data symbols*/
    writeLZ77data(bp, out, &lz77_encoded, &tree_ll, &tree_d);
    /*error: the length of the end code 256 must be larger than 0*/
    if(HuffmanTree_getLength(&tree_ll, 256) == 0) ERROR_BREAK(64);

    /*write the end code*/
    addHuffmanSymbol(bp, out, HuffmanTree_getCode(&tree_ll, 256), HuffmanTree_getLength(&tree_ll, 256));

    break; /*end of error-while*/
  }

  /*cleanup*/
  uivector_cleanup(&lz77_encoded);
  HuffmanTree_cleanup(&tree_ll);
  HuffmanTree_cleanup(&tree_d);
  HuffmanTree_cleanup(&tree_cl);
  uivector_cleanup(&frequencies_ll);
  uivector_cleanup(&frequencies_d);
  uivector_cleanup(&frequencies_cl);
  uivector_cleanup(&bitlen_lld_e);
  uivector_cleanup(&bitlen_lld);
  uivector_cleanup(&bitlen_cl);

  return error;
}

static unsigned deflateFixed(ucvector* out, size_t* bp, Hash* hash,
                             const unsigned char* data,
                             size_t datapos, size_t dataend,
                             const LodePNGCompressSettings* settings, int final)
{
  HuffmanTree tree_ll; /*tree for literal values and length codes*/
  HuffmanTree tree_d; /*tree for distance codes*/

  unsigned BFINAL = final;
  unsigned error = 0;
  size_t i;

  HuffmanTree_init(&tree_ll);
  HuffmanTree_init(&tree_d);

  generateFixedLitLenTree(&tree_ll);
  generateFixedDistanceTree(&tree_d);

  addBitToStream(bp, out, BFINAL);
  addBitToStream(bp, out, 1); /*first bit of BTYPE*/
  addBitToStream(bp, out, 0); /*second bit of BTYPE*/

  if(settings->use_lz77) /*LZ77 encoded*/
  {
    uivector lz77_encoded;
    uivector_init(&lz77_encoded);
    error = encodeLZ77(&lz77_encoded, hash, data, datapos, dataend, settings->windowsize,
                       settings->minmatch, settings->nicematch, settings->lazymatching);
    if(!error) writeLZ77data(bp, out, &lz77_encoded, &tree_ll, &tree_d);
    uivector_cleanup(&lz77_encoded);
  }
  else /*no LZ77, but still will be Huffman compressed*/
  {
    for(i = datapos; i < dataend; i++)
    {
      addHuffmanSymbol(bp, out, HuffmanTree_getCode(&tree_ll, data[i]), HuffmanTree_getLength(&tree_ll, data[i]));
    }
  }
  /*add END code*/
  if(!error) addHuffmanSymbol(bp, out, HuffmanTree_getCode(&tree_ll, 256), HuffmanTree_getLength(&tree_ll, 256));

  /*cleanup*/
  HuffmanTree_cleanup(&tree_ll);
  HuffmanTree_cleanup(&tree_d);

  return error;
}

static unsigned lodepng_deflatev(ucvector* out, const unsigned char* in, size_t insize,
                                 const LodePNGCompressSettings* settings)
{
  unsigned error = 0;
  size_t i, blocksize, numdeflateblocks;
  size_t bp = 0; /*the bit pointer*/
  Hash hash;

  if(settings->btype > 2) return 61;
  else if(settings->btype == 0) return deflateNoCompression(out, in, insize);
  else if(settings->btype == 1) blocksize = insize;
  else /*if(settings->btype == 2)*/
  {
    blocksize = insize / 8 + 8;
    if(blocksize < 65535) blocksize = 65535;
  }

  numdeflateblocks = (insize + blocksize - 1) / blocksize;
  if(numdeflateblocks == 0) numdeflateblocks = 1;

  error = hash_init(&hash, settings->windowsize);
  if(error) return error;

  for(i = 0; i < numdeflateblocks && !error; i++)
  {
    int final = i == numdeflateblocks - 1;
    size_t start = i * blocksize;
    size_t end = start + blocksize;
    if(end > insize) end = insize;

    if(settings->btype == 1) error = deflateFixed(out, &bp, &hash, in, start, end, settings, final);
    else if(settings->btype == 2) error = deflateDynamic(out, &bp, &hash, in, start, end, settings, final);
  }

  hash_cleanup(&hash);

  return error;
}

unsigned lodepng_deflate(unsigned char** out, size_t* outsize,
                         const unsigned char* in, size_t insize,
                         const LodePNGCompressSettings* settings)
{
  unsigned error;
  ucvector v;
  ucvector_init_buffer(&v, *out, *outsize);
  error = lodepng_deflatev(&v, in, insize, settings);
  *out = v.data;
  *outsize = v.size;
  return error;
}

static unsigned deflate(unsigned char** out, size_t* outsize,
                        const unsigned char* in, size_t insize,
                        const LodePNGCompressSettings* settings)
{
  if(settings->custom_deflate)
  {
    return settings->custom_deflate(out, outsize, in, insize, settings);
  }
  else
  {
    return lodepng_deflate(out, outsize, in, insize, settings);
  }
}

#endif /*LODEPNG_COMPILE_DECODER*/

/* ////////////////////////////////////////////////////////////////////////// */
/* / Adler32                                                                  */
/* ////////////////////////////////////////////////////////////////////////// */

static unsigned update_adler32(unsigned adler, const unsigned char* data, unsigned len)
{
   unsigned s1 = adler & 0xffff;
   unsigned s2 = (adler >> 16) & 0xffff;

  while(len > 0)
  {
    /*at least 5550 sums can be done before the sums overflow, saving a lot of module divisions*/
    unsigned amount = len > 5550 ? 5550 : len;
    len -= amount;
    while(amount > 0)
    {
      s1 += (*data++);
      s2 += s1;
      amount--;
    }
    s1 %= 65521;
    s2 %= 65521;
  }

  return (s2 << 16) | s1;
}

/*Return the adler32 of the bytes data[0..len-1]*/
static unsigned adler32(const unsigned char* data, unsigned len)
{
  return update_adler32(1L, data, len);
}

/* ////////////////////////////////////////////////////////////////////////// */
/* / Zlib                                                                   / */
/* ////////////////////////////////////////////////////////////////////////// */

#ifdef LODEPNG_COMPILE_DECODER

unsigned lodepng_zlib_decompress(unsigned char** out, size_t* outsize, const unsigned char* in,
                                 size_t insize, const LodePNGDecompressSettings* settings)
{
  unsigned error = 0;
  unsigned CM, CINFO, FDICT;

  if(insize < 2) return 53; /*error, size of zlib data too small*/
  /*read information from zlib header*/
  if((in[0] * 256 + in[1]) % 31 != 0)
  {
    /*error: 256 * in[0] + in[1] must be a multiple of 31, the FCHECK value is supposed to be made that way*/
    return 24;
  }

  CM = in[0] & 15;
  CINFO = (in[0] >> 4) & 15;
  /*FCHECK = in[1] & 31;*/ /*FCHECK is already tested above*/
  FDICT = (in[1] >> 5) & 1;
  /*FLEVEL = (in[1] >> 6) & 3;*/ /*FLEVEL is not used here*/

  if(CM != 8 || CINFO > 7)
  {
    /*error: only compression method 8: inflate with sliding window of 32k is supported by the PNG spec*/
    return 25;
  }
  if(FDICT != 0)
  {
    /*error: the specification of PNG says about the zlib stream:
      "The additional flags shall not specify a preset dictionary."*/
    return 26;
  }

  error = inflate(out, outsize, in + 2, insize - 2, settings);
  if(error) return error;

  if(!settings->ignore_adler32)
  {
    unsigned ADLER32 = lodepng_read32bitInt(&in[insize - 4]);
    unsigned checksum = adler32(*out, (unsigned)(*outsize));
    if(checksum != ADLER32) return 58; /*error, adler checksum not correct, data must be corrupted*/
  }

  return 0; /*no error*/
}

static unsigned zlib_decompress(unsigned char** out, size_t* outsize, const unsigned char* in,
                                size_t insize, const LodePNGDecompressSettings* settings)
{
  if(settings->custom_zlib)
    return settings->custom_zlib(out, outsize, in, insize, settings);
  else
    return lodepng_zlib_decompress(out, outsize, in, insize, settings);
}

#endif /*LODEPNG_COMPILE_DECODER*/

#ifdef LODEPNG_COMPILE_ENCODER

unsigned lodepng_zlib_compress(unsigned char** out, size_t* outsize, const unsigned char* in,
                               size_t insize, const LodePNGCompressSettings* settings)
{
  /*initially, *out must be NULL and outsize 0, if you just give some random *out
  that's pointing to a non allocated buffer, this'll crash*/
  ucvector outv;
  size_t i;
  unsigned error;
  unsigned char* deflatedata = 0;
  size_t deflatesize = 0;

  unsigned ADLER32;
  /*zlib data: 1 byte CMF (CM+CINFO), 1 byte FLG, deflate data, 4 byte ADLER32 checksum of the Decompressed data*/
  unsigned CMF = 120; /*0b01111000: CM 8, CINFO 7. With CINFO 7, any window size up to 32768 can be used.*/
  unsigned FLEVEL = 0;
  unsigned FDICT = 0;
  unsigned CMFFLG = 256 * CMF + FDICT * 32 + FLEVEL * 64;
  unsigned FCHECK = 31 - CMFFLG % 31;
  CMFFLG += FCHECK;

  /*ucvector-controlled version of the output buffer, for dynamic array*/
  ucvector_init_buffer(&outv, *out, *outsize);

  ucvector_push_back(&outv, (unsigned char)(CMFFLG / 256));
  ucvector_push_back(&outv, (unsigned char)(CMFFLG % 256));

  error = deflate(&deflatedata, &deflatesize, in, insize, settings);

  if(!error)
  {
    ADLER32 = adler32(in, (unsigned)insize);
    for(i = 0; i < deflatesize; i++) ucvector_push_back(&outv, deflatedata[i]);
    free(deflatedata);
    lodepng_add32bitInt(&outv, ADLER32);
  }

  *out = outv.data;
  *outsize = outv.size;

  return error;
}

/* compress using the default or custom zlib function */
static unsigned zlib_compress(unsigned char** out, size_t* outsize, const unsigned char* in,
                              size_t insize, const LodePNGCompressSettings* settings)
{
  if(settings->custom_zlib)
  {
    return settings->custom_zlib(out, outsize, in, insize, settings);
  }
  else
  {
    return lodepng_zlib_compress(out, outsize, in, insize, settings);
  }
}

#endif /*LODEPNG_COMPILE_ENCODER*/

#else /*no LODEPNG_COMPILE_ZLIB*/

#ifdef LODEPNG_COMPILE_DECODER
static unsigned zlib_decompress(unsigned char** out, size_t* outsize, const unsigned char* in,
                                size_t insize, const LodePNGDecompressSettings* settings)
{
  if (!settings->custom_zlib) return 87; /*no custom zlib function provided */
  return settings->custom_zlib(out, outsize, in, insize, settings);
}
#endif /*LODEPNG_COMPILE_DECODER*/
#ifdef LODEPNG_COMPILE_ENCODER
static unsigned zlib_compress(unsigned char** out, size_t* outsize, const unsigned char* in,
                              size_t insize, const LodePNGCompressSettings* settings)
{
  if (!settings->custom_zlib) return 87; /*no custom zlib function provided */
  return settings->custom_zlib(out, outsize, in, insize, settings);
}
#endif /*LODEPNG_COMPILE_ENCODER*/

#endif /*LODEPNG_COMPILE_ZLIB*/

/* ////////////////////////////////////////////////////////////////////////// */

#ifdef LODEPNG_COMPILE_ENCODER

/*this is a good tradeoff between speed and compression ratio*/
#define DEFAULT_WINDOWSIZE 2048

void lodepng_compress_settings_init(LodePNGCompressSettings* settings)
{
  /*compress with dynamic huffman tree (not in the mathematical sense, just not the predefined one)*/
  settings->btype = 2;
  settings->use_lz77 = 1;
  settings->windowsize = DEFAULT_WINDOWSIZE;
  settings->minmatch = 3;
  settings->nicematch = 128;
  settings->lazymatching = 1;

  settings->custom_zlib = 0;
  settings->custom_deflate = 0;
  settings->custom_context = 0;
}

const LodePNGCompressSettings lodepng_default_compress_settings = {2, 1, DEFAULT_WINDOWSIZE, 3, 128, 1, 0, 0, 0};


#endif /*LODEPNG_COMPILE_ENCODER*/

#ifdef LODEPNG_COMPILE_DECODER

void lodepng_decompress_settings_init(LodePNGDecompressSettings* settings)
{
  settings->ignore_adler32 = 0;

  settings->custom_zlib = 0;
  settings->custom_inflate = 0;
  settings->custom_context = 0;
}

const LodePNGDecompressSettings lodepng_default_decompress_settings = {0, 0, 0, 0};

#endif /*LODEPNG_COMPILE_DECODER*/

/* ////////////////////////////////////////////////////////////////////////// */
/* ////////////////////////////////////////////////////////////////////////// */
/* // End of Zlib related code. Begin of PNG related code.                 // */
/* ////////////////////////////////////////////////////////////////////////// */
/* ////////////////////////////////////////////////////////////////////////// */

#ifdef LODEPNG_COMPILE_PNG

/* ////////////////////////////////////////////////////////////////////////// */
/* / CRC32                                                                  / */
/* ////////////////////////////////////////////////////////////////////////// */

static unsigned Crc32_crc_table_computed = 0;
static unsigned Crc32_crc_table[256];

/*Make the table for a fast CRC.*/
static void Crc32_make_crc_table(void)
{
  unsigned c, k, n;
  for(n = 0; n < 256; n++)
  {
    c = n;
    for(k = 0; k < 8; k++)
    {
      if(c & 1) c = 0xedb88320L ^ (c >> 1);
      else c = c >> 1;
    }
    Crc32_crc_table[n] = c;
  }
  Crc32_crc_table_computed = 1;
}

/*Update a running CRC with the bytes buf[0..len-1]--the CRC should be
initialized to all 1's, and the transmitted value is the 1's complement of the
final running CRC (see the crc() routine below).*/
static unsigned Crc32_update_crc(const unsigned char* buf, unsigned crc, size_t len)
{
  unsigned c = crc;
  size_t n;

  if(!Crc32_crc_table_computed) Crc32_make_crc_table();
  for(n = 0; n < len; n++)
  {
    c = Crc32_crc_table[(c ^ buf[n]) & 0xff] ^ (c >> 8);
  }
  return c;
}

/*Return the CRC of the bytes buf[0..len-1].*/
unsigned lodepng_crc32(const unsigned char* buf, size_t len)
{
  return Crc32_update_crc(buf, 0xffffffffL, len) ^ 0xffffffffL;
}

/* ////////////////////////////////////////////////////////////////////////// */
/* / Reading and writing single bits and bytes from/to stream for LodePNG   / */
/* ////////////////////////////////////////////////////////////////////////// */

static unsigned char readBitFromReversedStream(size_t* bitpointer, const unsigned char* bitstream)
{
  unsigned char result = (unsigned char)((bitstream[(*bitpointer) >> 3] >> (7 - ((*bitpointer) & 0x7))) & 1);
  (*bitpointer)++;
  return result;
}

static unsigned readBitsFromReversedStream(size_t* bitpointer, const unsigned char* bitstream, size_t nbits)
{
  unsigned result = 0;
  size_t i;
  for(i = nbits - 1; i < nbits; i--)
  {
    result += (unsigned)readBitFromReversedStream(bitpointer, bitstream) << i;
  }
  return result;
}

#ifdef LODEPNG_COMPILE_DECODER
static void setBitOfReversedStream0(size_t* bitpointer, unsigned char* bitstream, unsigned char bit)
{
  /*the current bit in bitstream must be 0 for this to work*/
  if(bit)
  {
    /*earlier bit of huffman code is in a lesser significant bit of an earlier byte*/
    bitstream[(*bitpointer) >> 3] |= (bit << (7 - ((*bitpointer) & 0x7)));
  }
  (*bitpointer)++;
}
#endif /*LODEPNG_COMPILE_DECODER*/

static void setBitOfReversedStream(size_t* bitpointer, unsigned char* bitstream, unsigned char bit)
{
  /*the current bit in bitstream may be 0 or 1 for this to work*/
  if(bit == 0) bitstream[(*bitpointer) >> 3] &=  (unsigned char)(~(1 << (7 - ((*bitpointer) & 0x7))));
  else         bitstream[(*bitpointer) >> 3] |=  (1 << (7 - ((*bitpointer) & 0x7)));
  (*bitpointer)++;
}

/* ////////////////////////////////////////////////////////////////////////// */
/* / PNG chunks                                                             / */
/* ////////////////////////////////////////////////////////////////////////// */

unsigned lodepng_chunk_length(const unsigned char* chunk)
{
  return lodepng_read32bitInt(&chunk[0]);
}

void lodepng_chunk_type(char type[5], const unsigned char* chunk)
{
  unsigned i;
  for(i = 0; i < 4; i++) type[i] = chunk[4 + i];
  type[4] = 0; /*null termination char*/
}

unsigned char lodepng_chunk_type_equals(const unsigned char* chunk, const char* type)
{
  if(strlen(type) != 4) return 0;
  return (chunk[4] == type[0] && chunk[5] == type[1] && chunk[6] == type[2] && chunk[7] == type[3]);
}

unsigned char lodepng_chunk_ancillary(const unsigned char* chunk)
{
  return((chunk[4] & 32) != 0);
}

unsigned char lodepng_chunk_private(const unsigned char* chunk)
{
  return((chunk[6] & 32) != 0);
}

unsigned char lodepng_chunk_safetocopy(const unsigned char* chunk)
{
  return((chunk[7] & 32) != 0);
}

unsigned char* lodepng_chunk_data(unsigned char* chunk)
{
  return &chunk[8];
}

const unsigned char* lodepng_chunk_data_const(const unsigned char* chunk)
{
  return &chunk[8];
}

unsigned lodepng_chunk_check_crc(const unsigned char* chunk)
{
  unsigned length = lodepng_chunk_length(chunk);
  unsigned CRC = lodepng_read32bitInt(&chunk[length + 8]);
  /*the CRC is taken of the data and the 4 chunk type letters, not the length*/
  unsigned checksum = lodepng_crc32(&chunk[4], length + 4);
  if(CRC != checksum) return 1;
  else return 0;
}

void lodepng_chunk_generate_crc(unsigned char* chunk)
{
  unsigned length = lodepng_chunk_length(chunk);
  unsigned CRC = lodepng_crc32(&chunk[4], length + 4);
  lodepng_set32bitInt(chunk + 8 + length, CRC);
}

unsigned char* lodepng_chunk_next(unsigned char* chunk)
{
  unsigned total_chunk_length = lodepng_chunk_length(chunk) + 12;
  return &chunk[total_chunk_length];
}

const unsigned char* lodepng_chunk_next_const(const unsigned char* chunk)
{
  unsigned total_chunk_length = lodepng_chunk_length(chunk) + 12;
  return &chunk[total_chunk_length];
}

unsigned lodepng_chunk_append(unsigned char** out, size_t* outlength, const unsigned char* chunk)
{
  unsigned i;
  unsigned total_chunk_length = lodepng_chunk_length(chunk) + 12;
  unsigned char *chunk_start, *new_buffer;
  size_t new_length = (*outlength) + total_chunk_length;
  if(new_length < total_chunk_length || new_length < (*outlength)) return 77; /*integer overflow happened*/

  new_buffer = (unsigned char*)myrealloc(*out, new_length);
  if(!new_buffer) return 83; /*alloc fail*/
  (*out) = new_buffer;
  (*outlength) = new_length;
  chunk_start = &(*out)[new_length - total_chunk_length];

  for(i = 0; i < total_chunk_length; i++) chunk_start[i] = chunk[i];

  return 0;
}

unsigned lodepng_chunk_create(unsigned char** out, size_t* outlength, unsigned length,
                              const char* type, const unsigned char* data)
{
  unsigned i;
  unsigned char *chunk, *new_buffer;
  size_t new_length = (*outlength) + length + 12;
  if(new_length < length + 12 || new_length < (*outlength)) return 77; /*integer overflow happened*/
  new_buffer = (unsigned char*)myrealloc(*out, new_length);
  if(!new_buffer) return 83; /*alloc fail*/
  (*out) = new_buffer;
  (*outlength) = new_length;
  chunk = &(*out)[(*outlength) - length - 12];

  /*1: length*/
  lodepng_set32bitInt(chunk, (unsigned)length);

  /*2: chunk name (4 letters)*/
  chunk[4] = type[0];
  chunk[5] = type[1];
  chunk[6] = type[2];
  chunk[7] = type[3];

  /*3: the data*/
  for(i = 0; i < length; i++) chunk[8 + i] = data[i];

  /*4: CRC (of the chunkname characters and the data)*/
  lodepng_chunk_generate_crc(chunk);

  return 0;
}

/* ////////////////////////////////////////////////////////////////////////// */
/* / Color types and such                                                   / */
/* ////////////////////////////////////////////////////////////////////////// */

/*return type is a LodePNG error code*/
static unsigned checkColorValidity(LodePNGColorType colortype, unsigned bd) /*bd = bitdepth*/
{
  switch(colortype)
  {
    case 0: if(!(bd == 1 || bd == 2 || bd == 4 || bd == 8 || bd == 16)) return 37; break; /*grey*/
    case 2: if(!(                                 bd == 8 || bd == 16)) return 37; break; /*RGB*/
    case 3: if(!(bd == 1 || bd == 2 || bd == 4 || bd == 8            )) return 37; break; /*palette*/
    case 4: if(!(                                 bd == 8 || bd == 16)) return 37; break; /*grey + alpha*/
    case 6: if(!(                                 bd == 8 || bd == 16)) return 37; break; /*RGBA*/
    default: return 31;
  }
  return 0; /*allowed color type / bits combination*/
}

static unsigned getNumColorChannels(LodePNGColorType colortype)
{
  switch(colortype)
  {
    case 0: return 1; /*grey*/
    case 2: return 3; /*RGB*/
    case 3: return 1; /*palette*/
    case 4: return 2; /*grey + alpha*/
    case 6: return 4; /*RGBA*/
  }
  return 0; /*unexisting color type*/
}

static unsigned lodepng_get_bpp_lct(LodePNGColorType colortype, unsigned bitdepth)
{
  /*bits per pixel is amount of channels * bits per channel*/
  return getNumColorChannels(colortype) * bitdepth;
}

/* ////////////////////////////////////////////////////////////////////////// */

void lodepng_color_mode_init(LodePNGColorMode* info)
{
  info->key_defined = 0;
  info->key_r = info->key_g = info->key_b = 0;
  info->colortype = LCT_RGBA;
  info->bitdepth = 8;
  info->palette = 0;
  info->palettesize = 0;
}

void lodepng_color_mode_cleanup(LodePNGColorMode* info)
{
  lodepng_palette_clear(info);
}

unsigned lodepng_color_mode_copy(LodePNGColorMode* dest, const LodePNGColorMode* source)
{
  size_t i;
  lodepng_color_mode_cleanup(dest);
  *dest = *source;
  if(source->palette)
  {
    dest->palette = (unsigned char*)mymalloc(source->palettesize * 4);
    if(!dest->palette && source->palettesize) return 83; /*alloc fail*/
    for(i = 0; i < source->palettesize * 4; i++) dest->palette[i] = source->palette[i];
  }
  return 0;
}

static int lodepng_color_mode_equal(const LodePNGColorMode* a, const LodePNGColorMode* b)
{
  size_t i;
  if(a->colortype != b->colortype) return 0;
  if(a->bitdepth != b->bitdepth) return 0;
  if(a->key_defined != b->key_defined) return 0;
  if(a->key_defined)
  {
    if(a->key_r != b->key_r) return 0;
    if(a->key_g != b->key_g) return 0;
    if(a->key_b != b->key_b) return 0;
  }
  if(a->palettesize != b->palettesize) return 0;
  for(i = 0; i < a->palettesize * 4; i++)
  {
    if(a->palette[i] != b->palette[i]) return 0;
  }
  return 1;
}

void lodepng_palette_clear(LodePNGColorMode* info)
{
  if(info->palette) myfree(info->palette);
  info->palettesize = 0;
}

unsigned lodepng_palette_add(LodePNGColorMode* info,
                             unsigned char r, unsigned char g, unsigned char b, unsigned char a)
{
  unsigned char* data;
  /*the same resize technique as C++ std::vectors is used, and here it's made so that for a palette with
  the max of 256 colors, it'll have the exact alloc size*/
  if(!(info->palettesize & (info->palettesize - 1))) /*if palettesize is 0 or a power of two*/
  {
    /*allocated data must be at least 4* palettesize (for 4 color bytes)*/
    size_t alloc_size = info->palettesize == 0 ? 4 : info->palettesize * 4 * 2;
    data = (unsigned char*)myrealloc(info->palette, alloc_size);
    if(!data) return 83; /*alloc fail*/
    else info->palette = data;
  }
  info->palette[4 * info->palettesize + 0] = r;
  info->palette[4 * info->palettesize + 1] = g;
  info->palette[4 * info->palettesize + 2] = b;
  info->palette[4 * info->palettesize + 3] = a;
  info->palettesize++;
  return 0;
}

unsigned lodepng_get_bpp(const LodePNGColorMode* info)
{
  /*calculate bits per pixel out of colortype and bitdepth*/
  return lodepng_get_bpp_lct(info->colortype, info->bitdepth);
}

unsigned lodepng_get_channels(const LodePNGColorMode* info)
{
  return getNumColorChannels(info->colortype);
}

unsigned lodepng_is_greyscale_type(const LodePNGColorMode* info)
{
  return info->colortype == LCT_GREY || info->colortype == LCT_GREY_ALPHA;
}

unsigned lodepng_is_alpha_type(const LodePNGColorMode* info)
{
  return (info->colortype & 4) != 0; /*4 or 6*/
}

unsigned lodepng_is_palette_type(const LodePNGColorMode* info)
{
  return info->colortype == LCT_PALETTE;
}

unsigned lodepng_has_palette_alpha(const LodePNGColorMode* info)
{
  size_t i;
  for(i = 0; i < info->palettesize; i++)
  {
    if(info->palette[i * 4 + 3] < 255) return 1;
  }
  return 0;
}

unsigned lodepng_can_have_alpha(const LodePNGColorMode* info)
{
  return info->key_defined
      || lodepng_is_alpha_type(info)
      || lodepng_has_palette_alpha(info);
}

size_t lodepng_get_raw_size(unsigned w, unsigned h, const LodePNGColorMode* color)
{
  return (w * h * lodepng_get_bpp(color) + 7) / 8;
}

size_t lodepng_get_raw_size_lct(unsigned w, unsigned h, LodePNGColorType colortype, unsigned bitdepth)
{
  return (w * h * lodepng_get_bpp_lct(colortype, bitdepth) + 7) / 8;
}

#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS

static void LodePNGUnknownChunks_init(LodePNGInfo* info)
{
  unsigned i;
  for(i = 0; i < 3; i++) info->unknown_chunks_data[i] = 0;
  for(i = 0; i < 3; i++) info->unknown_chunks_size[i] = 0;
}

static void LodePNGUnknownChunks_cleanup(LodePNGInfo* info)
{
  unsigned i;
  for(i = 0; i < 3; i++) myfree(info->unknown_chunks_data[i]);
}

static unsigned LodePNGUnknownChunks_copy(LodePNGInfo* dest, const LodePNGInfo* src)
{
  unsigned i;

  LodePNGUnknownChunks_cleanup(dest);

  for(i = 0; i < 3; i++)
  {
    size_t j;
    dest->unknown_chunks_size[i] = src->unknown_chunks_size[i];
    dest->unknown_chunks_data[i] = (unsigned char*)mymalloc(src->unknown_chunks_size[i]);
    if(!dest->unknown_chunks_data[i] && dest->unknown_chunks_size[i]) return 83; /*alloc fail*/
    for(j = 0; j < src->unknown_chunks_size[i]; j++)
    {
      dest->unknown_chunks_data[i][j] = src->unknown_chunks_data[i][j];
    }
  }

  return 0;
}

/******************************************************************************/

static void LodePNGText_init(LodePNGInfo* info)
{
  info->text_num = 0;
  info->text_keys = NULL;
  info->text_strings = NULL;
}

static void LodePNGText_cleanup(LodePNGInfo* info)
{
  size_t i;
  for(i = 0; i < info->text_num; i++)
  {
    string_cleanup(&info->text_keys[i]);
    string_cleanup(&info->text_strings[i]);
  }
  myfree(info->text_keys);
  myfree(info->text_strings);
}

static unsigned LodePNGText_copy(LodePNGInfo* dest, const LodePNGInfo* source)
{
  size_t i = 0;
  dest->text_keys = 0;
  dest->text_strings = 0;
  dest->text_num = 0;
  for(i = 0; i < source->text_num; i++)
  {
    CERROR_TRY_RETURN(lodepng_add_text(dest, source->text_keys[i], source->text_strings[i]));
  }
  return 0;
}

void lodepng_clear_text(LodePNGInfo* info)
{
  LodePNGText_cleanup(info);
}

unsigned lodepng_add_text(LodePNGInfo* info, const char* key, const char* str)
{
  char** new_keys = (char**)(myrealloc(info->text_keys, sizeof(char*) * (info->text_num + 1)));
  char** new_strings = (char**)(myrealloc(info->text_strings, sizeof(char*) * (info->text_num + 1)));
  if(!new_keys || !new_strings)
  {
    myfree(new_keys);
    myfree(new_strings);
    return 83; /*alloc fail*/
  }

  info->text_num++;
  info->text_keys = new_keys;
  info->text_strings = new_strings;

  string_init(&info->text_keys[info->text_num - 1]);
  string_set(&info->text_keys[info->text_num - 1], key);

  string_init(&info->text_strings[info->text_num - 1]);
  string_set(&info->text_strings[info->text_num - 1], str);

  return 0;
}

/******************************************************************************/

static void LodePNGIText_init(LodePNGInfo* info)
{
  info->itext_num = 0;
  info->itext_keys = NULL;
  info->itext_langtags = NULL;
  info->itext_transkeys = NULL;
  info->itext_strings = NULL;
}

static void LodePNGIText_cleanup(LodePNGInfo* info)
{
  size_t i;
  for(i = 0; i < info->itext_num; i++)
  {
    string_cleanup(&info->itext_keys[i]);
    string_cleanup(&info->itext_langtags[i]);
    string_cleanup(&info->itext_transkeys[i]);
    string_cleanup(&info->itext_strings[i]);
  }
  myfree(info->itext_keys);
  myfree(info->itext_langtags);
  myfree(info->itext_transkeys);
  myfree(info->itext_strings);
}

static unsigned LodePNGIText_copy(LodePNGInfo* dest, const LodePNGInfo* source)
{
  size_t i = 0;
  dest->itext_keys = 0;
  dest->itext_langtags = 0;
  dest->itext_transkeys = 0;
  dest->itext_strings = 0;
  dest->itext_num = 0;
  for(i = 0; i < source->itext_num; i++)
  {
    CERROR_TRY_RETURN(lodepng_add_itext(dest, source->itext_keys[i], source->itext_langtags[i],
                                        source->itext_transkeys[i], source->itext_strings[i]));
  }
  return 0;
}

void lodepng_clear_itext(LodePNGInfo* info)
{
  LodePNGIText_cleanup(info);
}

unsigned lodepng_add_itext(LodePNGInfo* info, const char* key, const char* langtag,
                           const char* transkey, const char* str)
{
  char** new_keys = (char**)(myrealloc(info->itext_keys, sizeof(char*) * (info->itext_num + 1)));
  char** new_langtags = (char**)(myrealloc(info->itext_langtags, sizeof(char*) * (info->itext_num + 1)));
  char** new_transkeys = (char**)(myrealloc(info->itext_transkeys, sizeof(char*) * (info->itext_num + 1)));
  char** new_strings = (char**)(myrealloc(info->itext_strings, sizeof(char*) * (info->itext_num + 1)));
  if(!new_keys || !new_langtags || !new_transkeys || !new_strings)
  {
    myfree(new_keys);
    myfree(new_langtags);
    myfree(new_transkeys);
    myfree(new_strings);
    return 83; /*alloc fail*/
  }

  info->itext_num++;
  info->itext_keys = new_keys;
  info->itext_langtags = new_langtags;
  info->itext_transkeys = new_transkeys;
  info->itext_strings = new_strings;

  string_init(&info->itext_keys[info->itext_num - 1]);
  string_set(&info->itext_keys[info->itext_num - 1], key);

  string_init(&info->itext_langtags[info->itext_num - 1]);
  string_set(&info->itext_langtags[info->itext_num - 1], langtag);

  string_init(&info->itext_transkeys[info->itext_num - 1]);
  string_set(&info->itext_transkeys[info->itext_num - 1], transkey);

  string_init(&info->itext_strings[info->itext_num - 1]);
  string_set(&info->itext_strings[info->itext_num - 1], str);

  return 0;
}
#endif /*LODEPNG_COMPILE_ANCILLARY_CHUNKS*/

void lodepng_info_init(LodePNGInfo* info)
{
  lodepng_color_mode_init(&info->color);
  info->interlace_method = 0;
  info->compression_method = 0;
  info->filter_method = 0;
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
  info->background_defined = 0;
  info->background_r = info->background_g = info->background_b = 0;

  LodePNGText_init(info);
  LodePNGIText_init(info);

  info->time_defined = 0;
  info->phys_defined = 0;

  LodePNGUnknownChunks_init(info);
#endif /*LODEPNG_COMPILE_ANCILLARY_CHUNKS*/
}

void lodepng_info_cleanup(LodePNGInfo* info)
{
  lodepng_color_mode_cleanup(&info->color);
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
  LodePNGText_cleanup(info);
  LodePNGIText_cleanup(info);

  LodePNGUnknownChunks_cleanup(info);
#endif /*LODEPNG_COMPILE_ANCILLARY_CHUNKS*/
}

unsigned lodepng_info_copy(LodePNGInfo* dest, const LodePNGInfo* source)
{
  lodepng_info_cleanup(dest);
  *dest = *source;
  lodepng_color_mode_init(&dest->color);
  CERROR_TRY_RETURN(lodepng_color_mode_copy(&dest->color, &source->color));

#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
  CERROR_TRY_RETURN(LodePNGText_copy(dest, source));
  CERROR_TRY_RETURN(LodePNGIText_copy(dest, source));

  LodePNGUnknownChunks_init(dest);
  CERROR_TRY_RETURN(LodePNGUnknownChunks_copy(dest, source));
#endif /*LODEPNG_COMPILE_ANCILLARY_CHUNKS*/
  return 0;
}

void lodepng_info_swap(LodePNGInfo* a, LodePNGInfo* b)
{
  LodePNGInfo temp = *a;
  *a = *b;
  *b = temp;
}

/* ////////////////////////////////////////////////////////////////////////// */

/*index: bitgroup index, bits: bitgroup size(1, 2 or 4, in: bitgroup value, out: octet array to add bits to*/
static void addColorBits(unsigned char* out, size_t index, unsigned bits, unsigned in)
{
  /*p = the partial index in the byte, e.g. with 4 palettebits it is 0 for first half or 1 for second half*/
  unsigned p = index % (8 / bits);
  in &= (1 << bits) - 1; /*filter out any other bits of the input value*/
  in = in << (bits * (8 / bits - p - 1));
  if(p == 0) out[index * bits / 8] = in;
  else out[index * bits / 8] |= in;
}

typedef struct ColorTree ColorTree;

/*
One node of a color tree
This is the data structure used to count the number of unique colors and to get a palette
index for a color. It's like an octree, but because the alpha channel is used too, each
node has 16 instead of 8 children.
*/
struct ColorTree
{
  ColorTree* children[16]; /*up to 16 pointers to ColorTree of next level*/
  int index; /*the payload. Only has a meaningful value if this is in the last level*/
};

static void color_tree_init(ColorTree* tree)
{
  int i;
  for(i = 0; i < 16; i++) tree->children[i] = 0;
  tree->index = -1;
}

static void color_tree_cleanup(ColorTree* tree)
{
  int i;
  for(i = 0; i < 16; i++)
  {
    if(tree->children[i])
    {
      color_tree_cleanup(tree->children[i]);
      myfree(tree->children[i]);
    }
  }
}

/*returns -1 if color not present, its index otherwise*/
static int color_tree_get(ColorTree* tree, unsigned char r, unsigned char g, unsigned char b, unsigned char a)
{
  int bit = 0;
  for(bit = 0; bit < 8; bit++)
  {
    int i = 8 * ((r >> bit) & 1) + 4 * ((g >> bit) & 1) + 2 * ((b >> bit) & 1) + 1 * ((a >> bit) & 1);
    if(!tree->children[i]) return -1;
    else tree = tree->children[i];
  }
  return tree ? tree->index : -1;
}

#ifdef LODEPNG_COMPILE_ENCODER
static int color_tree_has(ColorTree* tree, unsigned char r, unsigned char g, unsigned char b, unsigned char a)
{
  return color_tree_get(tree, r, g, b, a) >= 0;
}
#endif /*LODEPNG_COMPILE_ENCODER*/

/*color is not allowed to already exist.
Index should be >= 0 (it's signed to be compatible with using -1 for "doesn't exist")*/
static void color_tree_add(ColorTree* tree,
                           unsigned char r, unsigned char g, unsigned char b, unsigned char a, int index)
{
  int bit;
  for(bit = 0; bit < 8; bit++)
  {
    int i = 8 * ((r >> bit) & 1) + 4 * ((g >> bit) & 1) + 2 * ((b >> bit) & 1) + 1 * ((a >> bit) & 1);
    if(!tree->children[i])
    {
      tree->children[i] = (ColorTree*)mymalloc(sizeof(ColorTree));
      color_tree_init(tree->children[i]);
    }
    tree = tree->children[i];
  }
  tree->index = index;
}

/*put a pixel, given its RGBA color, into image of any color type*/
static unsigned rgba8ToPixel(unsigned char* out, size_t i,
                             const LodePNGColorMode* mode, ColorTree* tree /*for palette*/,
                             unsigned char r, unsigned char g, unsigned char b, unsigned char a)
{
  if(mode->colortype == LCT_GREY)
  {
    unsigned char grey = r; /*((unsigned short)r + g + b) / 3*/;
    if(mode->bitdepth == 8) out[i] = grey;
    else if(mode->bitdepth == 16) out[i * 2 + 0] = out[i * 2 + 1] = grey;
    else
    {
      /*take the most significant bits of grey*/
      grey = (grey >> (8 - mode->bitdepth)) & ((1 << mode->bitdepth) - 1);
      addColorBits(out, i, mode->bitdepth, grey);
    }
  }
  else if(mode->colortype == LCT_RGB)
  {
    if(mode->bitdepth == 8)
    {
      out[i * 3 + 0] = r;
      out[i * 3 + 1] = g;
      out[i * 3 + 2] = b;
    }
    else
    {
      out[i * 6 + 0] = out[i * 6 + 1] = r;
      out[i * 6 + 2] = out[i * 6 + 3] = g;
      out[i * 6 + 4] = out[i * 6 + 5] = b;
    }
  }
  else if(mode->colortype == LCT_PALETTE)
  {
    int index = color_tree_get(tree, r, g, b, a);
    if(index < 0) return 82; /*color not in palette*/
    if(mode->bitdepth == 8) out[i] = index;
    else addColorBits(out, i, mode->bitdepth, index);
  }
  else if(mode->colortype == LCT_GREY_ALPHA)
  {
    unsigned char grey = r; /*((unsigned short)r + g + b) / 3*/;
    if(mode->bitdepth == 8)
    {
      out[i * 2 + 0] = grey;
      out[i * 2 + 1] = a;
    }
    else if(mode->bitdepth == 16)
    {
      out[i * 4 + 0] = out[i * 4 + 1] = grey;
      out[i * 4 + 2] = out[i * 4 + 3] = a;
    }
  }
  else if(mode->colortype == LCT_RGBA)
  {
    if(mode->bitdepth == 8)
    {
      out[i * 4 + 0] = r;
      out[i * 4 + 1] = g;
      out[i * 4 + 2] = b;
      out[i * 4 + 3] = a;
    }
    else
    {
      out[i * 8 + 0] = out[i * 8 + 1] = r;
      out[i * 8 + 2] = out[i * 8 + 3] = g;
      out[i * 8 + 4] = out[i * 8 + 5] = b;
      out[i * 8 + 6] = out[i * 8 + 7] = a;
    }
  }

  return 0; /*no error*/
}

/*put a pixel, given its RGBA16 color, into image of any color 16-bitdepth type*/
static unsigned rgba16ToPixel(unsigned char* out, size_t i,
                              const LodePNGColorMode* mode,
                              unsigned short r, unsigned short g, unsigned short b, unsigned short a)
{
  if(mode->bitdepth != 16) return 85; /*must be 16 for this function*/
  if(mode->colortype == LCT_GREY)
  {
    unsigned short grey = r; /*((unsigned)r + g + b) / 3*/;
    out[i * 2 + 0] = (grey >> 8) & 255;
    out[i * 2 + 1] = grey & 255;
  }
  else if(mode->colortype == LCT_RGB)
  {
    out[i * 6 + 0] = (r >> 8) & 255;
    out[i * 6 + 1] = r & 255;
    out[i * 6 + 2] = (g >> 8) & 255;
    out[i * 6 + 3] = g & 255;
    out[i * 6 + 4] = (b >> 8) & 255;
    out[i * 6 + 5] = b & 255;
  }
  else if(mode->colortype == LCT_GREY_ALPHA)
  {
    unsigned short grey = r; /*((unsigned)r + g + b) / 3*/;
    out[i * 4 + 0] = (grey >> 8) & 255;
    out[i * 4 + 1] = grey & 255;
    out[i * 4 + 2] = (a >> 8) & 255;
    out[i * 4 + 3] = a & 255;
  }
  else if(mode->colortype == LCT_RGBA)
  {
    out[i * 8 + 0] = (r >> 8) & 255;
    out[i * 8 + 1] = r & 255;
    out[i * 8 + 2] = (g >> 8) & 255;
    out[i * 8 + 3] = g & 255;
    out[i * 8 + 4] = (b >> 8) & 255;
    out[i * 8 + 5] = b & 255;
    out[i * 8 + 6] = (a >> 8) & 255;
    out[i * 8 + 7] = a & 255;
  }

  return 0; /*no error*/
}

/*Get RGBA8 color of pixel with index i (y * width + x) from the raw image with given color type.*/
static unsigned getPixelColorRGBA8(unsigned char* r, unsigned char* g,
                                   unsigned char* b, unsigned char* a,
                                   const unsigned char* in, size_t i,
                                   const LodePNGColorMode* mode)
{
  if(mode->colortype == LCT_GREY)
  {
    if(mode->bitdepth == 8)
    {
      *r = *g = *b = in[i];
      if(mode->key_defined && *r == mode->key_r) *a = 0;
      else *a = 255;
    }
    else if(mode->bitdepth == 16)
    {
      *r = *g = *b = in[i * 2 + 0];
      if(mode->key_defined && 256U * in[i * 2 + 0] + in[i * 2 + 1] == mode->key_r) *a = 0;
      else *a = 255;
    }
    else
    {
      unsigned highest = ((1U << mode->bitdepth) - 1U); /*highest possible value for this bit depth*/
      size_t j = i * mode->bitdepth;
      unsigned value = readBitsFromReversedStream(&j, in, mode->bitdepth);
      *r = *g = *b = (value * 255) / highest;
      if(mode->key_defined && value == mode->key_r) *a = 0;
      else *a = 255;
    }
  }
  else if(mode->colortype == LCT_RGB)
  {
    if(mode->bitdepth == 8)
    {
      *r = in[i * 3 + 0]; *g = in[i * 3 + 1]; *b = in[i * 3 + 2];
      if(mode->key_defined && *r == mode->key_r && *g == mode->key_g && *b == mode->key_b) *a = 0;
      else *a = 255;
    }
    else
    {
      *r = in[i * 6 + 0];
      *g = in[i * 6 + 2];
      *b = in[i * 6 + 4];
      if(mode->key_defined && 256U * in[i * 6 + 0] + in[i * 6 + 1] == mode->key_r
         && 256U * in[i * 6 + 2] + in[i * 6 + 3] == mode->key_g
         && 256U * in[i * 6 + 4] + in[i * 6 + 5] == mode->key_b) *a = 0;
      else *a = 255;
    }
  }
  else if(mode->colortype == LCT_PALETTE)
  {
    unsigned index;
    if(mode->bitdepth == 8) index = in[i];
    else
    {
      size_t j = i * mode->bitdepth;
      index = readBitsFromReversedStream(&j, in, mode->bitdepth);
    }
    if(index >= mode->palettesize) return 47; /*index out of palette*/
    *r = mode->palette[index * 4 + 0];
    *g = mode->palette[index * 4 + 1];
    *b = mode->palette[index * 4 + 2];
    *a = mode->palette[index * 4 + 3];
  }
  else if(mode->colortype == LCT_GREY_ALPHA)
  {
    if(mode->bitdepth == 8)
    {
      *r = *g = *b = in[i * 2 + 0];
      *a = in[i * 2 + 1];
    }
    else
    {
      *r = *g = *b = in[i * 4 + 0];
      *a = in[i * 4 + 2];
    }
  }
  else if(mode->colortype == LCT_RGBA)
  {
    if(mode->bitdepth == 8)
    {
      *r = in[i * 4 + 0];
      *g = in[i * 4 + 1];
      *b = in[i * 4 + 2];
      *a = in[i * 4 + 3];
    }
    else
    {
      *r = in[i * 8 + 0];
      *g = in[i * 8 + 2];
      *b = in[i * 8 + 4];
      *a = in[i * 8 + 6];
    }
  }

  return 0; /*no error*/
}

/*Similar to getPixelColorRGBA8, but with all the for loops inside of the color
mode test cases, optimized to convert the colors much faster, when converting
to RGBA or RGB with 8 bit per cannel. buffer must be RGBA or RGB output with
enough memory, if has_alpha is true the output is RGBA. mode has the color mode
of the input buffer.*/
static unsigned getPixelColorsRGBA8(unsigned char* buffer, size_t numpixels,
                                    unsigned has_alpha, const unsigned char* in,
                                    const LodePNGColorMode* mode)
{
  unsigned num_channels = has_alpha ? 4 : 3;
  size_t i;
  if(mode->colortype == LCT_GREY)
  {
    if(mode->bitdepth == 8)
    {
      for(i = 0; i < numpixels; i++, buffer += num_channels)
      {
        buffer[0] = buffer[1] = buffer[2] = in[i];
        if(has_alpha) buffer[3] = mode->key_defined && in[i] == mode->key_r ? 0 : 255;
      }
    }
    else if(mode->bitdepth == 16)
    {
      for(i = 0; i < numpixels; i++, buffer += num_channels)
      {
        buffer[0] = buffer[1] = buffer[2] = in[i * 2];
        if(has_alpha) buffer[3] = mode->key_defined && 256U * in[i * 2 + 0] + in[i * 2 + 1] == mode->key_r ? 0 : 255;
      }
    }
    else
    {
      unsigned highest = ((1U << mode->bitdepth) - 1U); /*highest possible value for this bit depth*/
      size_t j = 0;
      for(i = 0; i < numpixels; i++, buffer += num_channels)
      {
        unsigned value = readBitsFromReversedStream(&j, in, mode->bitdepth);
        buffer[0] = buffer[1] = buffer[2] = (value * 255) / highest;
        if(has_alpha) buffer[3] = mode->key_defined && value == mode->key_r ? 0 : 255;
      }
    }
  }
  else if(mode->colortype == LCT_RGB)
  {
    if(mode->bitdepth == 8)
    {
      for(i = 0; i < numpixels; i++, buffer += num_channels)
      {
        buffer[0] = in[i * 3 + 0];
        buffer[1] = in[i * 3 + 1];
        buffer[2] = in[i * 3 + 2];
        if(has_alpha) buffer[3] = mode->key_defined && buffer[0] == mode->key_r
           && buffer[1]== mode->key_g && buffer[2] == mode->key_b ? 0 : 255;
      }
    }
    else
    {
      for(i = 0; i < numpixels; i++, buffer += num_channels)
      {
        buffer[0] = in[i * 6 + 0];
        buffer[1] = in[i * 6 + 2];
        buffer[2] = in[i * 6 + 4];
        if(has_alpha) buffer[3] = mode->key_defined
           && 256U * in[i * 6 + 0] + in[i * 6 + 1] == mode->key_r
           && 256U * in[i * 6 + 2] + in[i * 6 + 3] == mode->key_g
           && 256U * in[i * 6 + 4] + in[i * 6 + 5] == mode->key_b ? 0 : 255;
      }
    }
  }
  else if(mode->colortype == LCT_PALETTE)
  {
    unsigned index;
    size_t j = 0;
    for(i = 0; i < numpixels; i++, buffer += num_channels)
    {
      if(mode->bitdepth == 8) index = in[i];
      else index = readBitsFromReversedStream(&j, in, mode->bitdepth);
      if(index >= mode->palettesize) return 47; /*index out of palette*/
      buffer[0] = mode->palette[index * 4 + 0];
      buffer[1] = mode->palette[index * 4 + 1];
      buffer[2] = mode->palette[index * 4 + 2];
      if(has_alpha) buffer[3] = mode->palette[index * 4 + 3];
    }
  }
  else if(mode->colortype == LCT_GREY_ALPHA)
  {
    if(mode->bitdepth == 8)
    {
      for(i = 0; i < numpixels; i++, buffer += num_channels)
      {
        buffer[0] = buffer[1] = buffer[2] = in[i * 2 + 0];
        if(has_alpha) buffer[3] = in[i * 2 + 1];
      }
    }
    else
    {
      for(i = 0; i < numpixels; i++, buffer += num_channels)
      {
        buffer[0] = buffer[1] = buffer[2] = in[i * 4 + 0];
        if(has_alpha) buffer[3] = in[i * 4 + 2];
      }
    }
  }
  else if(mode->colortype == LCT_RGBA)
  {
    if(mode->bitdepth == 8)
    {
      for(i = 0; i < numpixels; i++, buffer += num_channels)
      {
        buffer[0] = in[i * 4 + 0];
        buffer[1] = in[i * 4 + 1];
        buffer[2] = in[i * 4 + 2];
        if(has_alpha) buffer[3] = in[i * 4 + 3];
      }
    }
    else
    {
      for(i = 0; i < numpixels; i++, buffer += num_channels)
      {
        buffer[0] = in[i * 8 + 0];
        buffer[1] = in[i * 8 + 2];
        buffer[2] = in[i * 8 + 4];
        if(has_alpha) buffer[3] = in[i * 8 + 6];
      }
    }
  }

  return 0; /*no error*/
}

/*Get RGBA16 color of pixel with index i (y * width + x) from the raw image with
given color type, but the given color type must be 16-bit itself.*/
static unsigned getPixelColorRGBA16(unsigned short* r, unsigned short* g, unsigned short* b, unsigned short* a,
                                    const unsigned char* in, size_t i, const LodePNGColorMode* mode)
{
  if(mode->bitdepth != 16) return 85; /*error: this function only supports 16-bit input*/

  if(mode->colortype == LCT_GREY)
  {
    *r = *g = *b = 256 * in[i * 2 + 0] + in[i * 2 + 1];
    if(mode->key_defined && 256U * in[i * 2 + 0] + in[i * 2 + 1] == mode->key_r) *a = 0;
    else *a = 65535;
  }
  else if(mode->colortype == LCT_RGB)
  {
    *r = 256 * in[i * 6 + 0] + in[i * 6 + 1];
    *g = 256 * in[i * 6 + 2] + in[i * 6 + 3];
    *b = 256 * in[i * 6 + 4] + in[i * 6 + 5];
    if(mode->key_defined && 256U * in[i * 6 + 0] + in[i * 6 + 1] == mode->key_r
       && 256U * in[i * 6 + 2] + in[i * 6 + 3] == mode->key_g
       && 256U * in[i * 6 + 4] + in[i * 6 + 5] == mode->key_b) *a = 0;
    else *a = 65535;
  }
  else if(mode->colortype == LCT_GREY_ALPHA)
  {
    *r = *g = *b = 256 * in[i * 4 + 0] + in[i * 4 + 1];
    *a = 256 * in[i * 4 + 2] + in[i * 4 + 3];
  }
  else if(mode->colortype == LCT_RGBA)
  {
    *r = 256 * in[i * 8 + 0] + in[i * 8 + 1];
    *g = 256 * in[i * 8 + 2] + in[i * 8 + 3];
    *b = 256 * in[i * 8 + 4] + in[i * 8 + 5];
    *a = 256 * in[i * 8 + 6] + in[i * 8 + 7];
  }
  else return 85; /*error: this function only supports 16-bit input, not palettes*/

  return 0; /*no error*/
}

/*
converts from any color type to 24-bit or 32-bit (later maybe more supported). return value = LodePNG error code
the out buffer must have (w * h * bpp + 7) / 8 bytes, where bpp is the bits per pixel of the output color type
(lodepng_get_bpp) for < 8 bpp images, there may _not_ be padding bits at the end of scanlines.
*/
unsigned lodepng_convert(unsigned char* out, const unsigned char* in,
                         LodePNGColorMode* mode_out, LodePNGColorMode* mode_in,
                         unsigned w, unsigned h)
{
  unsigned error = 0;
  size_t i;
  ColorTree tree;
  size_t numpixels = w * h;

  if(lodepng_color_mode_equal(mode_out, mode_in))
  {
    size_t numbytes = lodepng_get_raw_size(w, h, mode_in);
    for(i = 0; i < numbytes; i++) out[i] = in[i];
    return error;
  }

  if(mode_out->colortype == LCT_PALETTE)
  {
    size_t palsize = 1 << mode_out->bitdepth;
    if(mode_out->palettesize < palsize) palsize = mode_out->palettesize;
    color_tree_init(&tree);
    for(i = 0; i < palsize; i++)
    {
      unsigned char* p = &mode_out->palette[i * 4];
      color_tree_add(&tree, p[0], p[1], p[2], p[3], i);
    }
  }

  if(mode_in->bitdepth == 16 && mode_out->bitdepth == 16)
  {
    for(i = 0; i < numpixels; i++)
    {
      unsigned short r = 0, g = 0, b = 0, a = 0;
      error = getPixelColorRGBA16(&r, &g, &b, &a, in, i, mode_in);
      if(error) break;
      error = rgba16ToPixel(out, i, mode_out, r, g, b, a);
      if(error) break;
    }
  }
  else if(mode_out->bitdepth == 8 && mode_out->colortype == LCT_RGBA)
  {
    error = getPixelColorsRGBA8(out, numpixels, 1, in, mode_in);
  }
  else if(mode_out->bitdepth == 8 && mode_out->colortype == LCT_RGB)
  {
    error = getPixelColorsRGBA8(out, numpixels, 0, in, mode_in);
  }
  else
  {
    unsigned char r = 0, g = 0, b = 0, a = 0;
    for(i = 0; i < numpixels; i++)
    {
      error = getPixelColorRGBA8(&r, &g, &b, &a, in, i, mode_in);
      if(error) break;
      error = rgba8ToPixel(out, i, mode_out, &tree, r, g, b, a);
      if(error) break;
    }
  }

  if(mode_out->colortype == LCT_PALETTE)
  {
    color_tree_cleanup(&tree);
  }

  return error;
}

#ifdef LODEPNG_COMPILE_ENCODER

typedef struct ColorProfile
{
  unsigned char sixteenbit; /*needs more than 8 bits per channel*/
  unsigned char sixteenbit_done;


  unsigned char colored; /*not greyscale*/
  unsigned char colored_done;

  unsigned char key; /*a color key is required, or more*/
  unsigned short key_r; /*these values are always in 16-bit bitdepth in the profile*/
  unsigned short key_g;
  unsigned short key_b;
  unsigned char alpha; /*alpha channel, or alpha palette, required*/
  unsigned char alpha_done;

  unsigned numcolors;
  ColorTree tree; /*for listing the counted colors, up to 256*/
  unsigned char* palette; /*size 1024. Remember up to the first 256 RGBA colors*/
  unsigned maxnumcolors; /*if more than that amount counted*/
  unsigned char numcolors_done;

  unsigned greybits; /*amount of bits required for greyscale (1, 2, 4, 8). Does not take 16 bit into account.*/
  unsigned char greybits_done;

} ColorProfile;

static void color_profile_init(ColorProfile* profile, LodePNGColorMode* mode)
{
  profile->sixteenbit = 0;
  profile->sixteenbit_done = mode->bitdepth == 16 ? 0 : 1;

  profile->colored = 0;
  profile->colored_done = lodepng_is_greyscale_type(mode) ? 1 : 0;

  profile->key = 0;
  profile->alpha = 0;
  profile->alpha_done = lodepng_can_have_alpha(mode) ? 0 : 1;

  profile->numcolors = 0;
  color_tree_init(&profile->tree);
  profile->palette = (unsigned char*)mymalloc(1024);
  profile->maxnumcolors = 257;
  if(lodepng_get_bpp(mode) <= 8)
  {
    int bpp = lodepng_get_bpp(mode);
    profile->maxnumcolors = bpp == 1 ? 2 : (bpp == 2 ? 4 : (bpp == 4 ? 16 : 256));
  }
  profile->numcolors_done = 0;

  profile->greybits = 1;
  profile->greybits_done = lodepng_get_bpp(mode) == 1 ? 1 : 0;
}

static void color_profile_cleanup(ColorProfile* profile)
{
  color_tree_cleanup(&profile->tree);
  myfree(profile->palette);
}

/*function used for debug purposes with C++*/
/*void printColorProfile(ColorProfile* p)
{
  std::cout << "sixteenbit: " << (int)p->sixteenbit << std::endl;
  std::cout << "sixteenbit_done: " << (int)p->sixteenbit_done << std::endl;
  std::cout << "colored: " << (int)p->colored << std::endl;
  std::cout << "colored_done: " << (int)p->colored_done << std::endl;
  std::cout << "key: " << (int)p->key << std::endl;
  std::cout << "key_r: " << (int)p->key_r << std::endl;
  std::cout << "key_g: " << (int)p->key_g << std::endl;
  std::cout << "key_b: " << (int)p->key_b << std::endl;
  std::cout << "alpha: " << (int)p->alpha << std::endl;
  std::cout << "alpha_done: " << (int)p->alpha_done << std::endl;
  std::cout << "numcolors: " << (int)p->numcolors << std::endl;
  std::cout << "maxnumcolors: " << (int)p->maxnumcolors << std::endl;
  std::cout << "numcolors_done: " << (int)p->numcolors_done << std::endl;
  std::cout << "greybits: " << (int)p->greybits << std::endl;
  std::cout << "greybits_done: " << (int)p->greybits_done << std::endl;
}*/

/*Returns how many bits needed to represent given value (max 8 bit)*/
unsigned getValueRequiredBits(unsigned short value)
{
  if(value == 0 || value == 255) return 1;
  /*The scaling of 2-bit and 4-bit values uses multiples of 85 and 17*/
  if(value % 17 == 0) return value % 85 == 0 ? 2 : 4;
  return 8;
}

/*profile must already have been inited with mode.
It's ok to set some parameters of profile to done already.*/
static unsigned get_color_profile(ColorProfile* profile,
                                  const unsigned char* in, size_t numpixels,
                                  LodePNGColorMode* mode)
{
  unsigned error = 0;
  size_t i;

  if(mode->bitdepth == 16)
  {
    for(i = 0; i < numpixels; i++)
    {
      unsigned short r, g, b, a;
      error = getPixelColorRGBA16(&r, &g, &b, &a, in, i, mode);
      if(error) break;

      /*a color is considered good for 8-bit if the first byte and the second byte are equal,
        (so if it's divisible through 257), NOT necessarily if the second byte is 0*/
      if(!profile->sixteenbit_done
          && (((r & 255) != ((r >> 8) & 255))
           || ((g & 255) != ((g >> 8) & 255))
           || ((b & 255) != ((b >> 8) & 255))))
      {
        profile->sixteenbit = 1;
        profile->sixteenbit_done = 1;
        profile->greybits_done = 1; /*greybits is not applicable anymore at 16-bit*/
        profile->numcolors_done = 1; /*counting colors no longer useful, palette doesn't support 16-bit*/
      }

      if(!profile->colored_done && (r != g || r != b))
      {
        profile->colored = 1;
        profile->colored_done = 1;
        profile->greybits_done = 1; /*greybits is not applicable anymore*/
      }

      if(!profile->alpha_done && a != 65535)
      {
        if(a == 0 && !(profile->key && (r != profile->key_r || g != profile->key_g || b != profile->key_b)))
        {
          if(!profile->key)
          {
            profile->key = 1;
            profile->key_r = r;
            profile->key_g = g;
            profile->key_b = b;
          }
        }
        else
        {
          profile->alpha = 1;
          profile->alpha_done = 1;
          profile->greybits_done = 1; /*greybits is not applicable anymore*/
        }
      }

      /* Color key cannot be used if an opaque pixel also has that RGB color. */
      if(!profile->alpha_done && a == 65535 && profile->key
          && r == profile->key_r && g == profile->key_g && b == profile->key_b)
      {
          profile->alpha = 1;
          profile->alpha_done = 1;
          profile->greybits_done = 1; /*greybits is not applicable anymore*/
      }

      if(!profile->greybits_done)
      {
        /*assuming 8-bit r, this test does not care about 16-bit*/
        unsigned bits = getValueRequiredBits(r);
        if(bits > profile->greybits) profile->greybits = bits;
        if(profile->greybits >= 8) profile->greybits_done = 1;
      }

      if(!profile->numcolors_done)
      {
        /*assuming 8-bit rgba, this test does not care about 16-bit*/
        if(!color_tree_has(&profile->tree, (unsigned char)r, (unsigned char)g, (unsigned char)b, (unsigned char)a))
        {
          color_tree_add(&profile->tree, (unsigned char)r, (unsigned char)g, (unsigned char)b, (unsigned char)a,
            profile->numcolors);
          if(profile->numcolors < 256)
          {
            unsigned char* p = profile->palette;
            unsigned i = profile->numcolors;
            p[i * 4 + 0] = (unsigned char)r;
            p[i * 4 + 1] = (unsigned char)g;
            p[i * 4 + 2] = (unsigned char)b;
            p[i * 4 + 3] = (unsigned char)a;
          }
          profile->numcolors++;
          if(profile->numcolors >= profile->maxnumcolors) profile->numcolors_done = 1;
        }
      }

      if(profile->alpha_done && profile->numcolors_done
      && profile->colored_done && profile->sixteenbit_done && profile->greybits_done)
      {
        break;
      }
    };
  }
  else /* < 16-bit */
  {
    for(i = 0; i < numpixels; i++)
    {
      unsigned char r = 0, g = 0, b = 0, a = 0;
      error = getPixelColorRGBA8(&r, &g, &b, &a, in, i, mode);
      if(error) break;

      if(!profile->colored_done && (r != g || r != b))
      {
        profile->colored = 1;
        profile->colored_done = 1;
        profile->greybits_done = 1; /*greybits is not applicable anymore*/
      }

      if(!profile->alpha_done && a != 255)
      {
        if(a == 0 && !(profile->key && (r != profile->key_r || g != profile->key_g || b != profile->key_b)))
        {
          if(!profile->key)
          {
            profile->key = 1;
            profile->key_r = r;
            profile->key_g = g;
            profile->key_b = b;
          }
        }
        else
        {
          profile->alpha = 1;
          profile->alpha_done = 1;
          profile->greybits_done = 1; /*greybits is not applicable anymore*/
        }
      }

      /* Color key cannot be used if an opaque pixel also has that RGB color. */
      if(!profile->alpha_done && a == 255 && profile->key
          && r == profile->key_r && g == profile->key_g && b == profile->key_b)
      {
          profile->alpha = 1;
          profile->alpha_done = 1;
          profile->greybits_done = 1; /*greybits is not applicable anymore*/
      }

      if(!profile->greybits_done)
      {
        unsigned bits = getValueRequiredBits(r);
        if(bits > profile->greybits) profile->greybits = bits;
        if(profile->greybits >= 8) profile->greybits_done = 1;
      }

      if(!profile->numcolors_done)
      {
        if(!color_tree_has(&profile->tree, r, g, b, a))
        {

          color_tree_add(&profile->tree, r, g, b, a, profile->numcolors);
          if(profile->numcolors < 256)
          {
            unsigned char* p = profile->palette;
            unsigned i = profile->numcolors;
            p[i * 4 + 0] = r;
            p[i * 4 + 1] = g;
            p[i * 4 + 2] = b;
            p[i * 4 + 3] = a;
          }
          profile->numcolors++;
          if(profile->numcolors >= profile->maxnumcolors) profile->numcolors_done = 1;
        }
      }

      if(profile->alpha_done && profile->numcolors_done && profile->colored_done && profile->greybits_done)
      {
        break;
      }
    };
  }

  /*make the profile's key always 16-bit for consistency*/
  if(mode->bitdepth < 16)
  {
    /*repeat each byte twice*/
    profile->key_r *= 257;
    profile->key_g *= 257;
    profile->key_b *= 257;
  }

  return error;
}

/*updates values of mode with a potentially smaller color model. mode_out should
contain the user chosen color model, but will be overwritten with the new chosen one.*/
static unsigned doAutoChooseColor(LodePNGColorMode* mode_out,
                                  const unsigned char* image, unsigned w, unsigned h, LodePNGColorMode* mode_in,
                                  LodePNGAutoConvert auto_convert)
{
  ColorProfile profile;
  unsigned error = 0;
  int no_nibbles = auto_convert == LAC_AUTO_NO_NIBBLES || auto_convert == LAC_AUTO_NO_NIBBLES_NO_PALETTE;
  int no_palette = auto_convert == LAC_AUTO_NO_PALETTE || auto_convert == LAC_AUTO_NO_NIBBLES_NO_PALETTE;

  if(auto_convert == LAC_ALPHA)
  {
    if(mode_out->colortype != LCT_RGBA && mode_out->colortype != LCT_GREY_ALPHA) return 0;
  }

  color_profile_init(&profile, mode_in);
  if(auto_convert == LAC_ALPHA)
  {
    profile.colored_done = 1;
    profile.greybits_done = 1;
    profile.numcolors_done = 1;
    profile.sixteenbit_done = 1;
  }
  error = get_color_profile(&profile, image, w * h, mode_in);

  if(!error && auto_convert == LAC_ALPHA)
  {
    if(!profile.alpha)
    {
      mode_out->colortype = (mode_out->colortype == LCT_RGBA ? LCT_RGB : LCT_GREY);
    }
  }
  else if(!error && auto_convert != LAC_ALPHA)
  {
    mode_out->key_defined = 0;

    if(profile.sixteenbit)
    {
      mode_out->bitdepth = 16;
      if(profile.alpha)
      {
        mode_out->colortype = profile.colored ? LCT_RGBA : LCT_GREY_ALPHA;
      }
      else
      {
        mode_out->colortype = profile.colored ? LCT_RGB : LCT_GREY;
        if(profile.key)
        {
          mode_out->key_defined = 1;
          mode_out->key_r = profile.key_r;
          mode_out->key_g = profile.key_g;
          mode_out->key_b = profile.key_b;
        }
      }
    }
    else /*less than 16 bits per channel*/
    {
      /*don't add palette overhead if image hasn't got a lot of pixels*/
      unsigned n = profile.numcolors;
      int palette_ok = !no_palette && n <= 256 && (n * 2 < w * h);
      unsigned palettebits = n <= 2 ? 1 : (n <= 4 ? 2 : (n <= 16 ? 4 : 8));
      int grey_ok = !profile.colored && !profile.alpha; /*grey without alpha, with potentially low bits*/
      if(palette_ok || grey_ok)
      {
        if(!palette_ok || (grey_ok && profile.greybits <= palettebits))
        {
          mode_out->colortype = LCT_GREY;
          mode_out->bitdepth = profile.greybits;
          if(profile.key)
          {
            unsigned keyval = profile.key_r;
            keyval &= (profile.greybits - 1); /*same subgroup of bits repeated, so taking right bits is fine*/
            mode_out->key_defined = 1;
            mode_out->key_r = keyval;
            mode_out->key_g = keyval;
            mode_out->key_b = keyval;
          }
        }
        else
        {
          /*fill in the palette*/
          unsigned i;
          unsigned char* p = profile.palette;
          for(i = 0; i < profile.numcolors; i++)
          {
            error = lodepng_palette_add(mode_out, p[i * 4 + 0], p[i * 4 + 1], p[i * 4 + 2], p[i * 4 + 3]);
            if(error) break;
          }

          mode_out->colortype = LCT_PALETTE;
          mode_out->bitdepth = palettebits;
        }
      }
      else /*8-bit per channel*/
      {
        mode_out->bitdepth = 8;
        if(profile.alpha)
        {
          mode_out->colortype = profile.colored ? LCT_RGBA : LCT_GREY_ALPHA;
        }
        else
        {
          mode_out->colortype = profile.colored ? LCT_RGB : LCT_GREY /*LCT_GREY normally won't occur, already done earlier*/;
          if(profile.key)
          {
            mode_out->key_defined = 1;
            mode_out->key_r = profile.key_r % 256;
            mode_out->key_g = profile.key_g % 256;
            mode_out->key_b = profile.key_b % 256;
          }
        }
      }
    }
  }

  color_profile_cleanup(&profile);

  if(mode_out->colortype == LCT_PALETTE && mode_in->palettesize == mode_out->palettesize)
  {
    /*In this case keep the palette order of the input, so that the user can choose an optimal one*/
    size_t i;
    for(i = 0; i < mode_in->palettesize * 4; i++)
    {
      mode_out->palette[i] = mode_in->palette[i];
    }
  }

  if(no_nibbles && mode_out->bitdepth < 8)
  {
    /*palette can keep its small amount of colors, as long as no indices use it*/
    mode_out->bitdepth = 8;
  }

  return error;
}

#endif /* #ifdef LODEPNG_COMPILE_ENCODER */

/*
Paeth predicter, used by PNG filter type 4
The parameters are of type short, but should come from unsigned chars, the shorts
are only needed to make the paeth calculation correct.
*/
static unsigned char paethPredictor(short a, short b, short c)
{
  short pa = abs(b - c);
  short pb = abs(a - c);
  short pc = abs(a + b - c - c);

  if(pc < pa && pc < pb) return (unsigned char)c;
  else if(pb < pa) return (unsigned char)b;
  else return (unsigned char)a;
}

/*shared values used by multiple Adam7 related functions*/

static const unsigned ADAM7_IX[7] = { 0, 4, 0, 2, 0, 1, 0 }; /*x start values*/
static const unsigned ADAM7_IY[7] = { 0, 0, 4, 0, 2, 0, 1 }; /*y start values*/
static const unsigned ADAM7_DX[7] = { 8, 8, 4, 4, 2, 2, 1 }; /*x delta values*/
static const unsigned ADAM7_DY[7] = { 8, 8, 8, 4, 4, 2, 2 }; /*y delta values*/

/*
Outputs various dimensions and positions in the image related to the Adam7 reduced images.
passw: output containing the width of the 7 passes
passh: output containing the height of the 7 passes
filter_passstart: output containing the index of the start and end of each
 reduced image with filter bytes
padded_passstart output containing the index of the start and end of each
 reduced image when without filter bytes but with padded scanlines
passstart: output containing the index of the start and end of each reduced
 image without padding between scanlines, but still padding between the images
w, h: width and height of non-interlaced image
bpp: bits per pixel
"padded" is only relevant if bpp is less than 8 and a scanline or image does not
 end at a full byte
*/
static void Adam7_getpassvalues(unsigned passw[7], unsigned passh[7], size_t filter_passstart[8],
                                size_t padded_passstart[8], size_t passstart[8], unsigned w, unsigned h, unsigned bpp)
{
  /*the passstart values have 8 values: the 8th one indicates the byte after the end of the 7th (= last) pass*/
  unsigned i;

  /*calculate width and height in pixels of each pass*/
  for(i = 0; i < 7; i++)
  {
    passw[i] = (w + ADAM7_DX[i] - ADAM7_IX[i] - 1) / ADAM7_DX[i];
    passh[i] = (h + ADAM7_DY[i] - ADAM7_IY[i] - 1) / ADAM7_DY[i];
    if(passw[i] == 0) passh[i] = 0;
    if(passh[i] == 0) passw[i] = 0;
  }

  filter_passstart[0] = padded_passstart[0] = passstart[0] = 0;
  for(i = 0; i < 7; i++)
  {
    /*if passw[i] is 0, it's 0 bytes, not 1 (no filtertype-byte)*/
    filter_passstart[i + 1] = filter_passstart[i]
                            + ((passw[i] && passh[i]) ? passh[i] * (1 + (passw[i] * bpp + 7) / 8) : 0);
    /*bits padded if needed to fill full byte at end of each scanline*/
    padded_passstart[i + 1] = padded_passstart[i] + passh[i] * ((passw[i] * bpp + 7) / 8);
    /*only padded at end of reduced image*/
    passstart[i + 1] = passstart[i] + (passh[i] * passw[i] * bpp + 7) / 8;
  }
}

#ifdef LODEPNG_COMPILE_DECODER

/* ////////////////////////////////////////////////////////////////////////// */
/* / PNG Decoder                                                            / */
/* ////////////////////////////////////////////////////////////////////////// */

/*read the information from the header and store it in the LodePNGInfo. return value is error*/
unsigned lodepng_inspect(unsigned* w, unsigned* h, LodePNGState* state,
                         const unsigned char* in, size_t insize)
{
  LodePNGInfo* info = &state->info_png;
  if(insize == 0 || in == 0)
  {
    CERROR_RETURN_ERROR(state->error, 48); /*error: the given data is empty*/
  }
  if(insize < 29)
  {
    CERROR_RETURN_ERROR(state->error, 27); /*error: the data length is smaller than the length of a PNG header*/
  }

  /*when decoding a new PNG image, make sure all parameters created after previous decoding are reset*/
  lodepng_info_cleanup(info);
  lodepng_info_init(info);

  if(in[0] != 137 || in[1] != 80 || in[2] != 78 || in[3] != 71
     || in[4] != 13 || in[5] != 10 || in[6] != 26 || in[7] != 10)
  {
    CERROR_RETURN_ERROR(state->error, 28); /*error: the first 8 bytes are not the correct PNG signature*/
  }
  if(in[12] != 'I' || in[13] != 'H' || in[14] != 'D' || in[15] != 'R')
  {
    CERROR_RETURN_ERROR(state->error, 29); /*error: it doesn't start with a IHDR chunk!*/
  }

  /*read the values given in the header*/
  *w = lodepng_read32bitInt(&in[16]);
  *h = lodepng_read32bitInt(&in[20]);
  info->color.bitdepth = in[24];
  info->color.colortype = (LodePNGColorType)in[25];
  info->compression_method = in[26];
  info->filter_method = in[27];
  info->interlace_method = in[28];

  if(!state->decoder.ignore_crc)
  {
    unsigned CRC = lodepng_read32bitInt(&in[29]);
    unsigned checksum = lodepng_crc32(&in[12], 17);
    if(CRC != checksum)
    {
      CERROR_RETURN_ERROR(state->error, 57); /*invalid CRC*/
    }
  }

  /*error: only compression method 0 is allowed in the specification*/
  if(info->compression_method != 0) CERROR_RETURN_ERROR(state->error, 32);
  /*error: only filter method 0 is allowed in the specification*/
  if(info->filter_method != 0) CERROR_RETURN_ERROR(state->error, 33);
  /*error: only interlace methods 0 and 1 exist in the specification*/
  if(info->interlace_method > 1) CERROR_RETURN_ERROR(state->error, 34);

  state->error = checkColorValidity(info->color.colortype, info->color.bitdepth);
  return state->error;
}

static unsigned unfilterScanline(unsigned char* recon, const unsigned char* scanline, const unsigned char* precon,
                                 size_t bytewidth, unsigned char filterType, size_t length)
{
  /*
  For PNG filter method 0
  unfilter a PNG image scanline by scanline. when the pixels are smaller than 1 byte,
  the filter works byte per byte (bytewidth = 1)
  precon is the previous unfiltered scanline, recon the result, scanline the current one
  the incoming scanlines do NOT include the filtertype byte, that one is given in the parameter filterType instead
  recon and scanline MAY be the same memory address! precon must be disjoint.
  */

  size_t i;
  switch(filterType)
  {
    case 0:
      for(i = 0; i < length; i++) recon[i] = scanline[i];
      break;
    case 1:
      for(i = 0; i < bytewidth; i++) recon[i] = scanline[i];
      for(i = bytewidth; i < length; i++) recon[i] = scanline[i] + recon[i - bytewidth];
      break;
    case 2:
      if(precon)
      {
        for(i = 0; i < length; i++) recon[i] = scanline[i] + precon[i];
      }
      else
      {
        for(i = 0; i < length; i++) recon[i] = scanline[i];
      }
      break;
    case 3:
      if(precon)
      {
        for(i = 0; i < bytewidth; i++) recon[i] = scanline[i] + precon[i] / 2;
        for(i = bytewidth; i < length; i++) recon[i] = scanline[i] + ((recon[i - bytewidth] + precon[i]) / 2);
      }
      else
      {
        for(i = 0; i < bytewidth; i++) recon[i] = scanline[i];
        for(i = bytewidth; i < length; i++) recon[i] = scanline[i] + recon[i - bytewidth] / 2;
      }
      break;
    case 4:
      if(precon)
      {
        for(i = 0; i < bytewidth; i++)
        {
          recon[i] = (scanline[i] + precon[i]); /*paethPredictor(0, precon[i], 0) is always precon[i]*/
        }
        for(i = bytewidth; i < length; i++)
        {
          recon[i] = (scanline[i] + paethPredictor(recon[i - bytewidth], precon[i], precon[i - bytewidth]));
        }
      }
      else
      {
        for(i = 0; i < bytewidth; i++)
        {
          recon[i] = scanline[i];
        }
        for(i = bytewidth; i < length; i++)
        {
          /*paethPredictor(recon[i - bytewidth], 0, 0) is always recon[i - bytewidth]*/
          recon[i] = (scanline[i] + recon[i - bytewidth]);
        }
      }
      break;
    default: return 36; /*error: unexisting filter type given*/
  }
  return 0;
}

static unsigned unfilter(unsigned char* out, const unsigned char* in, unsigned w, unsigned h, unsigned bpp)
{
  /*
  For PNG filter method 0
  this function unfilters a single image (e.g. without interlacing this is called once, with Adam7 seven times)
  out must have enough bytes allocated already, in must have the scanlines + 1 filtertype byte per scanline
  w and h are image dimensions or dimensions of reduced image, bpp is bits per pixel
  in and out are allowed to be the same memory address (but aren't the same size since in has the extra filter bytes)
  */

  unsigned y;
  unsigned char* prevline = 0;

  /*bytewidth is used for filtering, is 1 when bpp < 8, number of bytes per pixel otherwise*/
  size_t bytewidth = (bpp + 7) / 8;
  size_t linebytes = (w * bpp + 7) / 8;

  for(y = 0; y < h; y++)
  {
    size_t outindex = linebytes * y;
    size_t inindex = (1 + linebytes) * y; /*the extra filterbyte added to each row*/
    unsigned char filterType = in[inindex];

    CERROR_TRY_RETURN(unfilterScanline(&out[outindex], &in[inindex + 1], prevline, bytewidth, filterType, linebytes));

    prevline = &out[outindex];
  }

  return 0;
}

/*
in: Adam7 interlaced image, with no padding bits between scanlines, but between
 reduced images so that each reduced image starts at a byte.
out: the same pixels, but re-ordered so that they're now a non-interlaced image with size w*h
bpp: bits per pixel
out has the following size in bits: w * h * bpp.
in is possibly bigger due to padding bits between reduced images.
out must be big enough AND must be 0 everywhere if bpp < 8 in the current implementation
(because that's likely a little bit faster)
NOTE: comments about padding bits are only relevant if bpp < 8
*/
static void Adam7_deinterlace(unsigned char* out, const unsigned char* in, unsigned w, unsigned h, unsigned bpp)
{
  unsigned passw[7], passh[7];
  size_t filter_passstart[8], padded_passstart[8], passstart[8];
  unsigned i;

  Adam7_getpassvalues(passw, passh, filter_passstart, padded_passstart, passstart, w, h, bpp);

  if(bpp >= 8)
  {
    for(i = 0; i < 7; i++)
    {
      unsigned x, y, b;
      size_t bytewidth = bpp / 8;
      for(y = 0; y < passh[i]; y++)
      for(x = 0; x < passw[i]; x++)
      {
        size_t pixelinstart = passstart[i] + (y * passw[i] + x) * bytewidth;
        size_t pixeloutstart = ((ADAM7_IY[i] + y * ADAM7_DY[i]) * w + ADAM7_IX[i] + x * ADAM7_DX[i]) * bytewidth;
        for(b = 0; b < bytewidth; b++)
        {
          out[pixeloutstart + b] = in[pixelinstart + b];
        }
      }
    }
  }
  else /*bpp < 8: Adam7 with pixels < 8 bit is a bit trickier: with bit pointers*/
  {
    for(i = 0; i < 7; i++)
    {
      unsigned x, y, b;
      unsigned ilinebits = bpp * passw[i];
      unsigned olinebits = bpp * w;
      size_t obp, ibp; /*bit pointers (for out and in buffer)*/
      for(y = 0; y < passh[i]; y++)
      for(x = 0; x < passw[i]; x++)
      {
        ibp = (8 * passstart[i]) + (y * ilinebits + x * bpp);
        obp = (ADAM7_IY[i] + y * ADAM7_DY[i]) * olinebits + (ADAM7_IX[i] + x * ADAM7_DX[i]) * bpp;
        for(b = 0; b < bpp; b++)
        {
          unsigned char bit = readBitFromReversedStream(&ibp, in);
          /*note that this function assumes the out buffer is completely 0, use setBitOfReversedStream otherwise*/
          setBitOfReversedStream0(&obp, out, bit);
        }
      }
    }
  }
}

static void removePaddingBits(unsigned char* out, const unsigned char* in,
                              size_t olinebits, size_t ilinebits, unsigned h)
{
  /*
  After filtering there are still padding bits if scanlines have non multiple of 8 bit amounts. They need
  to be removed (except at last scanline of (Adam7-reduced) image) before working with pure image buffers
  for the Adam7 code, the color convert code and the output to the user.
  in and out are allowed to be the same buffer, in may also be higher but still overlapping; in must
  have >= ilinebits*h bits, out must have >= olinebits*h bits, olinebits must be <= ilinebits
  also used to move bits after earlier such operations happened, e.g. in a sequence of reduced images from Adam7
  only useful if (ilinebits - olinebits) is a value in the range 1..7
  */
  unsigned y;
  size_t diff = ilinebits - olinebits;
  size_t ibp = 0, obp = 0; /*input and output bit pointers*/
  for(y = 0; y < h; y++)
  {
    size_t x;
    for(x = 0; x < olinebits; x++)
    {
      unsigned char bit = readBitFromReversedStream(&ibp, in);
      setBitOfReversedStream(&obp, out, bit);
    }
    ibp += diff;
  }
}

/*out must be buffer big enough to contain full image, and in must contain the full decompressed data from
the IDAT chunks (with filter index bytes and possible padding bits)
return value is error*/
static unsigned postProcessScanlines(unsigned char* out, unsigned char* in,
                                     unsigned w, unsigned h, const LodePNGInfo* info_png)
{
  /*
  This function converts the filtered-padded-interlaced data into pure 2D image buffer with the PNG's colortype.
  Steps:
  *) if no Adam7: 1) unfilter 2) remove padding bits (= posible extra bits per scanline if bpp < 8)
  *) if adam7: 1) 7x unfilter 2) 7x remove padding bits 3) Adam7_deinterlace
  NOTE: the in buffer will be overwritten with intermediate data!
  */
  unsigned bpp = lodepng_get_bpp(&info_png->color);
  if(bpp == 0) return 31; /*error: invalid colortype*/

  if(info_png->interlace_method == 0)
  {
    if(bpp < 8 && w * bpp != ((w * bpp + 7) / 8) * 8)
    {
      CERROR_TRY_RETURN(unfilter(in, in, w, h, bpp));
      removePaddingBits(out, in, w * bpp, ((w * bpp + 7) / 8) * 8, h);
    }
    /*we can immediatly filter into the out buffer, no other steps needed*/
    else CERROR_TRY_RETURN(unfilter(out, in, w, h, bpp));
  }
  else /*interlace_method is 1 (Adam7)*/
  {
    unsigned passw[7], passh[7]; size_t filter_passstart[8], padded_passstart[8], passstart[8];
    unsigned i;

    Adam7_getpassvalues(passw, passh, filter_passstart, padded_passstart, passstart, w, h, bpp);

    for(i = 0; i < 7; i++)
    {
      CERROR_TRY_RETURN(unfilter(&in[padded_passstart[i]], &in[filter_passstart[i]], passw[i], passh[i], bpp));
      /*TODO: possible efficiency improvement: if in this reduced image the bits fit nicely in 1 scanline,
      move bytes instead of bits or move not at all*/
      if(bpp < 8)
      {
        /*remove padding bits in scanlines; after this there still may be padding
        bits between the different reduced images: each reduced image still starts nicely at a byte*/
        removePaddingBits(&in[passstart[i]], &in[padded_passstart[i]], passw[i] * bpp,
                          ((passw[i] * bpp + 7) / 8) * 8, passh[i]);
      }
    }

    Adam7_deinterlace(out, in, w, h, bpp);
  }

  return 0;
}

static unsigned readChunk_PLTE(LodePNGColorMode* color, const unsigned char* data, size_t chunkLength)
{
  unsigned pos = 0, i;
  if(color->palette) myfree(color->palette);
  color->palettesize = chunkLength / 3;
  color->palette = (unsigned char*)mymalloc(4 * color->palettesize);
  if(!color->palette && color->palettesize)
  {
    color->palettesize = 0;
    return 83; /*alloc fail*/
  }
  if(color->palettesize > 256) return 38; /*error: palette too big*/

  for(i = 0; i < color->palettesize; i++)
  {
    color->palette[4 * i + 0] = data[pos++]; /*R*/
    color->palette[4 * i + 1] = data[pos++]; /*G*/
    color->palette[4 * i + 2] = data[pos++]; /*B*/
    color->palette[4 * i + 3] = 255; /*alpha*/
  }

  return 0; /* OK */
}

static unsigned readChunk_tRNS(LodePNGColorMode* color, const unsigned char* data, size_t chunkLength)
{
  unsigned i;
  if(color->colortype == LCT_PALETTE)
  {
    /*error: more alpha values given than there are palette entries*/
    if(chunkLength > color->palettesize) return 38;

    for(i = 0; i < chunkLength; i++) color->palette[4 * i + 3] = data[i];
  }
  else if(color->colortype == LCT_GREY)
  {
    /*error: this chunk must be 2 bytes for greyscale image*/
    if(chunkLength != 2) return 30;

    color->key_defined = 1;
    color->key_r = color->key_g = color->key_b = 256 * data[0] + data[1];
  }
  else if(color->colortype == LCT_RGB)
  {
    /*error: this chunk must be 6 bytes for RGB image*/
    if(chunkLength != 6) return 41;

    color->key_defined = 1;
    color->key_r = 256 * data[0] + data[1];
    color->key_g = 256 * data[2] + data[3];
    color->key_b = 256 * data[4] + data[5];
  }
  else return 42; /*error: tRNS chunk not allowed for other color models*/

  return 0; /* OK */
}


#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
/*background color chunk (bKGD)*/
static unsigned readChunk_bKGD(LodePNGInfo* info, const unsigned char* data, size_t chunkLength)
{
  if(info->color.colortype == LCT_PALETTE)
  {
    /*error: this chunk must be 1 byte for indexed color image*/
    if(chunkLength != 1) return 43;

    info->background_defined = 1;
    info->background_r = info->background_g = info->background_b = data[0];
  }
  else if(info->color.colortype == LCT_GREY || info->color.colortype == LCT_GREY_ALPHA)
  {
    /*error: this chunk must be 2 bytes for greyscale image*/
    if(chunkLength != 2) return 44;

    info->background_defined = 1;
    info->background_r = info->background_g = info->background_b
                                 = 256 * data[0] + data[1];
  }
  else if(info->color.colortype == LCT_RGB || info->color.colortype == LCT_RGBA)
  {
    /*error: this chunk must be 6 bytes for greyscale image*/
    if(chunkLength != 6) return 45;

    info->background_defined = 1;
    info->background_r = 256 * data[0] + data[1];
    info->background_g = 256 * data[2] + data[3];
    info->background_b = 256 * data[4] + data[5];
  }

  return 0; /* OK */
}

/*text chunk (tEXt)*/
static unsigned readChunk_tEXt(LodePNGInfo* info, const unsigned char* data, size_t chunkLength)
{
  unsigned error = 0;
  char *key = 0, *str = 0;
  unsigned i;

  while(!error) /*not really a while loop, only used to break on error*/
  {
    unsigned length, string2_begin;

    length = 0;
    while(length < chunkLength && data[length] != 0) length++;
    /*even though it's not allowed by the standard, no error is thrown if
    there's no null termination char, if the text is empty*/
    if(length < 1 || length > 79) CERROR_BREAK(error, 89); /*keyword too short or long*/

    key = (char*)mymalloc(length + 1);
    if(!key) CERROR_BREAK(error, 83); /*alloc fail*/

    key[length] = 0;
    for(i = 0; i < length; i++) key[i] = data[i];

    string2_begin = length + 1; /*skip keyword null terminator*/

    length = chunkLength < string2_begin ? 0 : chunkLength - string2_begin;
    str = (char*)mymalloc(length + 1);
    if(!str) CERROR_BREAK(error, 83); /*alloc fail*/

    str[length] = 0;
    for(i = 0; i < length; i++) str[i] = data[string2_begin + i];

    error = lodepng_add_text(info, key, str);

    break;
  }

  myfree(key);
  myfree(str);

  return error;
}

/*compressed text chunk (zTXt)*/
static unsigned readChunk_zTXt(LodePNGInfo* info, const LodePNGDecompressSettings* zlibsettings,
                               const unsigned char* data, size_t chunkLength)
{
  unsigned error = 0;
  unsigned i;

  unsigned length, string2_begin;
  char *key = 0;
  ucvector decoded;

  ucvector_init(&decoded);

  while(!error) /*not really a while loop, only used to break on error*/
  {
    for(length = 0; length < chunkLength && data[length] != 0; length++) ;
    if(length + 2 >= chunkLength) CERROR_BREAK(error, 75); /*no null termination, corrupt?*/
    if(length < 1 || length > 79) CERROR_BREAK(error, 89); /*keyword too short or long*/

    key = (char*)mymalloc(length + 1);
    if(!key) CERROR_BREAK(error, 83); /*alloc fail*/

    key[length] = 0;
    for(i = 0; i < length; i++) key[i] = data[i];

    if(data[length + 1] != 0) CERROR_BREAK(error, 72); /*the 0 byte indicating compression must be 0*/

    string2_begin = length + 2;
    if(string2_begin > chunkLength) CERROR_BREAK(error, 75); /*no null termination, corrupt?*/

    length = chunkLength - string2_begin;
    /*will fail if zlib error, e.g. if length is too small*/
    error = zlib_decompress(&decoded.data, &decoded.size,
                            (unsigned char*)(&data[string2_begin]),
                            length, zlibsettings);
    if(error) break;
    ucvector_push_back(&decoded, 0);

    error = lodepng_add_text(info, key, (char*)decoded.data);

    break;
  }

  myfree(key);
  ucvector_cleanup(&decoded);

  return error;
}

/*international text chunk (iTXt)*/
static unsigned readChunk_iTXt(LodePNGInfo* info, const LodePNGDecompressSettings* zlibsettings,
                               const unsigned char* data, size_t chunkLength)
{
  unsigned error = 0;
  unsigned i;

  unsigned length, begin, compressed;
  char *key = 0, *langtag = 0, *transkey = 0;
  ucvector decoded;
  ucvector_init(&decoded);

  while(!error) /*not really a while loop, only used to break on error*/
  {
    /*Quick check if the chunk length isn't too small. Even without check
    it'd still fail with other error checks below if it's too short. This just gives a different error code.*/
    if(chunkLength < 5) CERROR_BREAK(error, 30); /*iTXt chunk too short*/

    /*read the key*/
    for(length = 0; length < chunkLength && data[length] != 0; length++) ;
    if(length + 3 >= chunkLength) CERROR_BREAK(error, 75); /*no null termination char, corrupt?*/
    if(length < 1 || length > 79) CERROR_BREAK(error, 89); /*keyword too short or long*/

    key = (char*)mymalloc(length + 1);
    if(!key) CERROR_BREAK(error, 83); /*alloc fail*/

    key[length] = 0;
    for(i = 0; i < length; i++) key[i] = data[i];

    /*read the compression method*/
    compressed = data[length + 1];
    if(data[length + 2] != 0) CERROR_BREAK(error, 72); /*the 0 byte indicating compression must be 0*/

    /*even though it's not allowed by the standard, no error is thrown if
    there's no null termination char, if the text is empty for the next 3 texts*/

    /*read the langtag*/
    begin = length + 3;
    length = 0;
    for(i = begin; i < chunkLength && data[i] != 0; i++) length++;

    langtag = (char*)mymalloc(length + 1);
    if(!langtag) CERROR_BREAK(error, 83); /*alloc fail*/

    langtag[length] = 0;
    for(i = 0; i < length; i++) langtag[i] = data[begin + i];

    /*read the transkey*/
    begin += length + 1;
    length = 0;
    for(i = begin; i < chunkLength && data[i] != 0; i++) length++;

    transkey = (char*)mymalloc(length + 1);
    if(!transkey) CERROR_BREAK(error, 83); /*alloc fail*/

    transkey[length] = 0;
    for(i = 0; i < length; i++) transkey[i] = data[begin + i];

    /*read the actual text*/
    begin += length + 1;

    length = chunkLength < begin ? 0 : chunkLength - begin;

    if(compressed)
    {
      /*will fail if zlib error, e.g. if length is too small*/
      error = zlib_decompress(&decoded.data, &decoded.size,
                              (unsigned char*)(&data[begin]),
                              length, zlibsettings);
      if(error) break;
      if(decoded.allocsize < decoded.size) decoded.allocsize = decoded.size;
      ucvector_push_back(&decoded, 0);
    }
    else
    {
      if(!ucvector_resize(&decoded, length + 1)) CERROR_BREAK(error, 83 /*alloc fail*/);

      decoded.data[length] = 0;
      for(i = 0; i < length; i++) decoded.data[i] = data[begin + i];
    }

    error = lodepng_add_itext(info, key, langtag, transkey, (char*)decoded.data);

    break;
  }

  myfree(key);
  myfree(langtag);
  myfree(transkey);
  ucvector_cleanup(&decoded);

  return error;
}

static unsigned readChunk_tIME(LodePNGInfo* info, const unsigned char* data, size_t chunkLength)
{
  if(chunkLength != 7) return 73; /*invalid tIME chunk size*/

  info->time_defined = 1;
  info->time.year = 256 * data[0] + data[+ 1];
  info->time.month = data[2];
  info->time.day = data[3];
  info->time.hour = data[4];
  info->time.minute = data[5];
  info->time.second = data[6];

  return 0; /* OK */
}

static unsigned readChunk_pHYs(LodePNGInfo* info, const unsigned char* data, size_t chunkLength)
{
  if(chunkLength != 9) return 74; /*invalid pHYs chunk size*/

  info->phys_defined = 1;
  info->phys_x = 16777216 * data[0] + 65536 * data[1] + 256 * data[2] + data[3];
  info->phys_y = 16777216 * data[4] + 65536 * data[5] + 256 * data[6] + data[7];
  info->phys_unit = data[8];

  return 0; /* OK */
}
#endif /*LODEPNG_COMPILE_ANCILLARY_CHUNKS*/

/*read a PNG, the result will be in the same color type as the PNG (hence "generic")*/
static void decodeGeneric(unsigned char** out, unsigned* w, unsigned* h,
                          LodePNGState* state,
                          const unsigned char* in, size_t insize)
{
  unsigned char IEND = 0;
  const unsigned char* chunk;
  size_t i;
  ucvector idat; /*the data from idat chunks*/

  /*for unknown chunk order*/
  unsigned unknown = 0;
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
  unsigned critical_pos = 1; /*1 = after IHDR, 2 = after PLTE, 3 = after IDAT*/
#endif /*LODEPNG_COMPILE_ANCILLARY_CHUNKS*/

  /*provide some proper output values if error will happen*/
  *out = 0;

  state->error = lodepng_inspect(w, h, state, in, insize); /*reads header and resets other parameters in state->info_png*/
  if(state->error) return;

  ucvector_init(&idat);
  chunk = &in[33]; /*first byte of the first chunk after the header*/

  /*loop through the chunks, ignoring unknown chunks and stopping at IEND chunk.
  IDAT data is put at the start of the in buffer*/
  while(!IEND && !state->error)
  {
    unsigned chunkLength;
    const unsigned char* data; /*the data in the chunk*/

    /*error: size of the in buffer too small to contain next chunk*/
    if((size_t)((chunk - in) + 12) > insize || chunk < in) CERROR_BREAK(state->error, 30);

    /*length of the data of the chunk, excluding the length bytes, chunk type and CRC bytes*/
    chunkLength = lodepng_chunk_length(chunk);
    /*error: chunk length larger than the max PNG chunk size*/
    if(chunkLength > 2147483647) CERROR_BREAK(state->error, 63);

    if((size_t)((chunk - in) + chunkLength + 12) > insize || (chunk + chunkLength + 12) < in)
    {
      CERROR_BREAK(state->error, 64); /*error: size of the in buffer too small to contain next chunk*/
    }

    data = lodepng_chunk_data_const(chunk);

    /*IDAT chunk, containing compressed image data*/
    if(lodepng_chunk_type_equals(chunk, "IDAT"))
    {
      size_t oldsize = idat.size;
      if(!ucvector_resize(&idat, oldsize + chunkLength)) CERROR_BREAK(state->error, 83 /*alloc fail*/);
      for(i = 0; i < chunkLength; i++) idat.data[oldsize + i] = data[i];
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
      critical_pos = 3;
#endif /*LODEPNG_COMPILE_ANCILLARY_CHUNKS*/
    }
    /*IEND chunk*/
    else if(lodepng_chunk_type_equals(chunk, "IEND"))
    {
      IEND = 1;
    }
    /*palette chunk (PLTE)*/
    else if(lodepng_chunk_type_equals(chunk, "PLTE"))
    {
      state->error = readChunk_PLTE(&state->info_png.color, data, chunkLength);
      if(state->error) break;
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
      critical_pos = 2;
#endif /*LODEPNG_COMPILE_ANCILLARY_CHUNKS*/
    }
    /*palette transparency chunk (tRNS)*/
    else if(lodepng_chunk_type_equals(chunk, "tRNS"))
    {
      state->error = readChunk_tRNS(&state->info_png.color, data, chunkLength);
      if(state->error) break;
    }
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
    /*background color chunk (bKGD)*/
    else if(lodepng_chunk_type_equals(chunk, "bKGD"))
    {
      state->error = readChunk_bKGD(&state->info_png, data, chunkLength);
      if(state->error) break;
    }
    /*text chunk (tEXt)*/
    else if(lodepng_chunk_type_equals(chunk, "tEXt"))
    {
      if(state->decoder.read_text_chunks)
      {
        state->error = readChunk_tEXt(&state->info_png, data, chunkLength);
        if(state->error) break;
      }
    }
    /*compressed text chunk (zTXt)*/
    else if(lodepng_chunk_type_equals(chunk, "zTXt"))
    {
      if(state->decoder.read_text_chunks)
      {
        state->error = readChunk_zTXt(&state->info_png, &state->decoder.zlibsettings, data, chunkLength);
        if(state->error) break;
      }
    }
    /*international text chunk (iTXt)*/
    else if(lodepng_chunk_type_equals(chunk, "iTXt"))
    {
      if(state->decoder.read_text_chunks)
      {
        state->error = readChunk_iTXt(&state->info_png, &state->decoder.zlibsettings, data, chunkLength);
        if(state->error) break;
      }
    }
    else if(lodepng_chunk_type_equals(chunk, "tIME"))
    {
      state->error = readChunk_tIME(&state->info_png, data, chunkLength);
      if(state->error) break;
    }
    else if(lodepng_chunk_type_equals(chunk, "pHYs"))
    {
      state->error = readChunk_pHYs(&state->info_png, data, chunkLength);
      if(state->error) break;
    }
#endif /*LODEPNG_COMPILE_ANCILLARY_CHUNKS*/
    else /*it's not an implemented chunk type, so ignore it: skip over the data*/
    {
      /*error: unknown critical chunk (5th bit of first byte of chunk type is 0)*/
      if(!lodepng_chunk_ancillary(chunk)) CERROR_BREAK(state->error, 69);

      unknown = 1;
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
      if(state->decoder.remember_unknown_chunks)
      {
        state->error = lodepng_chunk_append(&state->info_png.unknown_chunks_data[critical_pos - 1],
                                            &state->info_png.unknown_chunks_size[critical_pos - 1], chunk);
        if(state->error) break;
      }
#endif /*LODEPNG_COMPILE_ANCILLARY_CHUNKS*/
    }

    if(!state->decoder.ignore_crc && !unknown) /*check CRC if wanted, only on known chunk types*/
    {
      if(lodepng_chunk_check_crc(chunk)) CERROR_BREAK(state->error, 57); /*invalid CRC*/
    }

    if(!IEND) chunk = lodepng_chunk_next_const(chunk);
  }

  if(!state->error)
  {
    ucvector scanlines;
    ucvector_init(&scanlines);

    /*maximum final image length is already reserved in the vector's length - this is not really necessary*/
    if(!ucvector_resize(&scanlines, lodepng_get_raw_size(*w, *h, &state->info_png.color) + *h))
    {
      state->error = 83; /*alloc fail*/
    }
    if(!state->error)
    {
      /*decompress with the Zlib decompressor*/
      state->error = zlib_decompress(&scanlines.data, &scanlines.size, idat.data,
                                     idat.size, &state->decoder.zlibsettings);
    }

    if(!state->error)
    {
      ucvector outv;
      ucvector_init(&outv);
      if(!ucvector_resizev(&outv,
          lodepng_get_raw_size(*w, *h, &state->info_png.color), 0)) state->error = 83; /*alloc fail*/
      if(!state->error) state->error = postProcessScanlines(outv.data, scanlines.data, *w, *h, &state->info_png);
      *out = outv.data;
    }
    ucvector_cleanup(&scanlines);
  }

  ucvector_cleanup(&idat);
}

unsigned lodepng_decode(unsigned char** out, unsigned* w, unsigned* h,
                        LodePNGState* state,
                        const unsigned char* in, size_t insize)
{
  *out = 0;
  decodeGeneric(out, w, h, state, in, insize);
  if(state->error) return state->error;
  if(!state->decoder.color_convert || lodepng_color_mode_equal(&state->info_raw, &state->info_png.color))
  {
    /*same color type, no copying or converting of data needed*/
    /*store the info_png color settings on the info_raw so that the info_raw still reflects what colortype
    the raw image has to the end user*/
    if(!state->decoder.color_convert)
    {
      state->error = lodepng_color_mode_copy(&state->info_raw, &state->info_png.color);
      if(state->error) return state->error;
    }
  }
  else
  {
    /*color conversion needed; sort of copy of the data*/
    unsigned char* data = *out;
    size_t outsize;

    /*TODO: check if this works according to the statement in the documentation: "The converter can convert
    from greyscale input color type, to 8-bit greyscale or greyscale with alpha"*/
    if(!(state->info_raw.colortype == LCT_RGB || state->info_raw.colortype == LCT_RGBA)
       && !(state->info_raw.bitdepth == 8))
    {
      return 56; /*unsupported color mode conversion*/
    }

    outsize = lodepng_get_raw_size(*w, *h, &state->info_raw);
    *out = (unsigned char*)mymalloc(outsize);
    if(!(*out))
    {
      state->error = 83; /*alloc fail*/
    }
    else state->error = lodepng_convert(*out, data, &state->info_raw, &state->info_png.color, *w, *h);
    myfree(data);
  }
  return state->error;
}

unsigned lodepng_decode_memory(unsigned char** out, unsigned* w, unsigned* h, const unsigned char* in,
                               size_t insize, LodePNGColorType colortype, unsigned bitdepth)
{
  unsigned error;
  LodePNGState state;
  lodepng_state_init(&state);
  state.info_raw.colortype = colortype;
  state.info_raw.bitdepth = bitdepth;
  error = lodepng_decode(out, w, h, &state, in, insize);
  lodepng_state_cleanup(&state);
  return error;
}

unsigned lodepng_decode32(unsigned char** out, unsigned* w, unsigned* h, const unsigned char* in, size_t insize)
{
  return lodepng_decode_memory(out, w, h, in, insize, LCT_RGBA, 8);
}

unsigned lodepng_decode24(unsigned char** out, unsigned* w, unsigned* h, const unsigned char* in, size_t insize)
{
  return lodepng_decode_memory(out, w, h, in, insize, LCT_RGB, 8);
}

#ifdef LODEPNG_COMPILE_DISK
unsigned lodepng_decode_file(unsigned char** out, unsigned* w, unsigned* h, const char* filename,
                             LodePNGColorType colortype, unsigned bitdepth)
{
  unsigned char* buffer;
  size_t buffersize;
  unsigned error;
  error = lodepng_load_file(&buffer, &buffersize, filename);
  if(!error) error = lodepng_decode_memory(out, w, h, buffer, buffersize, colortype, bitdepth);
  myfree(buffer);
  return error;
}

unsigned lodepng_decode32_file(unsigned char** out, unsigned* w, unsigned* h, const char* filename)
{
  return lodepng_decode_file(out, w, h, filename, LCT_RGBA, 8);
}

unsigned lodepng_decode24_file(unsigned char** out, unsigned* w, unsigned* h, const char* filename)
{
  return lodepng_decode_file(out, w, h, filename, LCT_RGB, 8);
}
#endif /*LODEPNG_COMPILE_DISK*/

void lodepng_decoder_settings_init(LodePNGDecoderSettings* settings)
{
  settings->color_convert = 1;
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
  settings->read_text_chunks = 1;
  settings->remember_unknown_chunks = 0;
#endif /*LODEPNG_COMPILE_ANCILLARY_CHUNKS*/
  settings->ignore_crc = 0;
  lodepng_decompress_settings_init(&settings->zlibsettings);
}

#endif /*LODEPNG_COMPILE_DECODER*/

#if defined(LODEPNG_COMPILE_DECODER) || defined(LODEPNG_COMPILE_ENCODER)

void lodepng_state_init(LodePNGState* state)
{
#ifdef LODEPNG_COMPILE_DECODER
  lodepng_decoder_settings_init(&state->decoder);
#endif /*LODEPNG_COMPILE_DECODER*/
#ifdef LODEPNG_COMPILE_ENCODER
  lodepng_encoder_settings_init(&state->encoder);
#endif /*LODEPNG_COMPILE_ENCODER*/
  lodepng_color_mode_init(&state->info_raw);
  lodepng_info_init(&state->info_png);
  state->error = 1;
}

void lodepng_state_cleanup(LodePNGState* state)
{
  lodepng_color_mode_cleanup(&state->info_raw);
  lodepng_info_cleanup(&state->info_png);
}

void lodepng_state_copy(LodePNGState* dest, const LodePNGState* source)
{
  lodepng_state_cleanup(dest);
  *dest = *source;
  lodepng_color_mode_init(&dest->info_raw);
  lodepng_info_init(&dest->info_png);
  dest->error = lodepng_color_mode_copy(&dest->info_raw, &source->info_raw); if(dest->error) return;
  dest->error = lodepng_info_copy(&dest->info_png, &source->info_png); if(dest->error) return;
}

#endif /* defined(LODEPNG_COMPILE_DECODER) || defined(LODEPNG_COMPILE_ENCODER) */

#ifdef LODEPNG_COMPILE_ENCODER

/* ////////////////////////////////////////////////////////////////////////// */
/* / PNG Encoder                                                            / */
/* ////////////////////////////////////////////////////////////////////////// */

/*chunkName must be string of 4 characters*/
static unsigned addChunk(ucvector* out, const char* chunkName, const unsigned char* data, size_t length)
{
  CERROR_TRY_RETURN(lodepng_chunk_create(&out->data, &out->size, (unsigned)length, chunkName, data));
  out->allocsize = out->size; /*fix the allocsize again*/
  return 0;
}

static void writeSignature(ucvector* out)
{
  /*8 bytes PNG signature, aka the magic bytes*/
  ucvector_push_back(out, 137);
  ucvector_push_back(out, 80);
  ucvector_push_back(out, 78);
  ucvector_push_back(out, 71);
  ucvector_push_back(out, 13);
  ucvector_push_back(out, 10);
  ucvector_push_back(out, 26);
  ucvector_push_back(out, 10);
}

static unsigned addChunk_IHDR(ucvector* out, unsigned w, unsigned h,
                              LodePNGColorType colortype, unsigned bitdepth, unsigned interlace_method)
{
  unsigned error = 0;
  ucvector header;
  ucvector_init(&header);

  lodepng_add32bitInt(&header, w); /*width*/
  lodepng_add32bitInt(&header, h); /*height*/
  ucvector_push_back(&header, (unsigned char)bitdepth); /*bit depth*/
  ucvector_push_back(&header, (unsigned char)colortype); /*color type*/
  ucvector_push_back(&header, 0); /*compression method*/
  ucvector_push_back(&header, 0); /*filter method*/
  ucvector_push_back(&header, interlace_method); /*interlace method*/

  error = addChunk(out, "IHDR", header.data, header.size);
  ucvector_cleanup(&header);

  return error;
}

static unsigned addChunk_PLTE(ucvector* out, const LodePNGColorMode* info)
{
  unsigned error = 0;
  size_t i;
  ucvector PLTE;
  ucvector_init(&PLTE);
  for(i = 0; i < info->palettesize * 4; i++)
  {
    /*add all channels except alpha channel*/
    if(i % 4 != 3) ucvector_push_back(&PLTE, info->palette[i]);
  }
  error = addChunk(out, "PLTE", PLTE.data, PLTE.size);
  ucvector_cleanup(&PLTE);

  return error;
}

static unsigned addChunk_tRNS(ucvector* out, const LodePNGColorMode* info)
{
  unsigned error = 0;
  size_t i;
  ucvector tRNS;
  ucvector_init(&tRNS);
  if(info->colortype == LCT_PALETTE)
  {
    size_t amount = info->palettesize;
    /*the tail of palette values that all have 255 as alpha, does not have to be encoded*/
    for(i = info->palettesize; i > 0; i--)
    {
      if(info->palette[4 * (i - 1) + 3] == 255) amount--;
      else break;
    }
    /*add only alpha channel*/
    for(i = 0; i < amount; i++) ucvector_push_back(&tRNS, info->palette[4 * i + 3]);
  }
  else if(info->colortype == LCT_GREY)
  {
    if(info->key_defined)
    {
      ucvector_push_back(&tRNS, (unsigned char)(info->key_r / 256));
      ucvector_push_back(&tRNS, (unsigned char)(info->key_r % 256));
    }
  }
  else if(info->colortype == LCT_RGB)
  {
    if(info->key_defined)
    {
      ucvector_push_back(&tRNS, (unsigned char)(info->key_r / 256));
      ucvector_push_back(&tRNS, (unsigned char)(info->key_r % 256));
      ucvector_push_back(&tRNS, (unsigned char)(info->key_g / 256));
      ucvector_push_back(&tRNS, (unsigned char)(info->key_g % 256));
      ucvector_push_back(&tRNS, (unsigned char)(info->key_b / 256));
      ucvector_push_back(&tRNS, (unsigned char)(info->key_b % 256));
    }
  }

  error = addChunk(out, "tRNS", tRNS.data, tRNS.size);
  ucvector_cleanup(&tRNS);

  return error;
}

static unsigned addChunk_IDAT(ucvector* out, const unsigned char* data, size_t datasize,
                              LodePNGCompressSettings* zlibsettings)
{
  ucvector zlibdata;
  unsigned error = 0;

  /*compress with the Zlib compressor*/
  ucvector_init(&zlibdata);
  error = zlib_compress(&zlibdata.data, &zlibdata.size, data, datasize, zlibsettings);
  if(!error) error = addChunk(out, "IDAT", zlibdata.data, zlibdata.size);
  ucvector_cleanup(&zlibdata);

  return error;
}

static unsigned addChunk_IEND(ucvector* out)
{
  unsigned error = 0;
  error = addChunk(out, "IEND", 0, 0);
  return error;
}

#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS

static unsigned addChunk_tEXt(ucvector* out, const char* keyword, const char* textstring)
{
  unsigned error = 0;
  size_t i;
  ucvector text;
  ucvector_init(&text);
  for(i = 0; keyword[i] != 0; i++) ucvector_push_back(&text, (unsigned char)keyword[i]);
  if(i < 1 || i > 79) return 89; /*error: invalid keyword size*/
  ucvector_push_back(&text, 0); /*0 termination char*/
  for(i = 0; textstring[i] != 0; i++) ucvector_push_back(&text, (unsigned char)textstring[i]);
  error = addChunk(out, "tEXt", text.data, text.size);
  ucvector_cleanup(&text);

  return error;
}

static unsigned addChunk_zTXt(ucvector* out, const char* keyword, const char* textstring,
                              LodePNGCompressSettings* zlibsettings)
{
  unsigned error = 0;
  ucvector data, compressed;
  size_t i, textsize = strlen(textstring);

  ucvector_init(&data);
  ucvector_init(&compressed);
  for(i = 0; keyword[i] != 0; i++) ucvector_push_back(&data, (unsigned char)keyword[i]);
  if(i < 1 || i > 79) return 89; /*error: invalid keyword size*/
  ucvector_push_back(&data, 0); /*0 termination char*/
  ucvector_push_back(&data, 0); /*compression method: 0*/

  error = zlib_compress(&compressed.data, &compressed.size,
                        (unsigned char*)textstring, textsize, zlibsettings);
  if(!error)
  {
    for(i = 0; i < compressed.size; i++) ucvector_push_back(&data, compressed.data[i]);
    error = addChunk(out, "zTXt", data.data, data.size);
  }

  ucvector_cleanup(&compressed);
  ucvector_cleanup(&data);
  return error;
}

static unsigned addChunk_iTXt(ucvector* out, unsigned compressed, const char* keyword, const char* langtag,
                              const char* transkey, const char* textstring, LodePNGCompressSettings* zlibsettings)
{
  unsigned error = 0;
  ucvector data;
  size_t i, textsize = strlen(textstring);

  ucvector_init(&data);

  for(i = 0; keyword[i] != 0; i++) ucvector_push_back(&data, (unsigned char)keyword[i]);
  if(i < 1 || i > 79) return 89; /*error: invalid keyword size*/
  ucvector_push_back(&data, 0); /*null termination char*/
  ucvector_push_back(&data, compressed ? 1 : 0); /*compression flag*/
  ucvector_push_back(&data, 0); /*compression method*/
  for(i = 0; langtag[i] != 0; i++) ucvector_push_back(&data, (unsigned char)langtag[i]);
  ucvector_push_back(&data, 0); /*null termination char*/
  for(i = 0; transkey[i] != 0; i++) ucvector_push_back(&data, (unsigned char)transkey[i]);
  ucvector_push_back(&data, 0); /*null termination char*/

  if(compressed)
  {
    ucvector compressed_data;
    ucvector_init(&compressed_data);
    error = zlib_compress(&compressed_data.data, &compressed_data.size,
                          (unsigned char*)textstring, textsize, zlibsettings);
    if(!error)
    {
      for(i = 0; i < compressed_data.size; i++) ucvector_push_back(&data, compressed_data.data[i]);
    }
    ucvector_cleanup(&compressed_data);
  }
  else /*not compressed*/
  {
    for(i = 0; textstring[i] != 0; i++) ucvector_push_back(&data, (unsigned char)textstring[i]);
  }

  if(!error) error = addChunk(out, "iTXt", data.data, data.size);
  ucvector_cleanup(&data);
  return error;
}

static unsigned addChunk_bKGD(ucvector* out, const LodePNGInfo* info)
{
  unsigned error = 0;
  ucvector bKGD;
  ucvector_init(&bKGD);
  if(info->color.colortype == LCT_GREY || info->color.colortype == LCT_GREY_ALPHA)
  {
    ucvector_push_back(&bKGD, (unsigned char)(info->background_r / 256));
    ucvector_push_back(&bKGD, (unsigned char)(info->background_r % 256));
  }
  else if(info->color.colortype == LCT_RGB || info->color.colortype == LCT_RGBA)
  {
    ucvector_push_back(&bKGD, (unsigned char)(info->background_r / 256));
    ucvector_push_back(&bKGD, (unsigned char)(info->background_r % 256));
    ucvector_push_back(&bKGD, (unsigned char)(info->background_g / 256));
    ucvector_push_back(&bKGD, (unsigned char)(info->background_g % 256));
    ucvector_push_back(&bKGD, (unsigned char)(info->background_b / 256));
    ucvector_push_back(&bKGD, (unsigned char)(info->background_b % 256));
  }
  else if(info->color.colortype == LCT_PALETTE)
  {
    ucvector_push_back(&bKGD, (unsigned char)(info->background_r % 256)); /*palette index*/
  }

  error = addChunk(out, "bKGD", bKGD.data, bKGD.size);
  ucvector_cleanup(&bKGD);

  return error;
}

static unsigned addChunk_tIME(ucvector* out, const LodePNGTime* time)
{
  unsigned error = 0;
  unsigned char* data = (unsigned char*)mymalloc(7);
  if(!data) return 83; /*alloc fail*/
  data[0] = (unsigned char)(time->year / 256);
  data[1] = (unsigned char)(time->year % 256);
  data[2] = time->month;
  data[3] = time->day;
  data[4] = time->hour;
  data[5] = time->minute;
  data[6] = time->second;
  error = addChunk(out, "tIME", data, 7);
  myfree(data);
  return error;
}

static unsigned addChunk_pHYs(ucvector* out, const LodePNGInfo* info)
{
  unsigned error = 0;
  ucvector data;
  ucvector_init(&data);

  lodepng_add32bitInt(&data, info->phys_x);
  lodepng_add32bitInt(&data, info->phys_y);
  ucvector_push_back(&data, info->phys_unit);

  error = addChunk(out, "pHYs", data.data, data.size);
  ucvector_cleanup(&data);

  return error;
}

#endif /*LODEPNG_COMPILE_ANCILLARY_CHUNKS*/

static void filterScanline(unsigned char* out, const unsigned char* scanline, const unsigned char* prevline,
                           size_t length, size_t bytewidth, unsigned char filterType)
{
  size_t i;
  switch(filterType)
  {
    case 0: /*None*/
      for(i = 0; i < length; i++) out[i] = scanline[i];
      break;
    case 1: /*Sub*/
      if(prevline)
      {
        for(i = 0; i < bytewidth; i++) out[i] = scanline[i];
        for(i = bytewidth; i < length; i++) out[i] = scanline[i] - scanline[i - bytewidth];
      }
      else
      {
        for(i = 0; i < bytewidth; i++) out[i] = scanline[i];
        for(i = bytewidth; i < length; i++) out[i] = scanline[i] - scanline[i - bytewidth];
      }
      break;
    case 2: /*Up*/
      if(prevline)
      {
        for(i = 0; i < length; i++) out[i] = scanline[i] - prevline[i];
      }
      else
      {
        for(i = 0; i < length; i++) out[i] = scanline[i];
      }
      break;
    case 3: /*Average*/
      if(prevline)
      {
        for(i = 0; i < bytewidth; i++) out[i] = scanline[i] - prevline[i] / 2;
        for(i = bytewidth; i < length; i++) out[i] = scanline[i] - ((scanline[i - bytewidth] + prevline[i]) / 2);
      }
      else
      {
        for(i = 0; i < bytewidth; i++) out[i] = scanline[i];
        for(i = bytewidth; i < length; i++) out[i] = scanline[i] - scanline[i - bytewidth] / 2;
      }
      break;
    case 4: /*Paeth*/
      if(prevline)
      {
        /*paethPredictor(0, prevline[i], 0) is always prevline[i]*/
        for(i = 0; i < bytewidth; i++) out[i] = (scanline[i] - prevline[i]);
        for(i = bytewidth; i < length; i++)
        {
          out[i] = (scanline[i] - paethPredictor(scanline[i - bytewidth], prevline[i], prevline[i - bytewidth]));
        }
      }
      else
      {
        for(i = 0; i < bytewidth; i++) out[i] = scanline[i];
        /*paethPredictor(scanline[i - bytewidth], 0, 0) is always scanline[i - bytewidth]*/
        for(i = bytewidth; i < length; i++) out[i] = (scanline[i] - scanline[i - bytewidth]);
      }
      break;
    default: return; /*unexisting filter type given*/
  }
}

/* log2 approximation. A slight bit faster than std::log. */
static float flog2(float f)
{
  float result = 0;
  while(f > 32) { result += 4; f /= 16; }
  while(f > 2) { result++; f /= 2; }
  return result + 1.442695f * (f * f * f / 3 - 3 * f * f / 2 + 3 * f - 1.83333f);
}

static unsigned filter(unsigned char* out, const unsigned char* in, unsigned w, unsigned h,
                       const LodePNGColorMode* info, const LodePNGEncoderSettings* settings)
{
  /*
  For PNG filter method 0
  out must be a buffer with as size: h + (w * h * bpp + 7) / 8, because there are
  the scanlines with 1 extra byte per scanline
  */

  unsigned bpp = lodepng_get_bpp(info);
  /*the width of a scanline in bytes, not including the filter type*/
  size_t linebytes = (w * bpp + 7) / 8;
  /*bytewidth is used for filtering, is 1 when bpp < 8, number of bytes per pixel otherwise*/
  size_t bytewidth = (bpp + 7) / 8;
  const unsigned char* prevline = 0;
  unsigned x, y;
  unsigned error = 0;
  LodePNGFilterStrategy strategy = settings->filter_strategy;

  /*
  There is a heuristic called the minimum sum of absolute differences heuristic, suggested by the PNG standard:
   *  If the image type is Palette, or the bit depth is smaller than 8, then do not filter the image (i.e.
      use fixed filtering, with the filter None).
   * (The other case) If the image type is Grayscale or RGB (with or without Alpha), and the bit depth is
     not smaller than 8, then use adaptive filtering heuristic as follows: independently for each row, apply
     all five filters and select the filter that produces the smallest sum of absolute values per row.
  This heuristic is used if filter strategy is LFS_MINSUM and filter_palette_zero is true.

  If filter_palette_zero is true and filter_strategy is not LFS_MINSUM, the above heuristic is followed,
  but for "the other case", whatever strategy filter_strategy is set to instead of the minimum sum
  heuristic is used.
  */
  if(settings->filter_palette_zero &&
     (info->colortype == LCT_PALETTE || info->bitdepth < 8)) strategy = LFS_ZERO;

  if(bpp == 0) return 31; /*error: invalid color type*/

  if(strategy == LFS_ZERO)
  {
    for(y = 0; y < h; y++)
    {
      size_t outindex = (1 + linebytes) * y; /*the extra filterbyte added to each row*/
      size_t inindex = linebytes * y;
      out[outindex] = 0; /*filter type byte*/
      filterScanline(&out[outindex + 1], &in[inindex], prevline, linebytes, bytewidth, 0);
      prevline = &in[inindex];
    }
  }
  else if(strategy == LFS_MINSUM)
  {
    /*adaptive filtering*/
    size_t sum[5];
    ucvector attempt[5]; /*five filtering attempts, one for each filter type*/
    size_t smallest = 0;
    unsigned type, bestType = 0;

    for(type = 0; type < 5; type++)
    {
      ucvector_init(&attempt[type]);
      if(!ucvector_resize(&attempt[type], linebytes)) return 83; /*alloc fail*/
    }

    if(!error)
    {
      for(y = 0; y < h; y++)
      {
        /*try the 5 filter types*/
        for(type = 0; type < 5; type++)
        {
          filterScanline(attempt[type].data, &in[y * linebytes], prevline, linebytes, bytewidth, type);

          /*calculate the sum of the result*/
          sum[type] = 0;
          if(type == 0)
          {
            for(x = 0; x < linebytes; x++) sum[type] += (unsigned char)(attempt[type].data[x]);
          }
          else
          {
            for(x = 0; x < linebytes; x++)
            {
              /*For differences, each byte should be treated as signed, values above 127 are negative
              (converted to signed char). Filtertype 0 isn't a difference though, so use unsigned there.
              This means filtertype 0 is almost never chosen, but that is justified.*/
              signed char s = (signed char)(attempt[type].data[x]);
              sum[type] += s < 0 ? -s : s;
            }
          }

          /*check if this is smallest sum (or if type == 0 it's the first case so always store the values)*/
          if(type == 0 || sum[type] < smallest)
          {
            bestType = type;
            smallest = sum[type];
          }
        }

        prevline = &in[y * linebytes];

        /*now fill the out values*/
        out[y * (linebytes + 1)] = bestType; /*the first byte of a scanline will be the filter type*/
        for(x = 0; x < linebytes; x++) out[y * (linebytes + 1) + 1 + x] = attempt[bestType].data[x];
      }
    }

    for(type = 0; type < 5; type++) ucvector_cleanup(&attempt[type]);
  }
  else if(strategy == LFS_ENTROPY)
  {
    float sum[5];
    ucvector attempt[5]; /*five filtering attempts, one for each filter type*/
    float smallest = 0;
    unsigned type, bestType = 0;
    unsigned count[256];

    for(type = 0; type < 5; type++)
    {
      ucvector_init(&attempt[type]);
      if(!ucvector_resize(&attempt[type], linebytes)) return 83; /*alloc fail*/
    }

    for(y = 0; y < h; y++)
    {
      /*try the 5 filter types*/
      for(type = 0; type < 5; type++)
      {
        filterScanline(attempt[type].data, &in[y * linebytes], prevline, linebytes, bytewidth, type);
        for(x = 0; x < 256; x++) count[x] = 0;
        for(x = 0; x < linebytes; x++) count[attempt[type].data[x]]++;
        count[type]++; /*the filter type itself is part of the scanline*/
        sum[type] = 0;
        for(x = 0; x < 256; x++)
        {
          float p = count[x] / (float)(linebytes + 1);
          sum[type] += count[x] == 0 ? 0 : flog2(1 / p) * p;
        }
        /*check if this is smallest sum (or if type == 0 it's the first case so always store the values)*/
        if(type == 0 || sum[type] < smallest)
        {
          bestType = type;
          smallest = sum[type];
        }
      }

      prevline = &in[y * linebytes];

      /*now fill the out values*/
      out[y * (linebytes + 1)] = bestType; /*the first byte of a scanline will be the filter type*/
      for(x = 0; x < linebytes; x++) out[y * (linebytes + 1) + 1 + x] = attempt[bestType].data[x];
    }

    for(type = 0; type < 5; type++) ucvector_cleanup(&attempt[type]);
  }
  else if(strategy == LFS_PREDEFINED)
  {
    for(y = 0; y < h; y++)
    {
      size_t outindex = (1 + linebytes) * y; /*the extra filterbyte added to each row*/
      size_t inindex = linebytes * y;
      unsigned type = settings->predefined_filters[y];
      out[outindex] = type; /*filter type byte*/
      filterScanline(&out[outindex + 1], &in[inindex], prevline, linebytes, bytewidth, type);
      prevline = &in[inindex];
    }
  }
  else if(strategy == LFS_BRUTE_FORCE)
  {
    /*brute force filter chooser.
    deflate the scanline after every filter attempt to see which one deflates best.
    This is very slow and gives only slightly smaller, sometimes even larger, result*/
    size_t size[5];
    ucvector attempt[5]; /*five filtering attempts, one for each filter type*/
    size_t smallest = 0;
    unsigned type = 0, bestType = 0;
    unsigned char* dummy;
    LodePNGCompressSettings zlibsettings = settings->zlibsettings;
    /*use fixed tree on the attempts so that the tree is not adapted to the filtertype on purpose,
    to simulate the true case where the tree is the same for the whole image. Sometimes it gives
    better result with dynamic tree anyway. Using the fixed tree sometimes gives worse, but in rare
    cases better compression. It does make this a bit less slow, so it's worth doing this.*/
    zlibsettings.btype = 1;
    /*a custom encoder likely doesn't read the btype setting and is optimized for complete PNG
    images only, so disable it*/
    zlibsettings.custom_zlib = 0;
    zlibsettings.custom_deflate = 0;
    for(type = 0; type < 5; type++)
    {
      ucvector_init(&attempt[type]);
      ucvector_resize(&attempt[type], linebytes); /*todo: give error if resize failed*/
    }
    for(y = 0; y < h; y++) /*try the 5 filter types*/
    {
      for(type = 0; type < 5; type++)
      {
        unsigned testsize = attempt[type].size;
        /*if(testsize > 8) testsize /= 8;*/ /*it already works good enough by testing a part of the row*/

        filterScanline(attempt[type].data, &in[y * linebytes], prevline, linebytes, bytewidth, type);
        size[type] = 0;
        dummy = 0;
        zlib_compress(&dummy, &size[type], attempt[type].data, testsize, &zlibsettings);
        myfree(dummy);
        /*check if this is smallest size (or if type == 0 it's the first case so always store the values)*/
        if(type == 0 || size[type] < smallest)
        {
          bestType = type;
          smallest = size[type];
        }
      }
      prevline = &in[y * linebytes];
      out[y * (linebytes + 1)] = bestType; /*the first byte of a scanline will be the filter type*/
      for(x = 0; x < linebytes; x++) out[y * (linebytes + 1) + 1 + x] = attempt[bestType].data[x];
    }
    for(type = 0; type < 5; type++) ucvector_cleanup(&attempt[type]);
  }
  else return 88; /* unknown filter strategy */

  return error;
}

static void addPaddingBits(unsigned char* out, const unsigned char* in,
                           size_t olinebits, size_t ilinebits, unsigned h)
{
  /*The opposite of the removePaddingBits function
  olinebits must be >= ilinebits*/
  unsigned y;
  size_t diff = olinebits - ilinebits;
  size_t obp = 0, ibp = 0; /*bit pointers*/
  for(y = 0; y < h; y++)
  {
    size_t x;
    for(x = 0; x < ilinebits; x++)
    {
      unsigned char bit = readBitFromReversedStream(&ibp, in);
      setBitOfReversedStream(&obp, out, bit);
    }
    /*obp += diff; --> no, fill in some value in the padding bits too, to avoid
    "Use of uninitialised value of size ###" warning from valgrind*/
    for(x = 0; x < diff; x++) setBitOfReversedStream(&obp, out, 0);
  }
}

/*
in: non-interlaced image with size w*h
out: the same pixels, but re-ordered according to PNG's Adam7 interlacing, with
 no padding bits between scanlines, but between reduced images so that each
 reduced image starts at a byte.
bpp: bits per pixel
there are no padding bits, not between scanlines, not between reduced images
in has the following size in bits: w * h * bpp.
out is possibly bigger due to padding bits between reduced images
NOTE: comments about padding bits are only relevant if bpp < 8
*/
static void Adam7_interlace(unsigned char* out, const unsigned char* in, unsigned w, unsigned h, unsigned bpp)
{
  unsigned passw[7], passh[7];
  size_t filter_passstart[8], padded_passstart[8], passstart[8];
  unsigned i;

  Adam7_getpassvalues(passw, passh, filter_passstart, padded_passstart, passstart, w, h, bpp);

  if(bpp >= 8)
  {
    for(i = 0; i < 7; i++)
    {
      unsigned x, y, b;
      size_t bytewidth = bpp / 8;
      for(y = 0; y < passh[i]; y++)
      for(x = 0; x < passw[i]; x++)
      {
        size_t pixelinstart = ((ADAM7_IY[i] + y * ADAM7_DY[i]) * w + ADAM7_IX[i] + x * ADAM7_DX[i]) * bytewidth;
        size_t pixeloutstart = passstart[i] + (y * passw[i] + x) * bytewidth;
        for(b = 0; b < bytewidth; b++)
        {
          out[pixeloutstart + b] = in[pixelinstart + b];
        }
      }
    }
  }
  else /*bpp < 8: Adam7 with pixels < 8 bit is a bit trickier: with bit pointers*/
  {
    for(i = 0; i < 7; i++)
    {
      unsigned x, y, b;
      unsigned ilinebits = bpp * passw[i];
      unsigned olinebits = bpp * w;
      size_t obp, ibp; /*bit pointers (for out and in buffer)*/
      for(y = 0; y < passh[i]; y++)
      for(x = 0; x < passw[i]; x++)
      {
        ibp = (ADAM7_IY[i] + y * ADAM7_DY[i]) * olinebits + (ADAM7_IX[i] + x * ADAM7_DX[i]) * bpp;
        obp = (8 * passstart[i]) + (y * ilinebits + x * bpp);
        for(b = 0; b < bpp; b++)
        {
          unsigned char bit = readBitFromReversedStream(&ibp, in);
          setBitOfReversedStream(&obp, out, bit);
        }
      }
    }
  }
}

/*out must be buffer big enough to contain uncompressed IDAT chunk data, and in must contain the full image.
return value is error**/
static unsigned preProcessScanlines(unsigned char** out, size_t* outsize, const unsigned char* in,
                                    unsigned w, unsigned h,
                                    const LodePNGInfo* info_png, const LodePNGEncoderSettings* settings)
{
  /*
  This function converts the pure 2D image with the PNG's colortype, into filtered-padded-interlaced data. Steps:
  *) if no Adam7: 1) add padding bits (= posible extra bits per scanline if bpp < 8) 2) filter
  *) if adam7: 1) Adam7_interlace 2) 7x add padding bits 3) 7x filter
  */
  unsigned bpp = lodepng_get_bpp(&info_png->color);
  unsigned error = 0;

  if(info_png->interlace_method == 0)
  {
    *outsize = h + (h * ((w * bpp + 7) / 8)); /*image size plus an extra byte per scanline + possible padding bits*/
    *out = (unsigned char*)mymalloc(*outsize);
    if(!(*out) && (*outsize)) error = 83; /*alloc fail*/

    if(!error)
    {
      /*non multiple of 8 bits per scanline, padding bits needed per scanline*/
      if(bpp < 8 && w * bpp != ((w * bpp + 7) / 8) * 8)
      {
        unsigned char* padded = (unsigned char*)mymalloc(h * ((w * bpp + 7) / 8));
        if(!padded) error = 83; /*alloc fail*/
        if(!error)
        {
          addPaddingBits(padded, in, ((w * bpp + 7) / 8) * 8, w * bpp, h);
          error = filter(*out, padded, w, h, &info_png->color, settings);
        }
        myfree(padded);
      }
      else
      {
        /*we can immediatly filter into the out buffer, no other steps needed*/
        error = filter(*out, in, w, h, &info_png->color, settings);
      }
    }
  }
  else /*interlace_method is 1 (Adam7)*/
  {
    unsigned passw[7], passh[7];
    size_t filter_passstart[8], padded_passstart[8], passstart[8];
    unsigned char* adam7;

    Adam7_getpassvalues(passw, passh, filter_passstart, padded_passstart, passstart, w, h, bpp);

    *outsize = filter_passstart[7]; /*image size plus an extra byte per scanline + possible padding bits*/
    *out = (unsigned char*)mymalloc(*outsize);
    if(!(*out)) error = 83; /*alloc fail*/

    adam7 = (unsigned char*)mymalloc(passstart[7]);
    if(!adam7 && passstart[7]) error = 83; /*alloc fail*/

    if(!error)
    {
      unsigned i;

      Adam7_interlace(adam7, in, w, h, bpp);
      for(i = 0; i < 7; i++)
      {
        if(bpp < 8)
        {
          unsigned char* padded = (unsigned char*)mymalloc(padded_passstart[i + 1] - padded_passstart[i]);
          if(!padded) ERROR_BREAK(83); /*alloc fail*/
          addPaddingBits(padded, &adam7[passstart[i]],
                         ((passw[i] * bpp + 7) / 8) * 8, passw[i] * bpp, passh[i]);
          error = filter(&(*out)[filter_passstart[i]], padded,
                         passw[i], passh[i], &info_png->color, settings);
          myfree(padded);
        }
        else
        {
          error = filter(&(*out)[filter_passstart[i]], &adam7[padded_passstart[i]],
                         passw[i], passh[i], &info_png->color, settings);
        }

        if(error) break;
      }
    }

    myfree(adam7);
  }

  return error;
}

/*
palette must have 4 * palettesize bytes allocated, and given in format RGBARGBARGBARGBA...
returns 0 if the palette is opaque,
returns 1 if the palette has a single color with alpha 0 ==> color key
returns 2 if the palette is semi-translucent.
*/
static unsigned getPaletteTranslucency(const unsigned char* palette, size_t palettesize)
{
  size_t i, key = 0;
  unsigned r = 0, g = 0, b = 0; /*the value of the color with alpha 0, so long as color keying is possible*/
  for(i = 0; i < palettesize; i++)
  {
    if(!key && palette[4 * i + 3] == 0)
    {
      r = palette[4 * i + 0]; g = palette[4 * i + 1]; b = palette[4 * i + 2];
      key = 1;
      i = (size_t)(-1); /*restart from beginning, to detect earlier opaque colors with key's value*/
    }
    else if(palette[4 * i + 3] != 255) return 2;
    /*when key, no opaque RGB may have key's RGB*/
    else if(key && r == palette[i * 4 + 0] && g == palette[i * 4 + 1] && b == palette[i * 4 + 2]) return 2;
  }
  return key;
}

#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
static unsigned addUnknownChunks(ucvector* out, unsigned char* data, size_t datasize)
{
  unsigned char* inchunk = data;
  while((size_t)(inchunk - data) < datasize)
  {
    CERROR_TRY_RETURN(lodepng_chunk_append(&out->data, &out->size, inchunk));
    out->allocsize = out->size; /*fix the allocsize again*/
    inchunk = lodepng_chunk_next(inchunk);
  }
  return 0;
}
#endif /*LODEPNG_COMPILE_ANCILLARY_CHUNKS*/

unsigned lodepng_encode(unsigned char** out, size_t* outsize,
                        const unsigned char* image, unsigned w, unsigned h,
                        LodePNGState* state)
{
  LodePNGInfo info;
  ucvector outv;
  unsigned char* data = 0; /*uncompressed version of the IDAT chunk data*/
  size_t datasize = 0;

  /*provide some proper output values if error will happen*/
  *out = 0;
  *outsize = 0;
  state->error = 0;

  lodepng_info_init(&info);
  lodepng_info_copy(&info, &state->info_png);

  if((info.color.colortype == LCT_PALETTE || state->encoder.force_palette)
      && (info.color.palettesize == 0 || info.color.palettesize > 256))
  {
    state->error = 68; /*invalid palette size, it is only allowed to be 1-256*/
    return state->error;
  }

  if(state->encoder.auto_convert != LAC_NO)
  {
    state->error = doAutoChooseColor(&info.color, image, w, h, &state->info_raw,
                                     state->encoder.auto_convert);
  }
  if(state->error) return state->error;

  if(state->encoder.zlibsettings.windowsize > 32768)
  {
    CERROR_RETURN_ERROR(state->error, 60); /*error: windowsize larger than allowed*/
  }
  if(state->encoder.zlibsettings.btype > 2)
  {
    CERROR_RETURN_ERROR(state->error, 61); /*error: unexisting btype*/
  }
  if(state->info_png.interlace_method > 1)
  {
    CERROR_RETURN_ERROR(state->error, 71); /*error: unexisting interlace mode*/
  }

  state->error = checkColorValidity(info.color.colortype, info.color.bitdepth);
  if(state->error) return state->error; /*error: unexisting color type given*/
  state->error = checkColorValidity(state->info_raw.colortype, state->info_raw.bitdepth);
  if(state->error) return state->error; /*error: unexisting color type given*/

  if(!lodepng_color_mode_equal(&state->info_raw, &info.color))
  {
    unsigned char* converted;
    size_t size = (w * h * lodepng_get_bpp(&info.color) + 7) / 8;

    converted = (unsigned char*)mymalloc(size);
    if(!converted && size) state->error = 83; /*alloc fail*/
    if(!state->error)
    {
      state->error = lodepng_convert(converted, image, &info.color, &state->info_raw, w, h);
    }
    if(!state->error) preProcessScanlines(&data, &datasize, converted, w, h, &info, &state->encoder);
    myfree(converted);
  }
  else preProcessScanlines(&data, &datasize, image, w, h, &info, &state->encoder);

  ucvector_init(&outv);
  while(!state->error) /*while only executed once, to break on error*/
  {
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
    size_t i;
#endif /*LODEPNG_COMPILE_ANCILLARY_CHUNKS*/
    /*write signature and chunks*/
    writeSignature(&outv);
    /*IHDR*/
    addChunk_IHDR(&outv, w, h, info.color.colortype, info.color.bitdepth, info.interlace_method);
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
    /*unknown chunks between IHDR and PLTE*/
    if(info.unknown_chunks_data[0])
    {
      state->error = addUnknownChunks(&outv, info.unknown_chunks_data[0], info.unknown_chunks_size[0]);
      if(state->error) break;
    }
#endif /*LODEPNG_COMPILE_ANCILLARY_CHUNKS*/
    /*PLTE*/
    if(info.color.colortype == LCT_PALETTE)
    {
      addChunk_PLTE(&outv, &info.color);
    }
    if(state->encoder.force_palette && (info.color.colortype == LCT_RGB || info.color.colortype == LCT_RGBA))
    {
      addChunk_PLTE(&outv, &info.color);
    }
    /*tRNS*/
    if(info.color.colortype == LCT_PALETTE && getPaletteTranslucency(info.color.palette, info.color.palettesize) != 0)
    {
      addChunk_tRNS(&outv, &info.color);
    }
    if((info.color.colortype == LCT_GREY || info.color.colortype == LCT_RGB) && info.color.key_defined)
    {
      addChunk_tRNS(&outv, &info.color);
    }
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
    /*bKGD (must come between PLTE and the IDAt chunks*/
    if(info.background_defined) addChunk_bKGD(&outv, &info);
    /*pHYs (must come before the IDAT chunks)*/
    if(info.phys_defined) addChunk_pHYs(&outv, &info);

    /*unknown chunks between PLTE and IDAT*/
    if(info.unknown_chunks_data[1])
    {
      state->error = addUnknownChunks(&outv, info.unknown_chunks_data[1], info.unknown_chunks_size[1]);
      if(state->error) break;
    }
#endif /*LODEPNG_COMPILE_ANCILLARY_CHUNKS*/
    /*IDAT (multiple IDAT chunks must be consecutive)*/
    state->error = addChunk_IDAT(&outv, data, datasize, &state->encoder.zlibsettings);
    if(state->error) break;
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
    /*tIME*/
    if(info.time_defined) addChunk_tIME(&outv, &info.time);
    /*tEXt and/or zTXt*/
    for(i = 0; i < info.text_num; i++)
    {
      if(strlen(info.text_keys[i]) > 79)
      {
        state->error = 66; /*text chunk too large*/
        break;
      }
      if(strlen(info.text_keys[i]) < 1)
      {
        state->error = 67; /*text chunk too small*/
        break;
      }
      if(state->encoder.text_compression)
        addChunk_zTXt(&outv, info.text_keys[i], info.text_strings[i], &state->encoder.zlibsettings);
      else
        addChunk_tEXt(&outv, info.text_keys[i], info.text_strings[i]);
    }
    /*LodePNG version id in text chunk*/
    if(state->encoder.add_id)
    {
      unsigned alread_added_id_text = 0;
      for(i = 0; i < info.text_num; i++)
      {
        if(!strcmp(info.text_keys[i], "LodePNG"))
        {
          alread_added_id_text = 1;
          break;
        }
      }
      if(alread_added_id_text == 0)
        addChunk_tEXt(&outv, "LodePNG", VERSION_STRING); /*it's shorter as tEXt than as zTXt chunk*/
    }
    /*iTXt*/
    for(i = 0; i < info.itext_num; i++)
    {
      if(strlen(info.itext_keys[i]) > 79)
      {
        state->error = 66; /*text chunk too large*/
        break;
      }
      if(strlen(info.itext_keys[i]) < 1)
      {
        state->error = 67; /*text chunk too small*/
        break;
      }
      addChunk_iTXt(&outv, state->encoder.text_compression,
                    info.itext_keys[i], info.itext_langtags[i], info.itext_transkeys[i], info.itext_strings[i],
                    &state->encoder.zlibsettings);
    }

    /*unknown chunks between IDAT and IEND*/
    if(info.unknown_chunks_data[2])
    {
      state->error = addUnknownChunks(&outv, info.unknown_chunks_data[2], info.unknown_chunks_size[2]);
      if(state->error) break;
    }
#endif /*LODEPNG_COMPILE_ANCILLARY_CHUNKS*/
    /*IEND*/
    addChunk_IEND(&outv);

    break; /*this isn't really a while loop; no error happened so break out now!*/
  }

  lodepng_info_cleanup(&info);
  myfree(data);
  /*instead of cleaning the vector up, give it to the output*/
  *out = outv.data;
  *outsize = outv.size;

  return state->error;
}

unsigned lodepng_encode_memory(unsigned char** out, size_t* outsize, const unsigned char* image,
                               unsigned w, unsigned h, LodePNGColorType colortype, unsigned bitdepth)
{
  unsigned error;
  LodePNGState state;
  lodepng_state_init(&state);
  state.info_raw.colortype = colortype;
  state.info_raw.bitdepth = bitdepth;
  state.info_png.color.colortype = colortype;
  state.info_png.color.bitdepth = bitdepth;
  lodepng_encode(out, outsize, image, w, h, &state);
  error = state.error;
  lodepng_state_cleanup(&state);
  return error;
}

unsigned lodepng_encode32(unsigned char** out, size_t* outsize, const unsigned char* image, unsigned w, unsigned h)
{
  return lodepng_encode_memory(out, outsize, image, w, h, LCT_RGBA, 8);
}

unsigned lodepng_encode24(unsigned char** out, size_t* outsize, const unsigned char* image, unsigned w, unsigned h)
{
  return lodepng_encode_memory(out, outsize, image, w, h, LCT_RGB, 8);
}

#ifdef LODEPNG_COMPILE_DISK
unsigned lodepng_encode_file(const char* filename, const unsigned char* image, unsigned w, unsigned h,
                             LodePNGColorType colortype, unsigned bitdepth)
{
  unsigned char* buffer;
  size_t buffersize;
  unsigned error = lodepng_encode_memory(&buffer, &buffersize, image, w, h, colortype, bitdepth);
  if(!error) error = lodepng_save_file(buffer, buffersize, filename);
  myfree(buffer);
  return error;
}

unsigned lodepng_encode32_file(const char* filename, const unsigned char* image, unsigned w, unsigned h)
{
  return lodepng_encode_file(filename, image, w, h, LCT_RGBA, 8);
}

unsigned lodepng_encode24_file(const char* filename, const unsigned char* image, unsigned w, unsigned h)
{
  return lodepng_encode_file(filename, image, w, h, LCT_RGB, 8);
}
#endif /*LODEPNG_COMPILE_DISK*/

void lodepng_encoder_settings_init(LodePNGEncoderSettings* settings)
{
  lodepng_compress_settings_init(&settings->zlibsettings);
  settings->filter_palette_zero = 1;
  settings->filter_strategy = LFS_MINSUM;
  settings->auto_convert = LAC_AUTO;
  settings->force_palette = 0;
  settings->predefined_filters = 0;
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
  settings->add_id = 0;
  settings->text_compression = 1;
#endif /*LODEPNG_COMPILE_ANCILLARY_CHUNKS*/
}

#endif /*LODEPNG_COMPILE_ENCODER*/
#endif /*LODEPNG_COMPILE_PNG*/

#ifdef LODEPNG_COMPILE_ERROR_TEXT
/*
This returns the description of a numerical error code in English. This is also
the documentation of all the error codes.
*/
const char* lodepng_error_text(unsigned code)
{
  switch(code)
  {
    case 0: return "no error, everything went ok";
    case 1: return "nothing done yet"; /*the Encoder/Decoder has done nothing yet, error checking makes no sense yet*/
    case 10: return "end of input memory reached without huffman end code"; /*while huffman decoding*/
    case 11: return "error in code tree made it jump outside of huffman tree"; /*while huffman decoding*/
    case 13: return "problem while processing dynamic deflate block";
    case 14: return "problem while processing dynamic deflate block";
    case 15: return "problem while processing dynamic deflate block";
    case 16: return "unexisting code while processing dynamic deflate block";
    case 17: return "end of out buffer memory reached while inflating";
    case 18: return "invalid distance code while inflating";
    case 19: return "end of out buffer memory reached while inflating";
    case 20: return "invalid deflate block BTYPE encountered while decoding";
    case 21: return "NLEN is not ones complement of LEN in a deflate block";
     /*end of out buffer memory reached while inflating:
     This can happen if the inflated deflate data is longer than the amount of bytes required to fill up
     all the pixels of the image, given the color depth and image dimensions. Something that doesn't
     happen in a normal, well encoded, PNG image.*/
    case 22: return "end of out buffer memory reached while inflating";
    case 23: return "end of in buffer memory reached while inflating";
    case 24: return "invalid FCHECK in zlib header";
    case 25: return "invalid compression method in zlib header";
    case 26: return "FDICT encountered in zlib header while it's not used for PNG";
    case 27: return "PNG file is smaller than a PNG header";
    /*Checks the magic file header, the first 8 bytes of the PNG file*/
    case 28: return "incorrect PNG signature, it's no PNG or corrupted";
    case 29: return "first chunk is not the header chunk";
    case 30: return "chunk length too large, chunk broken off at end of file";
    case 31: return "illegal PNG color type or bpp";
    case 32: return "illegal PNG compression method";
    case 33: return "illegal PNG filter method";
    case 34: return "illegal PNG interlace method";
    case 35: return "chunk length of a chunk is too large or the chunk too small";
    case 36: return "illegal PNG filter type encountered";
    case 37: return "illegal bit depth for this color type given";
    case 38: return "the palette is too big"; /*more than 256 colors*/
    case 39: return "more palette alpha values given in tRNS chunk than there are colors in the palette";
    case 40: return "tRNS chunk has wrong size for greyscale image";
    case 41: return "tRNS chunk has wrong size for RGB image";
    case 42: return "tRNS chunk appeared while it was not allowed for this color type";
    case 43: return "bKGD chunk has wrong size for palette image";
    case 44: return "bKGD chunk has wrong size for greyscale image";
    case 45: return "bKGD chunk has wrong size for RGB image";
    /*Is the palette too small?*/
    case 46: return "a value in indexed image is larger than the palette size (bitdepth = 8)";
    /*Is the palette too small?*/
    case 47: return "a value in indexed image is larger than the palette size (bitdepth < 8)";
    /*the input data is empty, maybe a PNG file doesn't exist or is in the wrong path*/
    case 48: return "empty input or file doesn't exist";
    case 49: return "jumped past memory while generating dynamic huffman tree";
    case 50: return "jumped past memory while generating dynamic huffman tree";
    case 51: return "jumped past memory while inflating huffman block";
    case 52: return "jumped past memory while inflating";
    case 53: return "size of zlib data too small";
    case 54: return "repeat symbol in tree while there was no value symbol yet";
    /*jumped past tree while generating huffman tree, this could be when the
    tree will have more leaves than symbols after generating it out of the
    given lenghts. They call this an oversubscribed dynamic bit lengths tree in zlib.*/
    case 55: return "jumped past tree while generating huffman tree";
    case 56: return "given output image colortype or bitdepth not supported for color conversion";
    case 57: return "invalid CRC encountered (checking CRC can be disabled)";
    case 58: return "invalid ADLER32 encountered (checking ADLER32 can be disabled)";
    case 59: return "requested color conversion not supported";
    case 60: return "invalid window size given in the settings of the encoder (must be 0-32768)";
    case 61: return "invalid BTYPE given in the settings of the encoder (only 0, 1 and 2 are allowed)";
    /*LodePNG leaves the choice of RGB to greyscale conversion formula to the user.*/
    case 62: return "conversion from color to greyscale not supported";
    case 63: return "length of a chunk too long, max allowed for PNG is 2147483647 bytes per chunk"; /*(2^31-1)*/
    /*this would result in the inability of a deflated block to ever contain an end code. It must be at least 1.*/
    case 64: return "the length of the END symbol 256 in the Huffman tree is 0";
    case 66: return "the length of a text chunk keyword given to the encoder is longer than the maximum of 79 bytes";
    case 67: return "the length of a text chunk keyword given to the encoder is smaller than the minimum of 1 byte";
    case 68: return "tried to encode a PLTE chunk with a palette that has less than 1 or more than 256 colors";
    case 69: return "unknown chunk type with 'critical' flag encountered by the decoder";
    case 71: return "unexisting interlace mode given to encoder (must be 0 or 1)";
    case 72: return "while decoding, unexisting compression method encountering in zTXt or iTXt chunk (it must be 0)";
    case 73: return "invalid tIME chunk size";
    case 74: return "invalid pHYs chunk size";
    /*length could be wrong, or data chopped off*/
    case 75: return "no null termination char found while decoding text chunk";
    case 76: return "iTXt chunk too short to contain required bytes";
    case 77: return "integer overflow in buffer size";
    case 78: return "failed to open file for reading"; /*file doesn't exist or couldn't be opened for reading*/
    case 79: return "failed to open file for writing";
    case 80: return "tried creating a tree of 0 symbols";
    case 81: return "lazy matching at pos 0 is impossible";
    case 82: return "color conversion to palette requested while a color isn't in palette";
    case 83: return "memory allocation failed";
    case 84: return "given image too small to contain all pixels to be encoded";
    case 85: return "internal color conversion bug";
    case 86: return "impossible offset in lz77 encoding (internal bug)";
    case 87: return "must provide custom zlib function pointer if LODEPNG_COMPILE_ZLIB is not defined";
    case 88: return "invalid filter strategy given for LodePNGEncoderSettings.filter_strategy";
    case 89: return "text chunk keyword too short or long: must have size 1-79";
  }
  return "unknown error code";
}
#endif /*LODEPNG_COMPILE_ERROR_TEXT*/

/* ////////////////////////////////////////////////////////////////////////// */
/* ////////////////////////////////////////////////////////////////////////// */
/* // C++ Wrapper                                                          // */
/* ////////////////////////////////////////////////////////////////////////// */
/* ////////////////////////////////////////////////////////////////////////// */


#ifdef LODEPNG_COMPILE_CPP
namespace lodepng
{

#ifdef LODEPNG_COMPILE_DISK
void load_file(std::vector<unsigned char>& buffer, const std::string& filename)
{
  std::ifstream file(filename.c_str(), std::ios::in|std::ios::binary|std::ios::ate);

  /*get filesize*/
  std::streamsize size = 0;
  if(file.seekg(0, std::ios::end).good()) size = file.tellg();
  if(file.seekg(0, std::ios::beg).good()) size -= file.tellg();

  /*read contents of the file into the vector*/
  buffer.resize(size_t(size));
  if(size > 0) file.read((char*)(&buffer[0]), size);
}

/*write given buffer to the file, overwriting the file, it doesn't append to it.*/
void save_file(const std::vector<unsigned char>& buffer, const std::string& filename)
{
  std::ofstream file(filename.c_str(), std::ios::out|std::ios::binary);
  file.write(buffer.empty() ? 0 : (char*)&buffer[0], std::streamsize(buffer.size()));
}
#endif //LODEPNG_COMPILE_DISK

#ifdef LODEPNG_COMPILE_ZLIB
#ifdef LODEPNG_COMPILE_DECODER
unsigned decompress(std::vector<unsigned char>& out, const unsigned char* in, size_t insize,
                    const LodePNGDecompressSettings& settings)
{
  unsigned char* buffer = 0;
  size_t buffersize = 0;
  unsigned error = zlib_decompress(&buffer, &buffersize, in, insize, &settings);
  if(buffer)
  {
    out.insert(out.end(), &buffer[0], &buffer[buffersize]);
    myfree(buffer);
  }
  return error;
}

unsigned decompress(std::vector<unsigned char>& out, const std::vector<unsigned char>& in,
                    const LodePNGDecompressSettings& settings)
{
  return decompress(out, in.empty() ? 0 : &in[0], in.size(), settings);
}
#endif //LODEPNG_COMPILE_DECODER

#ifdef LODEPNG_COMPILE_ENCODER
unsigned compress(std::vector<unsigned char>& out, const unsigned char* in, size_t insize,
                  const LodePNGCompressSettings& settings)
{
  unsigned char* buffer = 0;
  size_t buffersize = 0;
  unsigned error = zlib_compress(&buffer, &buffersize, in, insize, &settings);
  if(buffer)
  {
    out.insert(out.end(), &buffer[0], &buffer[buffersize]);
    myfree(buffer);
  }
  return error;
}

unsigned compress(std::vector<unsigned char>& out, const std::vector<unsigned char>& in,
                  const LodePNGCompressSettings& settings)
{
  return compress(out, in.empty() ? 0 : &in[0], in.size(), settings);
}
#endif //LODEPNG_COMPILE_ENCODER
#endif //LODEPNG_COMPILE_ZLIB


#ifdef LODEPNG_COMPILE_PNG

State::State()
{
  lodepng_state_init(this);
}

State::State(const State& other)
{
  lodepng_state_init(this);
  lodepng_state_copy(this, &other);
}

State::~State()
{
  lodepng_state_cleanup(this);
}

State& State::operator=(const State& other)
{
  lodepng_state_copy(this, &other);
  return *this;
}

#ifdef LODEPNG_COMPILE_DECODER

unsigned decode(std::vector<unsigned char>& out, unsigned& w, unsigned& h, const unsigned char* in,
                size_t insize, LodePNGColorType colortype, unsigned bitdepth)
{
  unsigned char* buffer;
  unsigned error = lodepng_decode_memory(&buffer, &w, &h, in, insize, colortype, bitdepth);
  if(buffer && !error)
  {
    State state;
    state.info_raw.colortype = colortype;
    state.info_raw.bitdepth = bitdepth;
    size_t buffersize = lodepng_get_raw_size(w, h, &state.info_raw);
    out.insert(out.end(), &buffer[0], &buffer[buffersize]);
    myfree(buffer);
  }
  return error;
}

unsigned decode(std::vector<unsigned char>& out, unsigned& w, unsigned& h,
                const std::vector<unsigned char>& in, LodePNGColorType colortype, unsigned bitdepth)
{
  return decode(out, w, h, in.empty() ? 0 : &in[0], (unsigned)in.size(), colortype, bitdepth);
}

unsigned decode(std::vector<unsigned char>& out, unsigned& w, unsigned& h,
                State& state,
                const unsigned char* in, size_t insize)
{
  unsigned char* buffer;
  unsigned error = lodepng_decode(&buffer, &w, &h, &state, in, insize);
  if(buffer && !error)
  {
    size_t buffersize = lodepng_get_raw_size(w, h, &state.info_raw);
    out.insert(out.end(), &buffer[0], &buffer[buffersize]);
    myfree(buffer);
  }
  return error;
}

unsigned decode(std::vector<unsigned char>& out, unsigned& w, unsigned& h,
                State& state,
                const std::vector<unsigned char>& in)
{
  return decode(out, w, h, state, in.empty() ? 0 : &in[0], in.size());
}

#ifdef LODEPNG_COMPILE_DISK
unsigned decode(std::vector<unsigned char>& out, unsigned& w, unsigned& h, const std::string& filename,
                LodePNGColorType colortype, unsigned bitdepth)
{
  std::vector<unsigned char> buffer;
  load_file(buffer, filename);
  return decode(out, w, h, buffer, colortype, bitdepth);
}
#endif //LODEPNG_COMPILE_DECODER
#endif //LODEPNG_COMPILE_DISK

#ifdef LODEPNG_COMPILE_ENCODER
unsigned encode(std::vector<unsigned char>& out, const unsigned char* in, unsigned w, unsigned h,
                LodePNGColorType colortype, unsigned bitdepth)
{
  unsigned char* buffer;
  size_t buffersize;
  unsigned error = lodepng_encode_memory(&buffer, &buffersize, in, w, h, colortype, bitdepth);
  if(buffer)
  {
    out.insert(out.end(), &buffer[0], &buffer[buffersize]);
    myfree(buffer);
  }
  return error;
}

unsigned encode(std::vector<unsigned char>& out,
                const std::vector<unsigned char>& in, unsigned w, unsigned h,
                LodePNGColorType colortype, unsigned bitdepth)
{
  if(lodepng_get_raw_size_lct(w, h, colortype, bitdepth) > in.size()) return 84;
  return encode(out, in.empty() ? 0 : &in[0], w, h, colortype, bitdepth);
}

unsigned encode(std::vector<unsigned char>& out,
                const unsigned char* in, unsigned w, unsigned h,
                State& state)
{
  unsigned char* buffer;
  size_t buffersize;
  unsigned error = lodepng_encode(&buffer, &buffersize, in, w, h, &state);
  if(buffer)
  {
    out.insert(out.end(), &buffer[0], &buffer[buffersize]);
    myfree(buffer);
  }
  return error;
}

unsigned encode(std::vector<unsigned char>& out,
                const std::vector<unsigned char>& in, unsigned w, unsigned h,
                State& state)
{
  if(lodepng_get_raw_size(w, h, &state.info_raw) > in.size()) return 84;
  return encode(out, in.empty() ? 0 : &in[0], w, h, state);
}

#ifdef LODEPNG_COMPILE_DISK
unsigned encode(const std::string& filename,
                const unsigned char* in, unsigned w, unsigned h,
                LodePNGColorType colortype, unsigned bitdepth)
{
  std::vector<unsigned char> buffer;
  unsigned error = encode(buffer, in, w, h, colortype, bitdepth);
  if(!error) save_file(buffer, filename);
  return error;
}

unsigned encode(const std::string& filename,
                const std::vector<unsigned char>& in, unsigned w, unsigned h,
                LodePNGColorType colortype, unsigned bitdepth)
{
  if(lodepng_get_raw_size_lct(w, h, colortype, bitdepth) > in.size()) return 84;
  return encode(filename, in.empty() ? 0 : &in[0], w, h, colortype, bitdepth);
}
#endif //LODEPNG_COMPILE_DISK
#endif //LODEPNG_COMPILE_ENCODER
#endif //LODEPNG_COMPILE_PNG
} //namespace lodepng
#endif /*LODEPNG_COMPILE_CPP*/


void save_png ( char* fname, unsigned char* img, int w, int h )
{
	unsigned error = lodepng::encode ( "test.png", img, w, h );	  
	if (error) printf ( "png encoder error: %s\n", lodepng_error_text(error) );
}


//------------------------------------------------ TGA FORMAT
#ifndef TGA_NOIMPL

TGA::~TGA( void )
{
    if( m_nImageData != NULL )
    {
        free( m_nImageData );
        m_nImageData = NULL;
    }
}

int TGA::returnError( FILE *s, int error )
{
    // Called when there is an error loading the .tga texture file.
    fclose( s );
    return error;
}

unsigned char *TGA::getRGBA( FILE *s, int size )
{
    // Read in RGBA data for a 32bit image. 
    unsigned char *rgba;
    unsigned char temp;
    int bread;
    int i;

    rgba = (unsigned char *)malloc( size * 4 );

    if( rgba == NULL )
        return 0;

    bread = (int)fread( rgba, sizeof (unsigned char), size * 4, s ); 

    // TGA is stored in BGRA, make it RGBA  
    if( bread != size * 4 )
    {
        free( rgba );
        return 0;
    }

    for( i = 0; i < size * 4; i += 4 )
    {
        temp = rgba[i];
        rgba[i] = rgba[i + 2];
        rgba[i + 2] = temp;
    }

    m_texFormat = TGA::RGBA;
    return rgba;
}

unsigned char *TGA::getRGB( FILE *s, int size )
{
    // Read in RGB data for a 24bit image. 
    unsigned char *rgb;
    unsigned char temp;
    int bread;
    int i;

    rgb = (unsigned char*)malloc( size * 3 );

    if( rgb == NULL )
        return 0;

    bread = (int)fread( rgb, sizeof (unsigned char), size * 3, s );

    if(bread != size * 3)
    {
        free( rgb );
        return 0;
    }

    // TGA is stored in BGR, make it RGB  
    for( i = 0; i < size * 3; i += 3 )
    {
        temp = rgb[i];
        rgb[i] = rgb[i + 2];
        rgb[i + 2] = temp;
    }

    m_texFormat = TGA::RGB;

    return rgb;
}

unsigned char *TGA::getGray( FILE *s, int size )
{
    // Gets the grayscale image data.  Used as an alpha channel.
    unsigned char *grayData;
    int bread;

    grayData = (unsigned char*)malloc( size );

    if( grayData == NULL )
        return 0;

    bread = (int)fread( grayData, sizeof (unsigned char), size, s );

    if( bread != size )
    {
        free( grayData );
        return 0;
    }

    m_texFormat = TGA::ALPHA;

    return grayData;
}

TGA::TGAError TGA::load( const char *name )
{
    // Loads up a targa file. Supported types are 8,24 and 32 
    // uncompressed images.
    unsigned char type[4];
    unsigned char info[7];
    FILE *s = NULL;
    int size = 0;
    
    if( !(s = fopen( name, "rb" )) )
        return TGA_FILE_NOT_FOUND;

    fread( &type, sizeof (char), 3, s );   // Read in colormap info and image type, byte 0 ignored
    fseek( s, 12, SEEK_SET);			   // Seek past the header and useless info
    fread( &info, sizeof (char), 6, s );

    if( type[1] != 0 || (type[2] != 2 && type[2] != 3) )
        returnError( s, TGA_BAD_IMAGE_TYPE );

    m_nImageWidth  = info[0] + info[1] * 256; 
    m_nImageHeight = info[2] + info[3] * 256;
    m_nImageBits   = info[4]; 

    size = m_nImageWidth * m_nImageHeight;

    // Make sure we are loading a supported type  
    if( m_nImageBits != 32 && m_nImageBits != 24 && m_nImageBits != 8 )
        returnError( s, TGA_BAD_BITS );

    if( m_nImageBits == 32 )
        m_nImageData = getRGBA( s, size );
    else if( m_nImageBits == 24 )
        m_nImageData = getRGB( s, size );	
    else if( m_nImageBits == 8 )
        m_nImageData = getGray( s, size );

    // No image data 
    if( m_nImageData == NULL )
        returnError( s, TGA_BAD_DATA );

    fclose( s );

    return TGA_NO_ERROR;
}

void TGA::writeRGBA( FILE *s, const unsigned char *externalImage, int size )
{
    // Read in RGBA data for a 32bit image. 
    unsigned char *rgba;
    int bread;
    int i;

    rgba = (unsigned char *)malloc( size * 4 );

    // switch RGBA to BGRA
    for( i = 0; i < size * 4; i += 4 )
    {
        rgba[i + 0] = externalImage[i + 2];
        rgba[i + 1] = externalImage[i + 1];
        rgba[i + 2] = externalImage[i + 0];
        rgba[i + 3] = externalImage[i + 3];
    }

    bread = (int)fwrite( rgba, sizeof (unsigned char), size * 4, s ); 
    free( rgba );
}

void TGA::writeRGB( FILE *s, const unsigned char *externalImage, int size )
{
    // Read in RGBA data for a 32bit image. 
    unsigned char *rgb;
    int bread;
    int i;

    rgb = (unsigned char *)malloc( size * 3 );

    // switch RGB to BGR
    for( i = 0; i < size * 3; i += 3 )
    {
        rgb[i + 0] = externalImage[i + 2];
        rgb[i + 1] = externalImage[i + 1];
        rgb[i + 2] = externalImage[i + 0];
    }

    bread = (int)fwrite( rgb, sizeof (unsigned char), size * 3, s ); 
    free( rgb );
}

void TGA::writeGrayAsRGB( FILE *s, const unsigned char *externalImage, int size )
{
    // Read in RGBA data for a 32bit image. 
    unsigned char *rgb;
    int bread;
    int i;

    rgb = (unsigned char *)malloc( size * 3 );

    // switch RGB to BGR
    int j = 0;
    for( i = 0; i < size * 3; i += 3, j++ )
    {
        rgb[i + 0] = externalImage[j];
        rgb[i + 1] = externalImage[j];
        rgb[i + 2] = externalImage[j];
    }

    bread = (int)fwrite( rgb, sizeof (unsigned char), size * 3, s ); 
    free( rgb );
}

void TGA::writeGray( FILE *s, const unsigned char *externalImage, int size )
{
    // Gets the grayscale image data.  Used as an alpha channel.
    int bread;

    bread = (int)fwrite( externalImage, sizeof (unsigned char), size, s );
}

TGA::TGAError TGA::saveFromExternalData( const char *name, int w, int h, TGA::TGAFormat fmt, const unsigned char *externalImage )
{
    static unsigned char type[] = {0,0,2};
    static unsigned char dummy[] = {0,0,0,0,0,0,0,0,0};
    static unsigned char info[] = {0,0,0,0,0,0};
    FILE *s = NULL;
    int size = 0;
    
    if( !(s = fopen( name, "wb" )) )
        return TGA_FILE_NOT_FOUND;

    fwrite( type, sizeof (char), 3, s );   // Read in colormap info and image type, byte 0 ignored
    fwrite( dummy, sizeof (char), 9, s );   // Read in colormap info and image type, byte 0 ignored

    info[0] = w & 0xFF;
    info[1] = (w>>8) & 0xFF;
    info[2] = h & 0xFF;
    info[3] = (h>>8) & 0xFF;
    switch(fmt)
    {
    case ALPHA:
        info[4] = 8;
        break;
    case RGB:
        info[4] = 24;
        break;
    case RGBA:
        info[4] = 32;
        break;
    }
    fwrite( info, sizeof (char), 6, s );

    size = w*h;
    switch(fmt)
    {
    case ALPHA:
        writeGray(s, externalImage, size);
        break;
    case RGB:
        writeGrayAsRGB/*writeRGB*/(s, externalImage, size);
        break;
    case RGBA:
        writeRGBA(s, externalImage, size);
        break;
    }

    fclose( s );

    return TGA_NO_ERROR;
}
#endif		// #ifndef TGA_NOIMPL


#undef VTYPE
#define VTYPE	float

Vector3DF &Vector3DF::operator*= (const MatrixF &op)
{
	double *m = op.GetDataF ();
	float xa, ya, za;
	xa = x * float(*m++);	ya = x * float(*m++);	za = x * float(*m++);	m++;
	xa += y * float(*m++);	ya += y * float(*m++);	za += y * float(*m++);	m++;
	xa += z * float(*m++);	ya += z * float(*m++);	za += z * float(*m++);	m++;
	xa += float(*m++);		ya += float(*m++);		za += float(*m++);
	x = xa; y = ya; z = za;
	return *this;
}

// p' = Mp
Vector3DF &Vector3DF::operator*= (const Matrix4F &op)
{
	float xa, ya, za;
	xa = x * op.data[0] + y * op.data[4] + z * op.data[8] + op.data[12];
	ya = x * op.data[1] + y * op.data[5] + z * op.data[9] + op.data[13];
	za = x * op.data[2] + y * op.data[6] + z * op.data[10] + op.data[14];
	x = xa; y = ya; z = za;
	return *this;
}

	
#define min3(a,b,c)		( (a<b) ? ((a<c) ? a : c) : ((b<c) ? b : c) )
#define max3(a,b,c)		( (a>b) ? ((a>c) ? a : c) : ((b>c) ? b : c) )

Vector3DF Vector3DF::RGBtoHSV ()
{
	float h,s,v;
	float minv, maxv;
	int i;
	float f;

	minv = min3(x, y, z);
	maxv = max3(x, y, z);
	if (minv==maxv) {
		v = (float) maxv;
		h = 0.0; 
		s = 0.0;			
	} else {
		v = (float) maxv;
		s = (maxv - minv) / maxv;
		f = (x == minv) ? y - z : ((y == minv) ? z - x : x - y); 	
		i = (x == minv) ? 3 : ((y == minv) ? 5 : 1);
		h = (i - f / (maxv - minv) ) / 6.0f;	
	}
	return Vector3DF(h,s,v);
}

Vector3DF Vector3DF::HSVtoRGB ()
{
	float m, n, f;
	int i = (int) floor ( x*6.0 );
	f = x*6.0f - i;
	if ( i % 2 == 0 ) f = 1.0f - f;	
	m = z * (1.0f - y );
	n = z * (1.0f - y * f );	
	switch ( i ) {
	case 6: 
	case 0: return Vector3DF( z, n, m );	break;
	case 1: return Vector3DF( n, z, m );	break;
	case 2: return Vector3DF( m, z, n );	break;
	case 3: return Vector3DF( m, n, z );	break;
	case 4: return Vector3DF( n, m, z );	break;
	case 5: return Vector3DF( z, m, n );	break;
	};
	return Vector3DF(1,1,1);
}

Vector4DF &Vector4DF::operator*= (const MatrixF &op)
{
	double *m = op.GetDataF ();
	VTYPE xa, ya, za, wa;
	xa = x * float(*m++);	ya = x * float(*m++);	za = x * float(*m++);	wa = x * float(*m++);
	xa += y * float(*m++);	ya += y * float(*m++);	za += y * float(*m++);	wa += y * float(*m++);
	xa += z * float(*m++);	ya += z * float(*m++);	za += z * float(*m++);	wa += z * float(*m++);
	xa += w * float(*m++);	ya += w * float(*m++);	za += w * float(*m++);	wa += w * float(*m++);
	x = xa; y = ya; z = za; w = wa;
	return *this;
}

Vector4DF &Vector4DF::operator*= (const Matrix4F &op)
{
	float xa, ya, za, wa;
	xa = x * op.data[0] + y * op.data[4] + z * op.data[8] + w * op.data[12];
	ya = x * op.data[1] + y * op.data[5] + z * op.data[9] + w * op.data[13];
	za = x * op.data[2] + y * op.data[6] + z * op.data[10] + w * op.data[14];
	wa = x * op.data[3] + y * op.data[7] + z * op.data[11] + w * op.data[15];
	x = xa; y = ya; z = za; w = wa;
	return *this;
}


Vector4DF &Vector4DF::operator*= (const float* op)
{
	float xa, ya, za, wa;
	xa = x * op[0] + y * op[4] + z * op[8] + w * op[12];
	ya = x * op[1] + y * op[5] + z * op[9] + w * op[13];
	za = x * op[2] + y * op[6] + z * op[10] + w * op[14];
	wa = x * op[3] + y * op[7] + z * op[11] + w * op[15];
	x = xa; y = ya; z = za; w = wa;
	return *this;
}

//----------------------------------------------------------------------------------------------------------

// Vector Operations Implemented:
//		=, +, -, *, / (on vectors and scalars)
//		Cross			Cross product vector with op
//		Dot				Dot product vector with op
//		Dist (op)		Distance from vector to op
//		DistSq			Distance^2 from vector to op
//		Length ()		Length of vector
//		Normalize ()	Normalizes vector
//

#include <math.h>

// Vector2DC Code Definition

#undef VTYPE
#define VTYPE		unsigned char
#define VNAME		2DC

// Constructors/Destructors
Vector2DC::Vector2DC() {x=0; y=0;}
Vector2DC::Vector2DC (VTYPE xa, VTYPE ya) {x=xa; y=ya;}
Vector2DC::Vector2DC (Vector2DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}
Vector2DC::Vector2DC (Vector2DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}
Vector2DC::Vector2DC (Vector2DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}
Vector2DC::Vector2DC (Vector3DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}
Vector2DC::Vector2DC (Vector3DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}
Vector2DC::Vector2DC (Vector3DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}
Vector2DC::Vector2DC (Vector4DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}

// Member Functions
Vector2DC &Vector2DC::operator= (Vector2DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
Vector2DC &Vector2DC::operator= (Vector2DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
Vector2DC &Vector2DC::operator= (Vector2DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
Vector2DC &Vector2DC::operator= (Vector3DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
Vector2DC &Vector2DC::operator= (Vector3DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
Vector2DC &Vector2DC::operator= (Vector3DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
Vector2DC &Vector2DC::operator= (Vector4DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}	
	
Vector2DC &Vector2DC::operator+= (Vector2DC &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
Vector2DC &Vector2DC::operator+= (Vector2DI &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
Vector2DC &Vector2DC::operator+= (Vector2DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
Vector2DC &Vector2DC::operator+= (Vector3DC &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
Vector2DC &Vector2DC::operator+= (Vector3DI &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
Vector2DC &Vector2DC::operator+= (Vector3DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
Vector2DC &Vector2DC::operator+= (Vector4DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}

Vector2DC &Vector2DC::operator-= (Vector2DC &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
Vector2DC &Vector2DC::operator-= (Vector2DI &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
Vector2DC &Vector2DC::operator-= (Vector2DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
Vector2DC &Vector2DC::operator-= (Vector3DC &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
Vector2DC &Vector2DC::operator-= (Vector3DI &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
Vector2DC &Vector2DC::operator-= (Vector3DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
Vector2DC &Vector2DC::operator-= (Vector4DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
	
Vector2DC &Vector2DC::operator*= (Vector2DC &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
Vector2DC &Vector2DC::operator*= (Vector2DI &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
Vector2DC &Vector2DC::operator*= (Vector2DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
Vector2DC &Vector2DC::operator*= (Vector3DC &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
Vector2DC &Vector2DC::operator*= (Vector3DI &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
Vector2DC &Vector2DC::operator*= (Vector3DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
Vector2DC &Vector2DC::operator*= (Vector4DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}

Vector2DC &Vector2DC::operator/= (Vector2DC &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
Vector2DC &Vector2DC::operator/= (Vector2DI &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
Vector2DC &Vector2DC::operator/= (Vector2DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
Vector2DC &Vector2DC::operator/= (Vector3DC &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
Vector2DC &Vector2DC::operator/= (Vector3DI &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
Vector2DC &Vector2DC::operator/= (Vector3DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
Vector2DC &Vector2DC::operator/= (Vector4DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}

// Note: Cross product does not exist for 2D vectors (only 3D)
		
double Vector2DC::Dot(Vector2DC &v)			{double dot; dot = (double) x*v.x + (double) y*v.y; return dot;}
double Vector2DC::Dot(Vector2DI &v)			{double dot; dot = (double) x*v.x + (double) y*v.y; return dot;}
double Vector2DC::Dot(Vector2DF &v)			{double dot; dot = (double) x*v.x + (double) y*v.y; return dot;}

double Vector2DC::Dist (Vector2DC &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector2DC::Dist (Vector2DI &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector2DC::Dist (Vector2DF &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector2DC::Dist (Vector3DC &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector2DC::Dist (Vector3DI &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector2DC::Dist (Vector3DF &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector2DC::Dist (Vector4DF &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector2DC::DistSq (Vector2DC &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}
double Vector2DC::DistSq (Vector2DI &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}
double Vector2DC::DistSq (Vector2DF &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}
double Vector2DC::DistSq (Vector3DC &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}
double Vector2DC::DistSq (Vector3DI &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}
double Vector2DC::DistSq (Vector3DF &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}
double Vector2DC::DistSq (Vector4DF &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}

Vector2DC &Vector2DC::Normalize (void) {
	double n = (double) x*x + (double) y*y;
	if (n!=0.0) {
		n = sqrt(n);
		x = (VTYPE) (((double) x*255)/n); 
		y = (VTYPE) (((double) y*255)/n);				
	}
	return *this;
}
double Vector2DC::Length (void) { double n; n = (double) x*x + (double) y*y; if (n != 0.0) return sqrt(n); return 0.0; }

#undef VTYPE
#undef VNAME

// Vector2DI Code Definition

#define VNAME		2DI
#define VTYPE		int

// Constructors/Destructors
Vector2DI::Vector2DI() {x=0; y=0;}
Vector2DI::Vector2DI (const VTYPE xa, const VTYPE ya) {x=xa; y=ya;}
Vector2DI::Vector2DI (const Vector2DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}
Vector2DI::Vector2DI (const Vector2DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}
Vector2DI::Vector2DI (const Vector2DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}
Vector2DI::Vector2DI (const Vector3DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}
Vector2DI::Vector2DI (const Vector3DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}
Vector2DI::Vector2DI (const Vector3DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}
Vector2DI::Vector2DI (const Vector4DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}

// Member Functions
Vector2DI &Vector2DI::operator= (const Vector2DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
Vector2DI &Vector2DI::operator= (const Vector2DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
Vector2DI &Vector2DI::operator= (const Vector2DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
Vector2DI &Vector2DI::operator= (const Vector3DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
Vector2DI &Vector2DI::operator= (const Vector3DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
Vector2DI &Vector2DI::operator= (const Vector3DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
Vector2DI &Vector2DI::operator= (const Vector4DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}	
	
Vector2DI &Vector2DI::operator+= (const Vector2DC &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
Vector2DI &Vector2DI::operator+= (const Vector2DI &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
Vector2DI &Vector2DI::operator+= (const Vector2DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
Vector2DI &Vector2DI::operator+= (const Vector3DC &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
Vector2DI &Vector2DI::operator+= (const Vector3DI &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
Vector2DI &Vector2DI::operator+= (const Vector3DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
Vector2DI &Vector2DI::operator+= (const Vector4DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}

Vector2DI &Vector2DI::operator-= (const Vector2DC &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
Vector2DI &Vector2DI::operator-= (const Vector2DI &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
Vector2DI &Vector2DI::operator-= (const Vector2DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
Vector2DI &Vector2DI::operator-= (const Vector3DC &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
Vector2DI &Vector2DI::operator-= (const Vector3DI &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
Vector2DI &Vector2DI::operator-= (const Vector3DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
Vector2DI &Vector2DI::operator-= (const Vector4DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
	
Vector2DI &Vector2DI::operator*= (const Vector2DC &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
Vector2DI &Vector2DI::operator*= (const Vector2DI &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
Vector2DI &Vector2DI::operator*= (const Vector2DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
Vector2DI &Vector2DI::operator*= (const Vector3DC &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
Vector2DI &Vector2DI::operator*= (const Vector3DI &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
Vector2DI &Vector2DI::operator*= (const Vector3DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
Vector2DI &Vector2DI::operator*= (const Vector4DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}

Vector2DI &Vector2DI::operator/= (const Vector2DC &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
Vector2DI &Vector2DI::operator/= (const Vector2DI &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
Vector2DI &Vector2DI::operator/= (const Vector2DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
Vector2DI &Vector2DI::operator/= (const Vector3DC &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
Vector2DI &Vector2DI::operator/= (const Vector3DI &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
Vector2DI &Vector2DI::operator/= (const Vector3DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
Vector2DI &Vector2DI::operator/= (const Vector4DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}

// Note: Cross product does not exist for 2D vectors (only 3D)
		
double Vector2DI::Dot(const Vector2DC &v)			{double dot; dot = (double) x*v.x + (double) y*v.y; return dot;}
double Vector2DI::Dot(const Vector2DI &v)			{double dot; dot = (double) x*v.x + (double) y*v.y; return dot;}
double Vector2DI::Dot(const Vector2DF &v)			{double dot; dot = (double) x*v.x + (double) y*v.y; return dot;}

double Vector2DI::Dist (const Vector2DC &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector2DI::Dist (const Vector2DI &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector2DI::Dist (const Vector2DF &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector2DI::Dist (const Vector3DC &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector2DI::Dist (const Vector3DI &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector2DI::Dist (const Vector3DF &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector2DI::Dist (const Vector4DF &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector2DI::DistSq (const Vector2DC &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}
double Vector2DI::DistSq (const Vector2DI &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}
double Vector2DI::DistSq (const Vector2DF &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}
double Vector2DI::DistSq (const Vector3DC &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}
double Vector2DI::DistSq (const Vector3DI &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}
double Vector2DI::DistSq (const Vector3DF &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}
double Vector2DI::DistSq (const Vector4DF &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}

Vector2DI &Vector2DI::Normalize (void) {
	double n = (double) x*x + (double) y*y;
	if (n!=0.0) {
		n = sqrt(n);
		x = (VTYPE) (((double) x*255)/n);
		y = (VTYPE) (((double) y*255)/n);				
	}
	return *this;
}
double Vector2DI::Length (void) { double n; n = (double) x*x + (double) y*y; if (n != 0.0) return sqrt(n); return 0.0; }



#undef VTYPE
#undef VNAME

// Vector2DF Code Definition

#define VNAME		2DF
#define VTYPE		float

// Constructors/Destructors
Vector2DF::Vector2DF() {x=0; y=0;}
Vector2DF::Vector2DF (const VTYPE xa, const VTYPE ya) {x=xa; y=ya;}
Vector2DF::Vector2DF (const Vector2DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}
Vector2DF::Vector2DF (const Vector2DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}
Vector2DF::Vector2DF (const Vector2DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}
Vector2DF::Vector2DF (const Vector3DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}
Vector2DF::Vector2DF (const Vector3DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}
Vector2DF::Vector2DF (const Vector3DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}
Vector2DF::Vector2DF (const Vector4DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y;}

// Member Functions
Vector2DF &Vector2DF::operator= (const Vector2DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
Vector2DF &Vector2DF::operator= (const Vector2DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
Vector2DF &Vector2DF::operator= (const Vector2DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
Vector2DF &Vector2DF::operator= (const Vector3DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
Vector2DF &Vector2DF::operator= (const Vector3DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
Vector2DF &Vector2DF::operator= (const Vector3DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
Vector2DF &Vector2DF::operator= (const Vector4DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}	
	
Vector2DF &Vector2DF::operator+= (const Vector2DC &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
Vector2DF &Vector2DF::operator+= (const Vector2DI &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
Vector2DF &Vector2DF::operator+= (const Vector2DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
Vector2DF &Vector2DF::operator+= (const Vector3DC &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
Vector2DF &Vector2DF::operator+= (const Vector3DI &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
Vector2DF &Vector2DF::operator+= (const Vector3DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
Vector2DF &Vector2DF::operator+= (const Vector4DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}

Vector2DF &Vector2DF::operator-= (const Vector2DC &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
Vector2DF &Vector2DF::operator-= (const Vector2DI &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
Vector2DF &Vector2DF::operator-= (const Vector2DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
Vector2DF &Vector2DF::operator-= (const Vector3DC &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
Vector2DF &Vector2DF::operator-= (const Vector3DI &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
Vector2DF &Vector2DF::operator-= (const Vector3DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
Vector2DF &Vector2DF::operator-= (const Vector4DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
	
Vector2DF &Vector2DF::operator*= (const Vector2DC &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
Vector2DF &Vector2DF::operator*= (const Vector2DI &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
Vector2DF &Vector2DF::operator*= (const Vector2DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
Vector2DF &Vector2DF::operator*= (const Vector3DC &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
Vector2DF &Vector2DF::operator*= (const Vector3DI &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
Vector2DF &Vector2DF::operator*= (const Vector3DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
Vector2DF &Vector2DF::operator*= (const Vector4DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}

Vector2DF &Vector2DF::operator/= (const Vector2DC &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
Vector2DF &Vector2DF::operator/= (const Vector2DI &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
Vector2DF &Vector2DF::operator/= (const Vector2DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
Vector2DF &Vector2DF::operator/= (const Vector3DC &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
Vector2DF &Vector2DF::operator/= (const Vector3DI &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
Vector2DF &Vector2DF::operator/= (const Vector3DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
Vector2DF &Vector2DF::operator/= (const Vector4DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}

// Note: Cross product does not exist for 2D vectors (only 3D)
		
double Vector2DF::Dot(const Vector2DC &v)			{double dot; dot = (double) x*v.x + (double) y*v.y; return dot;}
double Vector2DF::Dot(const Vector2DI &v)			{double dot; dot = (double) x*v.x + (double) y*v.y; return dot;}
double Vector2DF::Dot(const Vector2DF &v)			{double dot; dot = (double) x*v.x + (double) y*v.y; return dot;}

double Vector2DF::Dist (const Vector2DC &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector2DF::Dist (const Vector2DI &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector2DF::Dist (const Vector2DF &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector2DF::Dist (const Vector3DC &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector2DF::Dist (const Vector3DI &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector2DF::Dist (const Vector3DF &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector2DF::Dist (const Vector4DF &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector2DF::DistSq (const Vector2DC &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}
double Vector2DF::DistSq (const Vector2DI &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}
double Vector2DF::DistSq (const Vector2DF &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}
double Vector2DF::DistSq (const Vector3DC &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}
double Vector2DF::DistSq (const Vector3DI &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}
double Vector2DF::DistSq (const Vector3DF &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}
double Vector2DF::DistSq (const Vector4DF &v)		{ double a,b; a = (double) x - (double) v.x; b = (double) y - (double) v.y; return (a*a + b*b);}

Vector2DF &Vector2DF::Normalize (void) {
	double n = (double) x*x + (double) y*y;
	if (n!=0.0) {
		n = sqrt(n);
		x /= (float) n;
		y /= (float) n;
	}
	return *this;
}
double Vector2DF::Length (void) { double n; n = (double) x*x + (double) y*y; if (n != 0.0) return sqrt(n); return 0.0; }

#undef VTYPE
#undef VNAME

// Vector3DC Code Definition

#define VNAME		3DC
#define VTYPE		unsigned char

// Constructors/Destructors
Vector3DC::Vector3DC() {x=0; y=0; z=0;}
Vector3DC::Vector3DC (const VTYPE xa, const VTYPE ya, const VTYPE za) {x=xa; y=ya; z=za;}
Vector3DC::Vector3DC (const Vector2DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0;}
Vector3DC::Vector3DC (const Vector2DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0;}
Vector3DC::Vector3DC (const Vector2DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0;}
Vector3DC::Vector3DC (const Vector3DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z;}
Vector3DC::Vector3DC (const Vector3DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z;}
Vector3DC::Vector3DC (const Vector3DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z;}
Vector3DC::Vector3DC (const Vector4DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z;}

// Member Functions
Vector3DC &Vector3DC::Set (const VTYPE xa, const VTYPE ya, const VTYPE za) {x=xa; y=ya; z=za; return *this;}

Vector3DC &Vector3DC::operator= (const Vector2DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0; return *this;}
Vector3DC &Vector3DC::operator= (const Vector2DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0; return *this;}
Vector3DC &Vector3DC::operator= (const Vector2DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0; return *this;}
Vector3DC &Vector3DC::operator= (const Vector3DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; return *this;}
Vector3DC &Vector3DC::operator= (const Vector3DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; return *this;}
Vector3DC &Vector3DC::operator= (const Vector3DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; return *this;}
Vector3DC &Vector3DC::operator= (const Vector4DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; return *this;}	
	
Vector3DC &Vector3DC::operator+= (const Vector2DC &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
Vector3DC &Vector3DC::operator+= (const Vector2DI &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
Vector3DC &Vector3DC::operator+= (const Vector2DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
Vector3DC &Vector3DC::operator+= (const Vector3DC &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; z+=(VTYPE) op.z; return *this;}
Vector3DC &Vector3DC::operator+= (const Vector3DI &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; z+=(VTYPE) op.z; return *this;}
Vector3DC &Vector3DC::operator+= (const Vector3DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; z+=(VTYPE) op.z; return *this;}
Vector3DC &Vector3DC::operator+= (const Vector4DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; z+=(VTYPE) op.z; return *this;}

Vector3DC &Vector3DC::operator-= (const Vector2DC &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
Vector3DC &Vector3DC::operator-= (const Vector2DI &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
Vector3DC &Vector3DC::operator-= (const Vector2DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
Vector3DC &Vector3DC::operator-= (const Vector3DC &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; z-=(VTYPE) op.z; return *this;}
Vector3DC &Vector3DC::operator-= (const Vector3DI &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; z-=(VTYPE) op.z; return *this;}
Vector3DC &Vector3DC::operator-= (const Vector3DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; z-=(VTYPE) op.z; return *this;}
Vector3DC &Vector3DC::operator-= (const Vector4DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; z-=(VTYPE) op.z; return *this;}
	
Vector3DC &Vector3DC::operator*= (const Vector2DC &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
Vector3DC &Vector3DC::operator*= (const Vector2DI &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
Vector3DC &Vector3DC::operator*= (const Vector2DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
Vector3DC &Vector3DC::operator*= (const Vector3DC &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; z*=(VTYPE) op.z; return *this;}
Vector3DC &Vector3DC::operator*= (const Vector3DI &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; z*=(VTYPE) op.z; return *this;}
Vector3DC &Vector3DC::operator*= (const Vector3DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; z*=(VTYPE) op.z; return *this;}
Vector3DC &Vector3DC::operator*= (const Vector4DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; z*=(VTYPE) op.z; return *this;}

Vector3DC &Vector3DC::operator/= (const Vector2DC &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
Vector3DC &Vector3DC::operator/= (const Vector2DI &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
Vector3DC &Vector3DC::operator/= (const Vector2DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
Vector3DC &Vector3DC::operator/= (const Vector3DC &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; z/=(VTYPE) op.z; return *this;}
Vector3DC &Vector3DC::operator/= (const Vector3DI &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; z/=(VTYPE) op.z; return *this;}
Vector3DC &Vector3DC::operator/= (const Vector3DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; z/=(VTYPE) op.z; return *this;}
Vector3DC &Vector3DC::operator/= (const Vector4DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; z/=(VTYPE) op.z; return *this;}

Vector3DC &Vector3DC::Cross (const Vector3DC &v) {double ax = x, ay = y, az = z; x = (VTYPE) (ay * (double) v.z - az * (double) v.y); y = (VTYPE) (-ax * (double) v.z + az * (double) v.x); z = (VTYPE) (ax * (double) v.y - ay * (double) v.x); return *this;}
Vector3DC &Vector3DC::Cross (const Vector3DI &v) {double ax = x, ay = y, az = z; x = (VTYPE) (ay * (double) v.z - az * (double) v.y); y = (VTYPE) (-ax * (double) v.z + az * (double) v.x); z = (VTYPE) (ax * (double) v.y - ay * (double) v.x); return *this;}
Vector3DC &Vector3DC::Cross (const Vector3DF &v) {double ax = x, ay = y, az = z; x = (VTYPE) (ay * (double) v.z - az * (double) v.y); y = (VTYPE) (-ax * (double) v.z + az * (double) v.x); z = (VTYPE) (ax * (double) v.y - ay * (double) v.x); return *this;}

double Vector3DC::Dot(const Vector3DC &v)			{double dot; dot = (double) x*v.x + (double) y*v.y + (double) z*v.z; return dot;}
double Vector3DC::Dot(const Vector3DI &v)			{double dot; dot = (double) x*v.x + (double) y*v.y + (double) z*v.z; return dot;}
double Vector3DC::Dot(const Vector3DF &v)			{double dot; dot = (double) x*v.x + (double) y*v.y + (double) z*v.z; return dot;}

double Vector3DC::Dist (const Vector2DC &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector3DC::Dist (const Vector2DI &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector3DC::Dist (const Vector2DF &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector3DC::Dist (const Vector3DC &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector3DC::Dist (const Vector3DI &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector3DC::Dist (const Vector3DF &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector3DC::Dist (const Vector4DF &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector3DC::DistSq (const Vector2DC &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z; return (a*a + b*b + c*c);}
double Vector3DC::DistSq (const Vector2DI &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z; return (a*a + b*b + c*c);}
double Vector3DC::DistSq (const Vector2DF &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z; return (a*a + b*b + c*c);}
double Vector3DC::DistSq (const Vector3DC &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z - (double) v.z; return (a*a + b*b + c*c);}
double Vector3DC::DistSq (const Vector3DI &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z - (double) v.z; return (a*a + b*b + c*c);}
double Vector3DC::DistSq (const Vector3DF &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z - (double) v.z; return (a*a + b*b + c*c);}
double Vector3DC::DistSq (const Vector4DF &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z - (double) v.z; return (a*a + b*b + c*c);}

Vector3DC &Vector3DC::Normalize (void) {
	double n = (double) x*x + (double) y*y + (double) z*z;
	if (n!=0.0) {
		n = sqrt(n);
		x = (VTYPE) (((double) x*255)/n);
		y = (VTYPE) (((double) y*255)/n);
		z = (VTYPE) (((double) z*255)/n);
	}
	return *this;
}
double Vector3DC::Length (void) { double n; n = (double) x*x + (double) y*y + (double) z*z; if (n != 0.0) return sqrt(n); return 0.0; }



#undef VTYPE
#undef VNAME

// Vector3DI Code Definition

#define VNAME		3DI
#define VTYPE		int

// Constructors/Destructors
Vector3DI::Vector3DI() {x=0; y=0; z=0;}
Vector3DI::Vector3DI (const VTYPE xa, const VTYPE ya, const VTYPE za) {x=xa; y=ya; z=za;}
Vector3DI::Vector3DI (const Vector2DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0;}
Vector3DI::Vector3DI (const Vector2DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0;}
Vector3DI::Vector3DI (const Vector2DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0;}
Vector3DI::Vector3DI (const Vector3DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z;}
Vector3DI::Vector3DI (const Vector3DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z;}
Vector3DI::Vector3DI (const Vector3DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z;}
Vector3DI::Vector3DI (const Vector4DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z;}

// Set Functions
Vector3DI &Vector3DI::Set (const int xa, const int ya, const int za)
{
	x = xa; y = ya; z = za;
	return *this;
}

// Member Functions
Vector3DI &Vector3DI::operator= (const Vector2DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
Vector3DI &Vector3DI::operator= (const Vector2DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
Vector3DI &Vector3DI::operator= (const Vector2DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
Vector3DI &Vector3DI::operator= (const Vector3DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; return *this;}
Vector3DI &Vector3DI::operator= (const Vector3DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; return *this;}
Vector3DI &Vector3DI::operator= (const Vector3DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; return *this;}
Vector3DI &Vector3DI::operator= (const Vector4DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; return *this;}	
	
Vector3DI &Vector3DI::operator+= (const Vector2DC &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
Vector3DI &Vector3DI::operator+= (const Vector2DI &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
Vector3DI &Vector3DI::operator+= (const Vector2DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
Vector3DI &Vector3DI::operator+= (const Vector3DC &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; z+=(VTYPE) op.z; return *this;}
Vector3DI &Vector3DI::operator+= (const Vector3DI &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; z+=(VTYPE) op.z; return *this;}
Vector3DI &Vector3DI::operator+= (const Vector3DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; z+=(VTYPE) op.z; return *this;}
Vector3DI &Vector3DI::operator+= (const Vector4DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; z+=(VTYPE) op.z; return *this;}

Vector3DI &Vector3DI::operator-= (const Vector2DC &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
Vector3DI &Vector3DI::operator-= (const Vector2DI &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
Vector3DI &Vector3DI::operator-= (const Vector2DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
Vector3DI &Vector3DI::operator-= (const Vector3DC &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; z-=(VTYPE) op.z; return *this;}
Vector3DI &Vector3DI::operator-= (const Vector3DI &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; z-=(VTYPE) op.z; return *this;}
Vector3DI &Vector3DI::operator-= (const Vector3DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; z-=(VTYPE) op.z; return *this;}
Vector3DI &Vector3DI::operator-= (const Vector4DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; z-=(VTYPE) op.z; return *this;}
	
Vector3DI &Vector3DI::operator*= (const Vector2DC &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
Vector3DI &Vector3DI::operator*= (const Vector2DI &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
Vector3DI &Vector3DI::operator*= (const Vector2DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
Vector3DI &Vector3DI::operator*= (const Vector3DC &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; z*=(VTYPE) op.z; return *this;}
Vector3DI &Vector3DI::operator*= (const Vector3DI &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; z*=(VTYPE) op.z; return *this;}
Vector3DI &Vector3DI::operator*= (const Vector3DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; z*=(VTYPE) op.z; return *this;}
Vector3DI &Vector3DI::operator*= (const Vector4DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; z*=(VTYPE) op.z; return *this;}

Vector3DI &Vector3DI::operator/= (const Vector2DC &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
Vector3DI &Vector3DI::operator/= (const Vector2DI &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
Vector3DI &Vector3DI::operator/= (const Vector2DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
Vector3DI &Vector3DI::operator/= (const Vector3DC &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; z/=(VTYPE) op.z; return *this;}
Vector3DI &Vector3DI::operator/= (const Vector3DI &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; z/=(VTYPE) op.z; return *this;}
Vector3DI &Vector3DI::operator/= (const Vector3DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; z/=(VTYPE) op.z; return *this;}
Vector3DI &Vector3DI::operator/= (const Vector4DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; z/=(VTYPE) op.z; return *this;}

Vector3DI &Vector3DI::Cross (const Vector3DC &v) {double ax = x, ay = y, az = z; x = (VTYPE) (ay * (double) v.z - az * (double) v.y); y = (VTYPE) (-ax * (double) v.z + az * (double) v.x); z = (VTYPE) (ax * (double) v.y - ay * (double) v.x); return *this;}
Vector3DI &Vector3DI::Cross (const Vector3DI &v) {double ax = x, ay = y, az = z; x = (VTYPE) (ay * (double) v.z - az * (double) v.y); y = (VTYPE) (-ax * (double) v.z + az * (double) v.x); z = (VTYPE) (ax * (double) v.y - ay * (double) v.x); return *this;}
Vector3DI &Vector3DI::Cross (const Vector3DF &v) {double ax = x, ay = y, az = z; x = (VTYPE) (ay * (double) v.z - az * (double) v.y); y = (VTYPE) (-ax * (double) v.z + az * (double) v.x); z = (VTYPE) (ax * (double) v.y - ay * (double) v.x); return *this;}
		
double Vector3DI::Dot(const Vector3DC &v)			{double dot; dot = (double) x*v.x + (double) y*v.y + (double) z*v.z; return dot;}
double Vector3DI::Dot(const Vector3DI &v)			{double dot; dot = (double) x*v.x + (double) y*v.y + (double) z*v.z; return dot;}
double Vector3DI::Dot(const Vector3DF &v)			{double dot; dot = (double) x*v.x + (double) y*v.y + (double) z*v.z; return dot;}

double Vector3DI::Dist (const Vector2DC &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector3DI::Dist (const Vector2DI &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector3DI::Dist (const Vector2DF &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector3DI::Dist (const Vector3DC &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector3DI::Dist (const Vector3DI &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector3DI::Dist (const Vector3DF &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector3DI::Dist (const Vector4DF &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector3DI::DistSq (const Vector2DC &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z; return (a*a + b*b + c*c);}
double Vector3DI::DistSq (const Vector2DI &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z; return (a*a + b*b + c*c);}
double Vector3DI::DistSq (const Vector2DF &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z; return (a*a + b*b + c*c);}
double Vector3DI::DistSq (const Vector3DC &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z - (double) v.z; return (a*a + b*b + c*c);}
double Vector3DI::DistSq (const Vector3DI &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z - (double) v.z; return (a*a + b*b + c*c);}
double Vector3DI::DistSq (const Vector3DF &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z - (double) v.z; return (a*a + b*b + c*c);}
double Vector3DI::DistSq (const Vector4DF &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z - (double) v.z; return (a*a + b*b + c*c);}

Vector3DI &Vector3DI::Normalize (void) {
	double n = (double) x*x + (double) y*y + (double) z*z;
	if (n!=0.0) {
		n = sqrt(n);
		x = (VTYPE) (((double) x*255)/n);
		y = (VTYPE) (((double) y*255)/n);
		z = (VTYPE) (((double) z*255)/n);
	}
	return *this;
}
double Vector3DI::Length (void) { double n; n = (double) x*x + (double) y*y + (double) z*z; if (n != 0.0) return sqrt(n); return 0.0; }


#undef VTYPE
#undef VNAME

// Vector3DF Code Definition

#define VNAME		3DF
#define VTYPE		float

Vector3DF::Vector3DF (const VTYPE xa, const VTYPE ya, const VTYPE za) {x=xa; y=ya; z=za;}
Vector3DF::Vector3DF (const Vector2DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0;}
Vector3DF::Vector3DF (const Vector2DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0;}
Vector3DF::Vector3DF (const Vector2DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0;}
Vector3DF::Vector3DF (const Vector3DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z;}
Vector3DF::Vector3DF (const Vector3DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z;}
Vector3DF::Vector3DF (const Vector3DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z;}
Vector3DF::Vector3DF (const Vector4DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z;}

// Set Functions
Vector3DF &Vector3DF::Set (const VTYPE xa, const VTYPE ya, const VTYPE za)
{
	x = (float) xa; y = (float) ya; z = (float) za;
	return *this;
}

// Member Functions
Vector3DF &Vector3DF::operator= (const int op) {x= (VTYPE) op; y= (VTYPE) op; z= (VTYPE) op; return *this;}
Vector3DF &Vector3DF::operator= (const double op) {x= (VTYPE) op; y= (VTYPE) op; z= (VTYPE) op; return *this;}
Vector3DF &Vector3DF::operator= (const Vector2DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
Vector3DF &Vector3DF::operator= (const Vector2DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
Vector3DF &Vector3DF::operator= (const Vector2DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; return *this;}
Vector3DF &Vector3DF::operator= (const Vector3DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; return *this;}
Vector3DF &Vector3DF::operator= (const Vector3DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; return *this;}
Vector3DF &Vector3DF::operator= (const Vector3DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; return *this;}
Vector3DF &Vector3DF::operator= (const Vector4DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; return *this;}	
	
Vector3DF &Vector3DF::operator+= (const int op) {x+= (VTYPE) op; y+= (VTYPE) op; z+= (VTYPE) op; return *this;}
Vector3DF &Vector3DF::operator+= (const double op) {x+= (VTYPE) op; y+= (VTYPE) op; z+= (VTYPE) op; return *this;}
Vector3DF &Vector3DF::operator+= (const Vector2DC &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
Vector3DF &Vector3DF::operator+= (const Vector2DI &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
Vector3DF &Vector3DF::operator+= (const Vector2DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
Vector3DF &Vector3DF::operator+= (const Vector3DC &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; z+=(VTYPE) op.z; return *this;}
Vector3DF &Vector3DF::operator+= (const Vector3DI &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; z+=(VTYPE) op.z; return *this;}
Vector3DF &Vector3DF::operator+= (const Vector3DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; z+=(VTYPE) op.z; return *this;}
Vector3DF &Vector3DF::operator+= (const Vector4DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; z+=(VTYPE) op.z; return *this;}

Vector3DF &Vector3DF::operator-= (const int op) {x-= (VTYPE) op; y-= (VTYPE) op; z-= (VTYPE) op; return *this;}
Vector3DF &Vector3DF::operator-= (const double op) {x-= (VTYPE) op; y-= (VTYPE) op; z-= (VTYPE) op; return *this;}
Vector3DF &Vector3DF::operator-= (const Vector2DC &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
Vector3DF &Vector3DF::operator-= (const Vector2DI &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
Vector3DF &Vector3DF::operator-= (const Vector2DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
Vector3DF &Vector3DF::operator-= (const Vector3DC &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; z-=(VTYPE) op.z; return *this;}
Vector3DF &Vector3DF::operator-= (const Vector3DI &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; z-=(VTYPE) op.z; return *this;}
Vector3DF &Vector3DF::operator-= (const Vector3DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; z-=(VTYPE) op.z; return *this;}
Vector3DF &Vector3DF::operator-= (const Vector4DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; z-=(VTYPE) op.z; return *this;}
	
Vector3DF &Vector3DF::operator*= (const int op) {x*= (VTYPE) op; y*= (VTYPE) op; z*= (VTYPE) op; return *this;}
Vector3DF &Vector3DF::operator*= (const double op) {x*= (VTYPE) op; y*= (VTYPE) op; z*= (VTYPE) op; return *this;}
Vector3DF &Vector3DF::operator*= (const Vector2DC &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
Vector3DF &Vector3DF::operator*= (const Vector2DI &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
Vector3DF &Vector3DF::operator*= (const Vector2DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
Vector3DF &Vector3DF::operator*= (const Vector3DC &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; z*=(VTYPE) op.z; return *this;}
Vector3DF &Vector3DF::operator*= (const Vector3DI &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; z*=(VTYPE) op.z; return *this;}
Vector3DF &Vector3DF::operator*= (const Vector3DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; z*=(VTYPE) op.z; return *this;}
Vector3DF &Vector3DF::operator*= (const Vector4DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; z*=(VTYPE) op.z; return *this;}

Vector3DF &Vector3DF::operator/= (const int op) {x/= (VTYPE) op; y/= (VTYPE) op; z/= (VTYPE) op; return *this;}
Vector3DF &Vector3DF::operator/= (const double op) {x/= (VTYPE) op; y/= (VTYPE) op; z/= (VTYPE) op; return *this;}
Vector3DF &Vector3DF::operator/= (const Vector2DC &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
Vector3DF &Vector3DF::operator/= (const Vector2DI &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
Vector3DF &Vector3DF::operator/= (const Vector2DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
Vector3DF &Vector3DF::operator/= (const Vector3DC &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; z/=(VTYPE) op.z; return *this;}
Vector3DF &Vector3DF::operator/= (const Vector3DI &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; z/=(VTYPE) op.z; return *this;}
Vector3DF &Vector3DF::operator/= (const Vector3DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; z/=(VTYPE) op.z; return *this;}
Vector3DF &Vector3DF::operator/= (const Vector4DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; z/=(VTYPE) op.z; return *this;}

Vector3DF &Vector3DF::Cross (const Vector3DC &v) {double ax = x, ay = y, az = z; x = (VTYPE) (ay * (double) v.z - az * (double) v.y); y = (VTYPE) (-ax * (double) v.z + az * (double) v.x); z = (VTYPE) (ax * (double) v.y - ay * (double) v.x); return *this;}
Vector3DF &Vector3DF::Cross (const Vector3DI &v) {double ax = x, ay = y, az = z; x = (VTYPE) (ay * (double) v.z - az * (double) v.y); y = (VTYPE) (-ax * (double) v.z + az * (double) v.x); z = (VTYPE) (ax * (double) v.y - ay * (double) v.x); return *this;}
Vector3DF &Vector3DF::Cross (const Vector3DF &v) {double ax = x, ay = y, az = z; x = (VTYPE) (ay * (double) v.z - az * (double) v.y); y = (VTYPE) (-ax * (double) v.z + az * (double) v.x); z = (VTYPE) (ax * (double) v.y - ay * (double) v.x); return *this;}
		
double Vector3DF::Dot(const Vector3DC &v)			{double dot; dot = (double) x*v.x + (double) y*v.y + (double) z*v.z; return dot;}
double Vector3DF::Dot(const Vector3DI &v)			{double dot; dot = (double) x*v.x + (double) y*v.y + (double) z*v.z; return dot;}
double Vector3DF::Dot(const Vector3DF &v)			{double dot; dot = (double) x*v.x + (double) y*v.y + (double) z*v.z; return dot;}

double Vector3DF::Dist (const Vector2DC &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector3DF::Dist (const Vector2DI &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector3DF::Dist (const Vector2DF &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector3DF::Dist (const Vector3DC &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector3DF::Dist (const Vector3DI &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector3DF::Dist (const Vector3DF &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector3DF::Dist (const Vector4DF &v)		{ double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}
double Vector3DF::DistSq (const Vector2DC &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z; return (a*a + b*b + c*c);}
double Vector3DF::DistSq (const Vector2DI &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z; return (a*a + b*b + c*c);}
double Vector3DF::DistSq (const Vector2DF &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z; return (a*a + b*b + c*c);}
double Vector3DF::DistSq (const Vector3DC &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z - (double) v.z; return (a*a + b*b + c*c);}
double Vector3DF::DistSq (const Vector3DI &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z - (double) v.z; return (a*a + b*b + c*c);}
double Vector3DF::DistSq (const Vector3DF &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z - (double) v.z; return (a*a + b*b + c*c);}
double Vector3DF::DistSq (const Vector4DF &v)		{ double a,b,c; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z - (double) v.z; return (a*a + b*b + c*c);}

Vector3DF &Vector3DF::Normalize (void) {
	double n = (double) x*x + (double) y*y + (double) z*z;
	if (n!=0.0) {
		n = sqrt(n);
		x /= (float) n; y /= (float) n; z /= (float) n;
	}
	return *this;
}
double Vector3DF::Length (void) { double n; n = (double) x*x + (double) y*y + (double) z*z; if (n != 0.0) return sqrt(n); return 0.0; }


#undef VTYPE
#undef VNAME

// Vector4DF Code Definition

#define VNAME		4DF
#define VTYPE		float

// Constructors/Destructors
Vector4DF::Vector4DF (const VTYPE xa, const VTYPE ya, const VTYPE za, const VTYPE wa) {x=xa; y=ya; z=za; w=wa;}
Vector4DF::Vector4DF (const Vector2DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0; w=(VTYPE) 0;}
Vector4DF::Vector4DF (const Vector2DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0; w=(VTYPE) 0;}
Vector4DF::Vector4DF (const Vector2DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0; w=(VTYPE) 0;}
Vector4DF::Vector4DF (const Vector3DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; w=(VTYPE) 0;}
Vector4DF::Vector4DF (const Vector3DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; w=(VTYPE) 0;}
Vector4DF::Vector4DF (const Vector3DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; w=(VTYPE) 0;}
Vector4DF::Vector4DF (const Vector4DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; w=(VTYPE) op.w;}

// Member Functions
Vector4DF &Vector4DF::operator= (const int op) {x= (VTYPE) op; y= (VTYPE) op; z= (VTYPE) op; w = (VTYPE) op; return *this;}
Vector4DF &Vector4DF::operator= (const double op) {x= (VTYPE) op; y= (VTYPE) op; z= (VTYPE) op; w = (VTYPE) op; return *this;}
Vector4DF &Vector4DF::operator= (const Vector2DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0; w=(VTYPE) 0; return *this;}
Vector4DF &Vector4DF::operator= (const Vector2DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0; w=(VTYPE) 0;  return *this;}
Vector4DF &Vector4DF::operator= (const Vector2DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) 0; w=(VTYPE) 0;  return *this;}
Vector4DF &Vector4DF::operator= (const Vector3DC &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; w=(VTYPE) 0;  return *this;}
Vector4DF &Vector4DF::operator= (const Vector3DI &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; w=(VTYPE) 0;  return *this;}
Vector4DF &Vector4DF::operator= (const Vector3DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; w=(VTYPE) 0; return *this;}
Vector4DF &Vector4DF::operator= (const Vector4DF &op) {x=(VTYPE) op.x; y=(VTYPE) op.y; z=(VTYPE) op.z; w=(VTYPE) op.w; return *this;}	
	
Vector4DF &Vector4DF::operator+= (const int op) {x+= (VTYPE) op; y+= (VTYPE) op; z+= (VTYPE) op; w += (VTYPE) op; return *this;}
Vector4DF &Vector4DF::operator+= (const float op) {x+= (VTYPE) op; y+= (VTYPE) op; z+= (VTYPE) op; w += (VTYPE) op; return *this;}
Vector4DF &Vector4DF::operator+= (const double op) {x+= (VTYPE) op; y+= (VTYPE) op; z+= (VTYPE) op; w += (VTYPE) op; return *this;}
Vector4DF &Vector4DF::operator+= (const Vector2DC &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
Vector4DF &Vector4DF::operator+= (const Vector2DI &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
Vector4DF &Vector4DF::operator+= (const Vector2DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; return *this;}
Vector4DF &Vector4DF::operator+= (const Vector3DC &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; z+=(VTYPE) op.z; return *this;}
Vector4DF &Vector4DF::operator+= (const Vector3DI &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; z+=(VTYPE) op.z; return *this;}
Vector4DF &Vector4DF::operator+= (const Vector3DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; z+=(VTYPE) op.z; return *this;}
Vector4DF &Vector4DF::operator+= (const Vector4DF &op) {x+=(VTYPE) op.x; y+=(VTYPE) op.y; z+=(VTYPE) op.z; w+=(VTYPE) op.w; return *this;}	

Vector4DF &Vector4DF::operator-= (const int op) {x-= (VTYPE) op; y-= (VTYPE) op; z-= (VTYPE) op; w -= (VTYPE) op; return *this;}
Vector4DF &Vector4DF::operator-= (const double op) {x-= (VTYPE) op; y-= (VTYPE) op; z-= (VTYPE) op; w -= (VTYPE) op; return *this;}
Vector4DF &Vector4DF::operator-= (const Vector2DC &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
Vector4DF &Vector4DF::operator-= (const Vector2DI &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
Vector4DF &Vector4DF::operator-= (const Vector2DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; return *this;}
Vector4DF &Vector4DF::operator-= (const Vector3DC &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; z-=(VTYPE) op.z; return *this;}
Vector4DF &Vector4DF::operator-= (const Vector3DI &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; z-=(VTYPE) op.z; return *this;}
Vector4DF &Vector4DF::operator-= (const Vector3DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; z-=(VTYPE) op.z; return *this;}
Vector4DF &Vector4DF::operator-= (const Vector4DF &op) {x-=(VTYPE) op.x; y-=(VTYPE) op.y; z-=(VTYPE) op.z; w-=(VTYPE) op.w; return *this;}	

Vector4DF &Vector4DF::operator*= (const int op) {x*= (VTYPE) op; y*= (VTYPE) op; z*= (VTYPE) op; w *= (VTYPE) op; return *this;}
Vector4DF &Vector4DF::operator*= (const double op) {x*= (VTYPE) op; y*= (VTYPE) op; z*= (VTYPE) op; w *= (VTYPE) op; return *this;}
Vector4DF &Vector4DF::operator*= (const Vector2DC &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
Vector4DF &Vector4DF::operator*= (const Vector2DI &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
Vector4DF &Vector4DF::operator*= (const Vector2DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; return *this;}
Vector4DF &Vector4DF::operator*= (const Vector3DC &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; z*=(VTYPE) op.z; return *this;}
Vector4DF &Vector4DF::operator*= (const Vector3DI &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; z*=(VTYPE) op.z; return *this;}
Vector4DF &Vector4DF::operator*= (const Vector3DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; z*=(VTYPE) op.z; return *this;}
Vector4DF &Vector4DF::operator*= (const Vector4DF &op) {x*=(VTYPE) op.x; y*=(VTYPE) op.y; z*=(VTYPE) op.z; w*=(VTYPE) op.w; return *this;}	

Vector4DF &Vector4DF::operator/= (const int op) {x/= (VTYPE) op; y/= (VTYPE) op; z/= (VTYPE) op; w /= (VTYPE) op; return *this;}
Vector4DF &Vector4DF::operator/= (const double op) {x/= (VTYPE) op; y/= (VTYPE) op; z/= (VTYPE) op; w /= (VTYPE) op; return *this;}
Vector4DF &Vector4DF::operator/= (const Vector2DC &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
Vector4DF &Vector4DF::operator/= (const Vector2DI &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
Vector4DF &Vector4DF::operator/= (const Vector2DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; return *this;}
Vector4DF &Vector4DF::operator/= (const Vector3DC &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; z/=(VTYPE) op.z; return *this;}
Vector4DF &Vector4DF::operator/= (const Vector3DI &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; z/=(VTYPE) op.z; return *this;}
Vector4DF &Vector4DF::operator/= (const Vector3DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; z/=(VTYPE) op.z; return *this;}
Vector4DF &Vector4DF::operator/= (const Vector4DF &op) {x/=(VTYPE) op.x; y/=(VTYPE) op.y; z/=(VTYPE) op.z; w/=(VTYPE) op.w; return *this;}	

Vector4DF &Vector4DF::Cross (const Vector4DF &v) {double ax = x, ay = y, az = z, aw = w; x = (VTYPE) (ay * (double) v.z - az * (double) v.y); y = (VTYPE) (-ax * (double) v.z + az * (double) v.x); z = (VTYPE) (ax * (double) v.y - ay * (double) v.x); w = (VTYPE) 0; return *this;}
		
double Vector4DF::Dot(const Vector4DF &v)			{double dot; dot = (double) x*v.x + (double) y*v.y + (double) z*v.z + (double) w*v.w; return dot;}

double Vector4DF::Dist (const Vector4DF &v)		{double distsq = DistSq (v); if (distsq!=0) return sqrt(distsq); return 0.0;}

double Vector4DF::DistSq (const Vector4DF &v)		{double a,b,c,d; a = (double) x - (double) v.x; b = (double) y - (double) v.y; c = (double) z - (double) v.z; d = (double) w - (double) v.w; return (a*a + b*b + c*c + d*d);}

Vector4DF &Vector4DF::Normalize (void) {
	double n = (double) x*x + (double) y*y + (double) z*z + (double) w*w;
	if (n!=0.0) {
		n = sqrt(n);
		x /= (float) n; y /= (float) n; z /= (float) n; w /= (float) n;
	}
	return *this;
}
double Vector4DF::Length (void) { double n; n = (double) x*x + (double) y*y + (double) z*z + (double) w*w; if (n != 0.0) return sqrt(n); return 0.0; }

#undef VTYPE
#undef VNAME

// MatrixC Code Definition
#define VNAME		C
#define VTYPE		unsigned char

// Constructors/Destructors

MatrixC::MatrixC (void) {data = NULL; Resize (0,0);}
MatrixC::~MatrixC (void) {if (data!=NULL) delete[] data;}
MatrixC::MatrixC (int r, int c) {data = NULL; Resize (c,r);}

// Member Functions


VTYPE &MatrixC::operator () (int c, int r)
{
	#ifdef DEBUG_MATRIX
		if (data==NULL) Error.Print ( ErrorLev::Matrix, ErrorDef::MatrixIsNull, true );		
		if (r<0 || r>=rows) Error.Print ( ErrorLev::Matrix, ErrorDef::RowOutOfBounds, true );	
		if (c<0 || c>=cols) Error.Print ( ErrorLev::Matrix, ErrorDef::ColOutOfBounds, true );	
	#endif
	return *(data + (r*cols+c));
}
MatrixC &MatrixC::operator= (unsigned char op)	{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ = (VTYPE) op; return *this;}
MatrixC &MatrixC::operator= (int op)				{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ = (VTYPE) op; return *this;}
MatrixC &MatrixC::operator= (double op)			{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ = (VTYPE) op; return *this;}
MatrixC &MatrixC::operator= (MatrixC &op)			{
	#ifdef DEBUG_MATRIX		
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixC::m=op: Operand matrix (op) data is null\n");        
    #endif
	if (rows!=op.rows || cols!=op.cols || data==NULL) Resize (op.cols, op.rows);		
	memcpy (data, op.data, len);	// Use only for matricies of like types
    return *this;
}
MatrixC &MatrixC::operator= (MatrixI &op)			{
	#ifdef DEBUG_MATRIX		
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixC::m=op: Operand matrix (op) data is null\n");        
    #endif
	if (rows!=op.rows || cols!=op.cols || data==NULL) Resize (op.cols, op.rows);		
	VTYPE *n, *ne;
	int *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne;) *n++ = (VTYPE) *b++;
	//memcpy (data, op.data, len);	// Use only for matricies of like types
    return *this;
}
MatrixC &MatrixC::operator= (MatrixF &op)			{
	#ifdef DEBUG_MATRIX		
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixC::m=op: Operand matrix (op) data is null\n");        
    #endif
	if (rows!=op.rows || cols!=op.cols || data==NULL) Resize (op.cols, op.rows);		
	VTYPE *n, *ne;
	double *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne; n++, b++) {		
		if (*b>255) {
			*n = (VTYPE) 255;
		} else if (*b<=0) {
			*n = (VTYPE) 0;
		} else {
			*n = (VTYPE) *b;	
		}
	}
    return *this;
}

MatrixC &MatrixC::operator+= (unsigned char op)	{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ += (VTYPE) op; return *this;}
MatrixC &MatrixC::operator+= (int op)				{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ += (VTYPE) op; return *this;}
MatrixC &MatrixC::operator+= (double op)			{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ += (VTYPE) op; return *this;}
MatrixC &MatrixC::operator+= (MatrixC &op)		{
	#ifdef DEBUG_MATRIX
		if (data==NULL)							Debug.Print (DEBUG_MATRIX, "MatrixC::m+=op: Matrix data is null\n");
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixC::m+=op: Operand matrix (op) data is null\n");
		if (rows!=op.rows || cols!=op.cols)		Debug.Print (DEBUG_MATRIX, "MatrixC::m+=op: Matricies must be same size\n");
    #endif	 
	VTYPE *n, *ne;
	unsigned char *b;	
	n = data; ne = data + len; b = op.data;
	for (; n<ne; n++)
		*n++ +=  *b++;
    return *this;
}
MatrixC &MatrixC::operator+= (MatrixI &op)		{
	#ifdef DEBUG_MATRIX
		if (data==NULL)							Debug.Print (DEBUG_MATRIX, "MatrixC::m+=op: Matrix data is null\n");
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixC::m+=op: Operand matrix (op) data is null\n");
		if (rows!=op.rows || cols!=op.cols)		Debug.Print (DEBUG_MATRIX, "MatrixC::m+=op: Matricies must be same size\n");
    #endif	
	VTYPE *n, *ne;
	int *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne;) *n++ += (VTYPE) *b++;
    return *this;
}
MatrixC &MatrixC::operator+= (MatrixF &op)		{
	#ifdef DEBUG_MATRIX
		if (data==NULL)							Debug.Print (DEBUG_MATRIX, "MatrixC::m+=op: Matrix data is null\n");
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixC::m+=op: Operand matrix (op) data is null\n");
        if (rows!=op.rows || cols!=op.cols)		Debug.Print (DEBUG_MATRIX, "MatrixC::m+=op: Matricies must be the same size\n");		
    #endif
	VTYPE *n, *ne;
	double *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne;) *n++ += (VTYPE) *b++;
	return *this;
}

MatrixC &MatrixC::operator-= (unsigned char op)	{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ -= (VTYPE) op; return *this;}
MatrixC &MatrixC::operator-= (int op)				{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ -= (VTYPE) op; return *this;}
MatrixC &MatrixC::operator-= (double op)			{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ -= (VTYPE) op; return *this;}
MatrixC &MatrixC::operator-= (MatrixC &op)		{
	#ifdef DEBUG_MATRIX
		if (data==NULL)							Debug.Print (DEBUG_MATRIX, "MatrixC::m-=op: Matrix data is null\n");
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixC::m-=op: Operand matrix (op) data is null\n");
		if (rows!=op.rows || cols!=op.cols)		Debug.Print (DEBUG_MATRIX, "MatrixC::m-=op: Matricies must be same size\n");
    #endif	 
	VTYPE *n, *ne;
	unsigned char *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne;) *n++ -= (VTYPE) *b++;	
    return *this;
}
MatrixC &MatrixC::operator-= (MatrixI &op)		{
	#ifdef DEBUG_MATRIX
		if (data==NULL)							Debug.Print (DEBUG_MATRIX, "MatrixC::m-=op: Matrix data is null\n");
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixC::m-=op: Operand matrix (op) data is null\n");
		if (rows!=op.rows || cols!=op.cols)		Debug.Print (DEBUG_MATRIX, "MatrixC::m-=op: Matricies must be same size\n");
    #endif	
	VTYPE *n, *ne;
	int *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne;) *n++ -= (VTYPE) *b++;
    return *this;
}
MatrixC &MatrixC::operator-= (MatrixF &op)		{
	#ifdef DEBUG_MATRIX
		if (data==NULL)							Debug.Print (DEBUG_MATRIX, "MatrixC::m-=op: Matrix data is null\n");
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixC::m-=op: Operand matrix (op) data is null\n");
        if (rows!=op.rows || cols!=op.cols)		Debug.Print (DEBUG_MATRIX, "MatrixC::m-=op: Matricies must be the same size\n");		
    #endif
	VTYPE *n, *ne;
	double *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne;) *n++ -= (VTYPE) *b++;
	return *this;
}

MatrixC &MatrixC::operator*= (unsigned char op)	{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ *= (VTYPE) op; return *this;}
MatrixC &MatrixC::operator*= (int op)				{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ *= (VTYPE) op; return *this;}
MatrixC &MatrixC::operator*= (double op)			{
	VTYPE *n = data, *nlen = data + len; 
	for (;n<nlen;) *n++ *= (VTYPE) op; 
	return *this;
}
MatrixC &MatrixC::operator*= (MatrixC &op)		{
	#ifdef DEBUG_MATRIX
		if (data==NULL)							Debug.Print (DEBUG_MATRIX, "MatrixC::m*=op: Matrix data is null\n");
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixC::m*=op: Operand matrix (op) data is null\n");
		if (rows!=op.rows || cols!=op.cols)		Debug.Print (DEBUG_MATRIX, "MatrixC::m*=op: Matricies must be same size\n");
    #endif	 
	VTYPE *n, *ne;
	unsigned char *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne;) *n++ *= (VTYPE) *b++;	
    return *this;
}
MatrixC &MatrixC::operator*= (MatrixI &op)		{
	#ifdef DEBUG_MATRIX
		if (data==NULL)							Debug.Print (DEBUG_MATRIX, "MatrixC::m*=op: Matrix data is null\n");
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixC::m*=op: Operand matrix (op) data is null\n");
		if (rows!=op.rows || cols!=op.cols)		Debug.Print (DEBUG_MATRIX, "MatrixC::m*=op: Matricies must be same size\n");
    #endif	
	VTYPE *n, *ne;
	int *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne;) *n++ *= (VTYPE) *b++;
    return *this;
}
MatrixC &MatrixC::operator*= (MatrixF &op)		{
	#ifdef DEBUG_MATRIX
		if (data==NULL)							Debug.Print (DEBUG_MATRIX, "MatrixC::m*=op: Matrix data is null\n");
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixC::m*=op: Operand matrix (op) data is null\n");
        if (rows!=op.rows || cols!=op.cols)		Debug.Print (DEBUG_MATRIX, "MatrixC::m*=op: Matricies must be the same size\n");		
    #endif
	VTYPE *n, *ne;
	double *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne;) *n++ *= (VTYPE) *b++;
	return *this;
}

MatrixC &MatrixC::operator/= (unsigned char op)	{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ /= (VTYPE) op; return *this;}
MatrixC &MatrixC::operator/= (int op)				{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ /= (VTYPE) op; return *this;}
MatrixC &MatrixC::operator/= (double op)			{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ /= (VTYPE) op; return *this;}
MatrixC &MatrixC::operator/= (MatrixC &op)		{
	#ifdef DEBUG_MATRIX
		if (data==NULL)							Debug.Print (DEBUG_MATRIX, "MatrixC::m/=op: Matrix data is null\n");
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixC::m/=op: Operand matrix (op) data is null\n");
		if (rows!=op.rows || cols!=op.cols)		Debug.Print (DEBUG_MATRIX, "MatrixC::m/=op: Matricies must be same size\n");
    #endif	 
	VTYPE *n, *ne;
	unsigned char *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne;) if (*b!=(VTYPE) 0) {*n++ /= (VTYPE) *b++;} else {*n++ = (VTYPE) 0; b++;}
    return *this;
}
MatrixC &MatrixC::operator/= (MatrixI &op)		{
	#ifdef DEBUG_MATRIX
		if (data==NULL)							Debug.Print (DEBUG_MATRIX, "MatrixC::m/=op: Matrix data is null\n");
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixC::m/=op: Operand matrix (op) data is null\n");
		if (rows!=op.rows || cols!=op.cols)		Debug.Print (DEBUG_MATRIX, "MatrixC::m/=op: Matricies must be same size\n");
    #endif	
	VTYPE *n, *ne;
	int *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne;)  if (*b!=(VTYPE) 0) {*n++ /= (VTYPE) *b++;} else {*n++ = (VTYPE) 0; b++;}
    return *this;
}
MatrixC &MatrixC::operator/= (MatrixF &op)		{
	#ifdef DEBUG_MATRIX
		if (data==NULL)							Debug.Print (DEBUG_MATRIX, "MatrixC::m/=op: Matrix data is null\n");
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixC::m/=op: Operand matrix (op) data is null\n");
        if (rows!=op.rows || cols!=op.cols)		Debug.Print (DEBUG_MATRIX, "MatrixC::m/=op: Matricies must be the same size\n");		
    #endif
	VTYPE *n, *ne;
	double *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne;)  if (*b!=(VTYPE) 0) {*n++ /= (VTYPE) *b++;} else {*n++ = (VTYPE) 0; b++;}
	return *this;
}

MatrixC &MatrixC::Multiply (MatrixF &op) {
	#ifdef DEBUG_MATRIX 
		if (data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixC::m mult op: Matrix data is null\n");
		if (op.data==NULL)					Debug.Print (DEBUG_MATRIX, "MatrixC::m mult op: Operand matrix (op) data is null\n");
        if (cols!=op.rows)					Debug.Print (DEBUG_MATRIX, "MatrixC::m mult op: Matricies not compatible (m.cols != op.rows)\n");
    #endif
	if (cols==op.rows) {
		VTYPE *newdata, *n, *ne, *a, *as;		// Pointers into A and new A matricies
		double *b, *bs, *bce, *be;				// Pointers into B matrix
		int newr = rows, newc = op.cols;		// Set new rows and columns
		int newlen = newr * newc;				// Determine new matrix size
		newdata = new VTYPE[newlen];			// Allocate new matrix to hold multiplication
//		if (newdata==NULL)						{debug.Print ( (char*) "MatrixF::m*=op: Cannot allocate new matrix.\n"); exit(-1);}
		ne = newdata + newlen;					// Calculate end of new matrix
		int bskip = op.cols;					// Calculate row increment for B matrix	
		bce = op.data + bskip;					// Calculate end of first row in B matrix
		be = op.data + op.rows*op.cols;			// Calculate end of B matrix	
		as = data; bs = op.data;				// Goto start of A and B matricies
		for (n=newdata ;n<ne;) {				// Compute C = A*B		
			a = as; b = bs;						// Goto beginning of row in A, top of col in B
			*n = (VTYPE) 0;						// Initialize n element in C
			for (; b<be;) {*n += (VTYPE) ((*a++) * (*b)); b += bskip;}	// Compute n element in C
			if (++bs >= bce) {					// If last col in B..
				bs = op.data;					// Go back to first column in B
				as += cols;					// Goto next row in A
			}
			n++;								// Goto next element in C
		}	
		delete[] data;							// Destroy old A matrix
		data = newdata; rows = newr; cols = newc; len = newlen;		// Replace with new A matrix	
	}
	return *this;
}

MatrixC &MatrixC::Resize (int x, int y)
{
	if (data!=NULL) {
		if (rows!=y || cols!=x) {
			delete[] data;	
			len = (rows = y) * (cols = x); 
			data = new VTYPE[len];
		}
	} else {
		len = (rows = y) * (cols = x); data = new VTYPE[len];
	}
	#ifdef DEBUG_MATRIX
		if (data==NULL) Debug.Print (DEBUG_MATRIX, "MatrixC::Resize: Out of memory for construction.\n");
	#endif	
	#ifdef MATRIX_INITIALIZE
		memset (data, 0, sizeof(VTYPE)*len);		
	#endif		
	return *this;
}
MatrixC &MatrixC::ResizeSafe (int x, int y)
{
	VTYPE *newdata;
	int newlen;
	VTYPE *n, *ne;
	VTYPE *b, *be;
	int bskip;
		
			
	if (data!=NULL) {
		newlen = x*y;		
		newdata = new VTYPE[newlen];
		#ifdef DEBUG_MATRIX
			if (newdata==NULL)
				Debug.Print (DEBUG_MATRIX, "MatrixC::SizeSafe: Out of memory for construction.\n");
		#endif		
		if (y>=rows && x>=cols) {			// New size is larger (in both r and c)			
			memset (newdata, 0, newlen*sizeof(VTYPE));	// Clear new matrix
			ne = data + len;					// Calculate end of current matrix
			b = newdata;						// Start of new matrix
			be = newdata + cols;				// Last filled column+1 in new matrix
			bskip = x-cols;
			for (n = data; n<ne;) {				// Fill new matrix with old
				for (; b<be;) *b++ = *n++;
				b += bskip; 
				be += x;
			}
		} else if (y<rows && x<cols) {		// New size is smaller (in both r and c)
			ne = newdata + newlen;			// Calculate end of new matrix
			b = data;						// Start of old matrix
			be = data + x;					// Last retrieved column+1 in old matrix
			bskip = cols-x;
			for (n = newdata; n<ne;) {		// Fill new matrix with old
				for (; b<be;) *n++ = *b++;
				b += bskip;
				be += x;
			}
		} else {							// Asymetrical resize
			#ifdef DEBUG_MATRIX
				Debug.Print (DEBUG_MATRIX, "MatrixC::SizeSafe: Asymetrical resize NOT YET IMPLEMENTED.\n");
			#endif
			exit (202);
		}
		delete[] data;
		rows = y; cols = x;
		data = newdata; len = newlen;
	} else {
		len = (rows = y) * (cols = x);
		data = new VTYPE[len];
		#ifdef DEBUG_MATRIX
			if (data==NULL)
				Debug.Print (DEBUG_MATRIX, "MatrixC::SizeSafe: Out of memory for construction.\n");
		#endif
	}	
	return *this;
}
MatrixC &MatrixC::InsertRow (int r)
{
	VTYPE *newdata;
	VTYPE *r_src, *r_dest;
	int newlen;

	if (data!=NULL) {
		newlen = (rows+1)*cols;
		newdata = new VTYPE[newlen];
		#ifdef DEBUG_MATRIX
			if (newdata==NULL)
				Debug.Print (DEBUG_MATRIX, "MatrixC::InsertRow: Out of memory for construction.\n");
		#endif
		memcpy (newdata, data, r*cols*sizeof(VTYPE));
		if (r<rows) {
			r_src = data + r*cols;
			r_dest = newdata + (r+1)*cols;		
			if (r<rows) memcpy (r_dest, r_src, (rows-r)*cols*sizeof(VTYPE));		
		}
		r_dest = newdata + r*cols;
		memset (r_dest, 0, cols*sizeof(VTYPE));
		rows++;		
		delete[] data;
		data = newdata; len = newlen;	
	} else {		
		#ifdef DEBUG_MATRIX
			Debug.Print (DEBUG_MATRIX, "MatrixC::InsertRow: Cannot insert row in a null matrix.\n");
		#endif
	}
	return *this;
}
MatrixC &MatrixC::InsertCol (int c)
{
	VTYPE *newdata;
	int newlen;

	if (data!=NULL) {		
		newlen = rows*(cols+1);
		newdata = new VTYPE[newlen];
		#ifdef DEBUG_MATRIX
			if (newdata==NULL)
				Debug.Print (DEBUG_MATRIX, "MatrixC::InsertCol: Out of memory for construction.\n");
		#endif
		VTYPE *n, *ne;
		VTYPE *b, *be;
		int bskip, nskip;
		
		if (c>0) {				
			n = data;							// Copy columns to left of c
			ne = data + len;
			nskip = (cols-c);
			b = newdata;
			be = newdata + c;
			bskip = (cols-c)+1;
			for (; n<ne;) {
				for (; b<be; ) *b++ = *n++;
				b += bskip;
				be += (cols+1);
				n += nskip;
			}
		}
		if (c<cols) {
			n = data + c;						// Copy columns to right of c
			ne = data + len;
			nskip = c;
			b = newdata + (c+1);
			be = newdata + (cols+1);
			bskip = c+1;
			for (; n<ne;) {
				for (; b<be; ) *b++ = *n++;
				b += bskip;
				be += (cols+1);
				n += nskip;
			}
		}
		cols++;
		for (n=newdata+c, ne=newdata+len; n<ne; n+=cols) *n = (VTYPE) 0;
		delete[] data;
		data = newdata; len = newlen;					
	} else {		
		#ifdef DEBUG_MATRIX
			Debug.Print (DEBUG_MATRIX, "MatrixF::InsertCol: Cannot insert col in a null matrix.\n");
		#endif
	}
	return *this;
}
MatrixC &MatrixC::Transpose (void)
{
	VTYPE *newdata;
	int r = rows;
	
	if (data!=NULL) {
		if (rows==1) {
			rows = cols; cols = 1;
		} else if (cols==1) {
			cols = rows; rows = 1;	
		} else {		
			newdata = new VTYPE[len];
			#ifdef DEBUG_MATRIX
				if (newdata==NULL)
					Debug.Print (DEBUG_MATRIX, "MatrixF::Transpose: Out of memory for construction.\n");
			#endif	
			VTYPE *n, *ne;
			VTYPE *b, *be;			
			n = data;						// Goto start of old matrix
			ne = data + len;
			b = newdata;					// Goto start of new matrix
			be = newdata + len;					
			for (; n<ne; ) {				// Copy rows of old to columns of new
				for (; b<be; b+=r) *b  = *n++;
				b -= len;
				b++;
			}
		}		
		delete[] data;
		data = newdata;
		rows = cols; cols = r;
	} else {
		#ifdef DEBUG_MATRIX
			Debug.Print (DEBUG_MATRIX, "MatrixC::Transpose: Cannot transpose a null matrix.\n");
		#endif
	}
	return *this;
}
MatrixC &MatrixC::Identity (int order)
{
  	Resize (order, order);
	VTYPE *n, *ne;	
	memset (data, 0, len*sizeof(VTYPE));	// Fill matrix with zeros
	n = data;
	ne = data + len;
	for (; n<ne; ) {
		*n = 1;								// Set diagonal element to 1
		n+= cols;							
		n++;								// Next diagonal element
	}
	return *this;
}

MatrixC &MatrixC::Basis (Vector3DF &c1, Vector3DF &c2, Vector3DF &c3)
{
	Resize (4,4);
	VTYPE *n = data;	
	*n++ = (VTYPE) c1.x; *n++ = (VTYPE) c2.x; *n++ = (VTYPE) c3.x; *n++ = (VTYPE) 0;
	*n++ = (VTYPE) c1.y; *n++ = (VTYPE) c2.y; *n++ = (VTYPE) c3.y; *n++ = (VTYPE) 0;
	*n++ = (VTYPE) c1.z; *n++ = (VTYPE) c2.z; *n++ = (VTYPE) c3.z; *n++ = (VTYPE) 0;
	*n++ = (VTYPE) 0; *n++ = (VTYPE) 0; *n++ = (VTYPE) 0; *n++ = (VTYPE) 0;
	return *this;
}
MatrixC &MatrixC::GaussJordan (MatrixF &b)
{
	// Gauss-Jordan solves the matrix equation Ax = b
	// Given the problem:
	//		A*x = b		(where A is 'this' matrix and b is provided)
	// The solution is:
	//		Ainv*b = x
	// This function returns Ainv in A and x in b... that is:
	//		A (this) -> Ainv
	//		b -> solution x
	//

	#ifdef DEBUG_MATRIX
		Debug.Print (DEBUG_MATRIX, "MatrixC::GaussJordan: Not implemented for char matrix\n");
	#endif
	return *this;
}
int MatrixC::GetX()						{return cols;}
int MatrixC::GetY()						{return rows;}
int MatrixC::GetRows(void)				{return rows;}
int MatrixC::GetCols(void)				{return cols;}
int MatrixC::GetLength(void)				{return len;}
VTYPE *MatrixC::GetData(void)			{return data;}

double MatrixC::GetF (int r, int c)		{return (double) (*(data + r*cols + c));}

#undef VTYPE
#undef VNAME

// MatrixI Code Definition
#define VNAME		I
#define VTYPE		int

// Constructors/Destructors

MatrixI::MatrixI (void) {data = NULL; Resize (0,0);}
MatrixI::~MatrixI (void) {if (data!=NULL) delete[] data;}
MatrixI::MatrixI (int r, int c) {data = NULL; Resize (c,r);}

// Member Functions

VTYPE &MatrixI::operator () (int c, int r)
{
	#ifdef DEBUG_MATRIX
		if (data==NULL) Debug.Print (DEBUG_MATRIX, "MatrixI::op(): Matrix data is null\n");
		if (r<0 || r>=rows) Debug.Print (DEBUG_MATRIX, "MatrixI:op(): Row is out of bounds\n");
		if (c<0 || c>=cols) Debug.Print (DEBUG_MATRIX, "MatrixI:op(): Col is out of bounds\n");
	#endif
	return *(data + (r*cols+c));
}
MatrixI &MatrixI::operator= (unsigned char op)	{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ = (VTYPE) op; return *this;}
MatrixI &MatrixI::operator= (int op)				{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ = (VTYPE) op; return *this;}
MatrixI &MatrixI::operator= (double op)			{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ = (VTYPE) op; return *this;}
MatrixI &MatrixI::operator= (MatrixC &op)			{
	#ifdef DEBUG_MATRIX		
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixI::m=op: Operand matrix (op) data is null\n");        
    #endif
	if (rows!=op.rows || cols!=op.cols || data==NULL) Resize (op.cols, op.rows);		
	VTYPE *n, *ne;
	unsigned char *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne;) *n++ = (VTYPE) *b++;		
	// memcpy (data, op.data, len);	// Use only for matricies of like types
    return *this;
}
MatrixI &MatrixI::operator= (MatrixI &op)			{
	#ifdef DEBUG_MATRIX		
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixI::m=op: Operand matrix (op) data is null\n");        
    #endif
	if (rows!=op.rows || cols!=op.cols || data==NULL) Resize (op.cols, op.rows);		
	memcpy (data, op.data, len);	// Use only for matricies of like types
    return *this;
}
MatrixI &MatrixI::operator= (MatrixF &op)			{
	#ifdef DEBUG_MATRIX		
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixI::m=op: Operand matrix (op) data is null\n");        
    #endif
	if (rows!=op.rows || cols!=op.cols || data==NULL) Resize (op.cols, op.rows);		
	VTYPE *n, *ne;
	double *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne;) *n++ = (VTYPE) *b++;	
	//memcpy (data, op.data, len);	
    return *this;
}

MatrixI &MatrixI::operator+= (unsigned char op)	{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ += (VTYPE) op; return *this;}
MatrixI &MatrixI::operator+= (int op)				{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ += (VTYPE) op; return *this;}
MatrixI &MatrixI::operator+= (double op)			{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ += (VTYPE) op; return *this;}
MatrixI &MatrixI::operator+= (MatrixC &op)		{
	#ifdef DEBUG_MATRIX
		if (data==NULL)							Debug.Print (DEBUG_MATRIX, "MatrixI::m+=op: Matrix data is null\n");
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixI::m+=op: Operand matrix (op) data is null\n");
		if (rows!=op.rows || cols!=op.cols)		Debug.Print (DEBUG_MATRIX, "MatrixI::m+=op: Matricies must be same size\n");
    #endif	 
	VTYPE *n, *ne;
	unsigned char *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne;) *n++ += (VTYPE) *b++;	
    return *this;
}
MatrixI &MatrixI::operator+= (MatrixI &op)		{
	#ifdef DEBUG_MATRIX
		if (data==NULL)							Debug.Print (DEBUG_MATRIX, "MatrixI::m+=op: Matrix data is null\n");
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixI::m+=op: Operand matrix (op) data is null\n");
		if (rows!=op.rows || cols!=op.cols)		Debug.Print (DEBUG_MATRIX, "MatrixI::m+=op: Matricies must be same size\n");
    #endif	
	VTYPE *n, *ne;
	int *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne;) *n++ += (VTYPE) *b++;
    return *this;
}
MatrixI &MatrixI::operator+= (MatrixF &op)		{
	#ifdef DEBUG_MATRIX
		if (data==NULL)							Debug.Print (DEBUG_MATRIX, "MatrixI::m+=op: Matrix data is null\n");
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixI::m+=op: Operand matrix (op) data is null\n");
        if (rows!=op.rows || cols!=op.cols)		Debug.Print (DEBUG_MATRIX, "MatrixI::m+=op: Matricies must be the same size\n");		
    #endif
	VTYPE *n, *ne;
	double *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne;) *n++ += (VTYPE) *b++;
	return *this;
}

MatrixI &MatrixI::operator-= (unsigned char op)	{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ -= (VTYPE) op; return *this;}
MatrixI &MatrixI::operator-= (int op)				{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ -= (VTYPE) op; return *this;}
MatrixI &MatrixI::operator-= (double op)			{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ -= (VTYPE) op; return *this;}
MatrixI &MatrixI::operator-= (MatrixC &op)		{
	#ifdef DEBUG_MATRIX
		if (data==NULL)							Debug.Print (DEBUG_MATRIX, "MatrixI::m-=op: Matrix data is null\n");
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixI::m-=op: Operand matrix (op) data is null\n");
		if (rows!=op.rows || cols!=op.cols)		Debug.Print (DEBUG_MATRIX, "MatrixI::m-=op: Matricies must be same size\n");
    #endif	 
	VTYPE *n, *ne;
	unsigned char *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne;) *n++ -= (VTYPE) *b++;	
    return *this;
}
MatrixI &MatrixI::operator-= (MatrixI &op)		{
	#ifdef DEBUG_MATRIX
		if (data==NULL)							Debug.Print (DEBUG_MATRIX, "MatrixI::m-=op: Matrix data is null\n");
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixI::m-=op: Operand matrix (op) data is null\n");
		if (rows!=op.rows || cols!=op.cols)		Debug.Print (DEBUG_MATRIX, "MatrixI::m-=op: Matricies must be same size\n");
    #endif	
	VTYPE *n, *ne;
	int *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne;) *n++ -= (VTYPE) *b++;
    return *this;
}
MatrixI &MatrixI::operator-= (MatrixF &op)		{
	#ifdef DEBUG_MATRIX
		if (data==NULL)							Debug.Print (DEBUG_MATRIX, "MatrixI::m-=op: Matrix data is null\n");
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixI::m-=op: Operand matrix (op) data is null\n");
        if (rows!=op.rows || cols!=op.cols)		Debug.Print (DEBUG_MATRIX, "MatrixI::m-=op: Matricies must be the same size\n");		
    #endif
	VTYPE *n, *ne;
	double *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne;) *n++ -= (VTYPE) *b++;
	return *this;
}

MatrixI &MatrixI::operator*= (unsigned char op)	{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ *= (VTYPE) op; return *this;}
MatrixI &MatrixI::operator*= (int op)				{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ *= (VTYPE) op; return *this;}
MatrixI &MatrixI::operator*= (double op)			{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ *= (VTYPE) op; return *this;}
MatrixI &MatrixI::operator*= (MatrixC &op)		{
	#ifdef DEBUG_MATRIX
		if (data==NULL)							Debug.Print (DEBUG_MATRIX, "MatrixI::m*=op: Matrix data is null\n");
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixI::m*=op: Operand matrix (op) data is null\n");
		if (rows!=op.rows || cols!=op.cols)		Debug.Print (DEBUG_MATRIX, "MatrixI::m*=op: Matricies must be same size\n");
    #endif	 
	VTYPE *n, *ne;
	unsigned char *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne;) *n++ *= (VTYPE) *b++;	
    return *this;
}
MatrixI &MatrixI::operator*= (MatrixI &op)		{
	#ifdef DEBUG_MATRIX
		if (data==NULL)							Debug.Print (DEBUG_MATRIX, "MatrixI::m*=op: Matrix data is null\n");
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixI::m*=op: Operand matrix (op) data is null\n");
		if (rows!=op.rows || cols!=op.cols)		Debug.Print (DEBUG_MATRIX, "MatrixI::m*=op: Matricies must be same size\n");
    #endif	
	VTYPE *n, *ne;
	int *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne;) *n++ *= (VTYPE) *b++;
    return *this;
}
MatrixI &MatrixI::operator*= (MatrixF &op)		{
	#ifdef DEBUG_MATRIX
		if (data==NULL)							Debug.Print (DEBUG_MATRIX, "MatrixI::m*=op: Matrix data is null\n");
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixI::m*=op: Operand matrix (op) data is null\n");
        if (rows!=op.rows || cols!=op.cols)		Debug.Print (DEBUG_MATRIX, "MatrixI::m*=op: Matricies must be the same size\n");		
    #endif
	VTYPE *n, *ne;
	double *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne;) *n++ *= (VTYPE) *b++;
	return *this;
}

MatrixI &MatrixI::operator/= (unsigned char op)	{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ /= (VTYPE) op; return *this;}
MatrixI &MatrixI::operator/= (int op)				{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ /= (VTYPE) op; return *this;}
MatrixI &MatrixI::operator/= (double op)			{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ /= (VTYPE) op; return *this;}
MatrixI &MatrixI::operator/= (MatrixC &op)		{
	#ifdef DEBUG_MATRIX
		if (data==NULL)							Debug.Print (DEBUG_MATRIX, "MatrixI::m/=op: Matrix data is null\n");
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixI::m/=op: Operand matrix (op) data is null\n");
		if (rows!=op.rows || cols!=op.cols)		Debug.Print (DEBUG_MATRIX, "MatrixI::m/=op: Matricies must be same size\n");
    #endif	 
	VTYPE *n, *ne;
	unsigned char *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne;) if (*b!=(VTYPE) 0) {*n++ /= (VTYPE) *b++;} else {*n++ = (VTYPE) 0; b++;}	
    return *this;
}
MatrixI &MatrixI::operator/= (MatrixI &op)		{
	#ifdef DEBUG_MATRIX
		if (data==NULL)							Debug.Print (DEBUG_MATRIX, "MatrixI::m/=op: Matrix data is null\n");
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixI::m/=op: Operand matrix (op) data is null\n");
		if (rows!=op.rows || cols!=op.cols)		Debug.Print (DEBUG_MATRIX, "MatrixI::m/=op: Matricies must be same size\n");
    #endif	
	VTYPE *n, *ne;
	int *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne;) if (*b!=(VTYPE) 0) {*n++ /= (VTYPE) *b++;} else {*n++ = (VTYPE) 0; b++;}
    return *this;
}
MatrixI &MatrixI::operator/= (MatrixF &op)		{
	#ifdef DEBUG_MATRIX
		if (data==NULL)							Debug.Print (DEBUG_MATRIX, "MatrixI::m/=op: Matrix data is null\n");
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixI::m/=op: Operand matrix (op) data is null\n");
        if (rows!=op.rows || cols!=op.cols)		Debug.Print (DEBUG_MATRIX, "MatrixI::m/=op: Matricies must be the same size\n");		
    #endif
	VTYPE *n, *ne;
	double *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne;) if (*b!=(VTYPE) 0) {*n++ /= (VTYPE) *b++;} else {*n++ = (VTYPE) 0; b++;}
	return *this;
}

MatrixI &MatrixI::Multiply (MatrixF &op) {
	#ifdef DEBUG_MATRIX 
		if (data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixI::m mult op: Matrix data is null\n");
		if (op.data==NULL)					Debug.Print (DEBUG_MATRIX, "MatrixI::m mult op: Operand matrix (op) data is null\n");
        if (cols!=op.rows)					Debug.Print (DEBUG_MATRIX, "MatrixI::m mult op: Matricies not compatible (m.cols != op.rows)\n");
    #endif
	if (cols==op.rows) {
		VTYPE *newdata, *n, *ne, *a, *as;		// Pointers into A and new A matricies
		double *b, *bs, *bce, *be;				// Pointers into B matrix
		int newr = rows, newc = op.cols;		// Set new rows and columns
		int newlen = newr * newc;				// Determine new matrix size
		newdata = new VTYPE[newlen];			// Allocate new matrix to hold multiplication
//		if (newdata==NULL)						{debug.Print ((char*) "MatrixF::m*=op: Cannot allocate new matrix.\n"); exit(-1);}
		ne = newdata + newlen;					// Calculate end of new matrix
		int bskip = op.cols;					// Calculate row increment for B matrix	
		bce = op.data + bskip;					// Calculate end of first row in B matrix
		be = op.data + op.rows*op.cols;			// Calculate end of B matrix	
		as = data; bs = op.data;				// Goto start of A and B matricies
		for (n=newdata ;n<ne;) {				// Compute C = A*B		
			a = as; b = bs;						// Goto beginning of row in A, top of col in B
			*n = (VTYPE) 0;						// Initialize n element in C
			for (; b<be;) {*n += (VTYPE) ((*a++) * (*b)); b += bskip;}	// Compute n element in C
			if (++bs >= bce) {					// If last col in B..
				bs = op.data;					// Go back to first column in B
				as += cols;					// Goto next row in A
			}
			n++;								// Goto next element in C
		}	
		delete[] data;							// Destroy old A matrix
		data = newdata; rows = newr; cols = newc; len = newlen;		// Replace with new A matrix	
	}
	return *this;
}

MatrixI &MatrixI::Resize (int x, int y)
{
	if (data!=NULL) {
		if (rows!=y || cols!=x) {delete[] data;	len = (rows = y) * (cols = x); data = new VTYPE[len];}
	} else {
		len = (rows = y) * (cols = x); data = new VTYPE[len];
	}
	#ifdef DEBUG_MATRIX
		if (data==NULL) Debug.Print (DEBUG_MATRIX, "MatrixC::Size: Out of memory for construction.\n");
	#endif	
	#ifdef MATRIX_INITIALIZE
		memset (data, 0, sizeof(VTYPE)*len);		
	#endif		
	return *this;
}
MatrixI &MatrixI::ResizeSafe (int x, int y)
{
	VTYPE *newdata;
	int newlen;
	VTYPE *n, *ne;
	VTYPE *b, *be;
	int bskip;
		
			
	if (data!=NULL) {
		newlen = x*y;		
		newdata = new VTYPE[newlen];
		#ifdef DEBUG_MATRIX
			if (newdata==NULL)
				Debug.Print (DEBUG_MATRIX, "MatrixC::SizeSafe: Out of memory for construction.\n");
		#endif		
		if (y>=rows && x>=cols) {			// New size is larger (in both r and c)			
			memset (newdata, 0, newlen*sizeof(VTYPE));	// Clear new matrix
			ne = data + len;					// Calculate end of current matrix
			b = newdata;						// Start of new matrix
			be = newdata + cols;				// Last filled column+1 in new matrix
			bskip = x-cols;
			for (n = data; n<ne;) {				// Fill new matrix with old
				for (; b<be;) *b++ = *n++;
				b += bskip; 
				be += x;
			}
		} else if (y<rows && x<cols) {		// New size is smaller (in both r and c)
			ne = newdata + newlen;			// Calculate end of new matrix
			b = data;						// Start of old matrix
			be = data + x;					// Last retrieved column+1 in old matrix
			bskip = cols-x;
			for (n = newdata; n<ne;) {		// Fill new matrix with old
				for (; b<be;) *n++ = *b++;
				b += bskip;
				be += x;
			}
		} else {							// Asymetrical resize
			#ifdef DEBUG_MATRIX
				Debug.Print (DEBUG_MATRIX, "MatrixC::SizeSafe: Asymetrical resize NOT YET IMPLEMENTED.\n");
			#endif
			exit (202);
		}
		delete[] data;
		rows = y; cols = x;
		data = newdata; len = newlen;
	} else {
		len = (rows = y) * (cols = x);
		data = new VTYPE[len];
		#ifdef DEBUG_MATRIX
			if (data==NULL)
				Debug.Print (DEBUG_MATRIX, "MatrixC::SizeSafe: Out of memory for construction.\n");
		#endif
	}	
	return *this;
}
MatrixI &MatrixI::InsertRow (int r)
{
	VTYPE *newdata;
	VTYPE *r_src, *r_dest;
	int newlen;

	if (data!=NULL) {
		newlen = (rows+1)*cols;
		newdata = new VTYPE[newlen];
		#ifdef DEBUG_MATRIX
			if (newdata==NULL)
				Debug.Print (DEBUG_MATRIX, "MatrixC::InsertRow: Out of memory for construction.\n");
		#endif
		memcpy (newdata, data, r*cols*sizeof(VTYPE));
		if (r<rows) {
			r_src = data + r*cols;
			r_dest = newdata + (r+1)*cols;		
			if (r<rows) memcpy (r_dest, r_src, (rows-r)*cols*sizeof(VTYPE));		
		}
		r_dest = newdata + r*cols;
		memset (r_dest, 0, cols*sizeof(VTYPE));
		rows++;		
		delete[] data;
		data = newdata; len = newlen;	
	} else {		
		#ifdef DEBUG_MATRIX
			Debug.Print (DEBUG_MATRIX, "MatrixC::InsertRow: Cannot insert row in a null matrix.\n");
		#endif
	}
	return *this;
}
MatrixI &MatrixI::InsertCol (int c)
{
	VTYPE *newdata;
	int newlen;

	if (data!=NULL) {		
		newlen = rows*(cols+1);
		newdata = new VTYPE[newlen];
		#ifdef DEBUG_MATRIX
			if (newdata==NULL)
				Debug.Print (DEBUG_MATRIX, "MatrixC::InsertCol: Out of memory for construction.\n");
		#endif
		VTYPE *n, *ne;
		VTYPE *b, *be;
		int bskip, nskip;
		
		if (c>0) {				
			n = data;							// Copy columns to left of c
			ne = data + len;
			nskip = (cols-c);
			b = newdata;
			be = newdata + c;
			bskip = (cols-c)+1;
			for (; n<ne;) {
				for (; b<be; ) *b++ = *n++;
				b += bskip;
				be += (cols+1);
				n += nskip;
			}
		}
		if (c<cols) {
			n = data + c;						// Copy columns to right of c
			ne = data + len;
			nskip = c;
			b = newdata + (c+1);
			be = newdata + (cols+1);
			bskip = c+1;
			for (; n<ne;) {
				for (; b<be; ) *b++ = *n++;
				b += bskip;
				be += (cols+1);
				n += nskip;
			}
		}
		cols++;
		for (n=newdata+c, ne=newdata+len; n<ne; n+=cols) *n = (VTYPE) 0;		
		delete[] data;
		data = newdata; len = newlen;					
	} else {		
		#ifdef DEBUG_MATRIX
			Debug.Print (DEBUG_MATRIX, "MatrixI::InsertCol: Cannot insert col in a null matrix.\n");
		#endif
	}
	return *this;
}
MatrixI &MatrixI::Transpose (void)
{
	VTYPE *newdata;
	int r = rows;
	
	if (data!=NULL) {
		if (rows==1) {
			rows = cols; cols = 1;
		} else if (cols==1) {
			cols = rows; rows = 1;	
		} else {		
			newdata = new VTYPE[len];
			#ifdef DEBUG_MATRIX
				if (newdata==NULL)
					Debug.Print (DEBUG_MATRIX, "MatrixF::Transpose: Out of memory for construction.\n");
			#endif	
			VTYPE *n, *ne;
			VTYPE *b, *be;			
			n = data;						// Goto start of old matrix
			ne = data + len;
			b = newdata;					// Goto start of new matrix
			be = newdata + len;					
			for (; n<ne; ) {				// Copy rows of old to columns of new
				for (; b<be; b+=r) *b  = *n++;
				b -= len;
				b++;
			}
		}		
		delete[] data;
		data = newdata;
		rows = cols; cols = r;
	} else {
		#ifdef DEBUG_MATRIX
			Debug.Print (DEBUG_MATRIX, "MatrixC::Transpose: Cannot transpose a null matrix.\n");
		#endif
	}
	return *this;
}
MatrixI &MatrixI::Identity (int order)
{
  	Resize (order, order);
	VTYPE *n, *ne;	
	memset (data, 0, len*sizeof(VTYPE));	// Fill matrix with zeros
	n = data;
	ne = data + len;
	for (; n<ne; ) {
		*n = 1;								// Set diagonal element to 1
		n+= cols;							
		n++;								// Next diagonal element
	}
	return *this;
}
MatrixI &MatrixI::Basis (Vector3DF &c1, Vector3DF &c2, Vector3DF &c3)
{
	Resize (4,4);
	VTYPE *n = data;	
	*n++ = (VTYPE) c1.x; *n++ = (VTYPE) c2.x; *n++ = (VTYPE) c3.x; *n++ = (VTYPE) 0;
	*n++ = (VTYPE) c1.y; *n++ = (VTYPE) c2.y; *n++ = (VTYPE) c3.y; *n++ = (VTYPE) 0;
	*n++ = (VTYPE) c1.z; *n++ = (VTYPE) c2.z; *n++ = (VTYPE) c3.z; *n++ = (VTYPE) 0;
	*n++ = (VTYPE) 0; *n++ = (VTYPE) 0; *n++ = (VTYPE) 0; *n++ = (VTYPE) 0;
	return *this;
}
MatrixI &MatrixI::GaussJordan (MatrixF &b)
{
	// Gauss-Jordan solves the matrix equation Ax = b
	// Given the problem:
	//		A*x = b		(where A is 'this' matrix and b is provided)
	// The solution is:
	//		Ainv*b = x
	// This function returns Ainv in A and x in b... that is:
	//		A (this) -> Ainv
	//		b -> solution x
	//

	#ifdef DEBUG_MATRIX
		Debug.Print (DEBUG_MATRIX, "MatrixI::GaussJordan: Not implemented for int matrix\n");
	#endif	
	return *this;
}
int MatrixI::GetX()						{return cols;}
int MatrixI::GetY()						{return rows;}
int MatrixI::GetRows(void)				{return rows;}
int MatrixI::GetCols(void)				{return cols;}
int MatrixI::GetLength(void)			{return len;}
VTYPE *MatrixI::GetData(void)			{return data;}

double MatrixI::GetF (int r, int c)		{return (double) (*(data + r*cols + c));}

#undef VTYPE
#undef VNAME


// MatrixF Code Definition
#define VNAME		F
#define VTYPE		double

// Constructors/Destructors

MatrixF::MatrixF (void) {data = NULL; Resize (0,0);}

MatrixF::~MatrixF (void) {
	if (data!=NULL) 
		delete [] data;
}
MatrixF::MatrixF (const int r, const int c) {data = NULL; Resize (r,c);}

// Member Functions

VTYPE MatrixF::GetVal ( int c, int r )
{
	#ifdef DEBUG_MATRIX
		if (data==NULL) Error.Print ( ErrorLev::Matrix, ErrorDef::MatrixIsNull, true );		
		if (r<0 || r>=rows) Error.Print ( ErrorLev::Matrix, ErrorDef::RowOutOfBounds, true );	
		if (c<0 || c>=cols) Error.Print ( ErrorLev::Matrix, ErrorDef::ColOutOfBounds, true );	
	#endif
	return *(data + (r*cols+c));
}

VTYPE &MatrixF::operator () (const int c, const int r)
{
	#ifdef DEBUG_MATRIX
		if (data==NULL) 
			Error.Print ( ErrorLev::Matrix, ErrorDef::MatrixIsNull, true );		
		if (r<0 || r>=rows)
			Error.Print ( ErrorLev::Matrix, ErrorDef::RowOutOfBounds, true );	
		if (c<0 || c>=cols) 
			Error.Print ( ErrorLev::Matrix, ErrorDef::ColOutOfBounds, true );	
	#endif
	return *(data + (r*cols+c));
}
MatrixF &MatrixF::operator= (const unsigned char op)	{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ = (VTYPE) op; return *this;}
MatrixF &MatrixF::operator= (const int op)				{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ = (VTYPE) op; return *this;}
MatrixF &MatrixF::operator= (const double op)			{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ = (VTYPE) op; return *this;}
MatrixF &MatrixF::operator= (const MatrixC &op)			{
	#ifdef DEBUG_MATRIX		
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixF::m=op: Operand matrix (op) data is null\n");        
    #endif
	if (rows!=op.rows || cols!=op.cols || data==NULL) Resize (op.cols, op.rows);		
	VTYPE *n, *ne;
	unsigned char *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne;) *n++ = (VTYPE) *b++;	
	//memcpy (data, op.data, len*sizeof(VTYPE));	// Use only for matricies of like types
    return *this;
}
MatrixF &MatrixF::operator= (const MatrixI &op)			{
	#ifdef DEBUG_MATRIX		
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixF::m=op: Operand matrix (op) data is null\n");        
    #endif
	if (rows!=op.rows || cols!=op.cols || data==NULL) Resize (op.cols, op.rows);		
	VTYPE *n, *ne;
	int *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne;) *n++ = (VTYPE) *b++;
	//memcpy (data, op.data, len*sizeof(VTYPE));	// Use only for matricies of like types
    return *this;
}
MatrixF &MatrixF::operator= (const MatrixF &op)			{
	#ifdef DEBUG_MATRIX		
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixF::m=op: Operand matrix (op) data is null\n");        
    #endif
	if (rows!=op.rows || cols!=op.cols || data==NULL) Resize (op.cols, op.rows);		
	memcpy (data, op.data, len*sizeof(VTYPE));	
    return *this;
}

MatrixF &MatrixF::operator+= (const unsigned char op)	{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ += (VTYPE) op; return *this;}
MatrixF &MatrixF::operator+= (const int op)				{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ += (VTYPE) op; return *this;}
MatrixF &MatrixF::operator+= (const double op)			{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ += (VTYPE) op; return *this;}
MatrixF &MatrixF::operator+= (const MatrixC &op)		{
	#ifdef DEBUG_MATRIX
		if (data==NULL)							Debug.Print (DEBUG_MATRIX, "MatrixF::m+=op: Matrix data is null\n");
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixF::m+=op: Operand matrix (op) data is null\n");
		if (rows!=op.rows || cols!=op.cols)		Debug.Print (DEBUG_MATRIX, "MatrixF::m+=op: Matricies must be same size\n");
    #endif	 
	VTYPE *n, *ne;
	unsigned char *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne;) *n++ += (VTYPE) *b++;	
    return *this;
}
MatrixF &MatrixF::operator+= (const MatrixI &op)		{
	#ifdef DEBUG_MATRIX
		if (data==NULL)							Debug.Print (DEBUG_MATRIX, "MatrixF::m+=op: Matrix data is null\n");
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixF::m+=op: Operand matrix (op) data is null\n");
		if (rows!=op.rows || cols!=op.cols)		Debug.Print (DEBUG_MATRIX, "MatrixF::m+=op: Matricies must be same size\n");
    #endif	
	VTYPE *n, *ne;
	int *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne;) *n++ += (VTYPE) *b++;
    return *this;
}
MatrixF &MatrixF::operator+= (const MatrixF &op)		{
	#ifdef DEBUG_MATRIX
		if (data==NULL)							Debug.Print (DEBUG_MATRIX, "MatrixF::m+=op: Matrix data is null\n");
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixF::m+=op: Operand matrix (op) data is null\n");
        if (rows!=op.rows || cols!=op.cols)		Debug.Print (DEBUG_MATRIX, "MatrixF::m+=op: Matricies must be the same size\n");		
    #endif
	VTYPE *n, *ne;
	double *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne;) *n++ += (VTYPE) *b++;
	return *this;
}

MatrixF &MatrixF::operator-= (const unsigned char op)	{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ -= (VTYPE) op; return *this;}
MatrixF &MatrixF::operator-= (const int op)				{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ -= (VTYPE) op; return *this;}
MatrixF &MatrixF::operator-= (const double op)			{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ -= (VTYPE) op; return *this;}
MatrixF &MatrixF::operator-= (const MatrixC &op)		{
	#ifdef DEBUG_MATRIX
		if (data==NULL)							Debug.Print (DEBUG_MATRIX, "MatrixF::m-=op: Matrix data is null\n");
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixF::m-=op: Operand matrix (op) data is null\n");
		if (rows!=op.rows || cols!=op.cols)		Debug.Print (DEBUG_MATRIX, "MatrixF::m-=op: Matricies must be same size\n");
    #endif	 
	VTYPE *n, *ne;
	unsigned char *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne;) *n++ -= (VTYPE) *b++;	
    return *this;
}
MatrixF &MatrixF::operator-= (const MatrixI &op)		{
	#ifdef DEBUG_MATRIX
		if (data==NULL)							Debug.Print (DEBUG_MATRIX, "MatrixF::m-=op: Matrix data is null\n");
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixF::m-=op: Operand matrix (op) data is null\n");
		if (rows!=op.rows || cols!=op.cols)		Debug.Print (DEBUG_MATRIX, "MatrixF::m-=op: Matricies must be same size\n");
    #endif	
	VTYPE *n, *ne;
	int *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne;) *n++ -= (VTYPE) *b++;
    return *this;
}
MatrixF &MatrixF::operator-= (const MatrixF &op)		{
	#ifdef DEBUG_MATRIX
		if (data==NULL)							Debug.Print (DEBUG_MATRIX, "MatrixF::m-=op: Matrix data is null\n");
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixF::m-=op: Operand matrix (op) data is null\n");
        if (rows!=op.rows || cols!=op.cols)		Debug.Print (DEBUG_MATRIX, "MatrixF::m-=op: Matricies must be the same size\n");		
    #endif
	VTYPE *n, *ne;
	double *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne;) *n++ -= (VTYPE) *b++;
	return *this;
}

MatrixF &MatrixF::operator*= (const unsigned char op)	{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ *= (VTYPE) op; return *this;}
MatrixF &MatrixF::operator*= (const int op)				{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ *= (VTYPE) op; return *this;}
MatrixF &MatrixF::operator*= (const double op)			{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ *= (VTYPE) op; return *this;}
MatrixF &MatrixF::operator*= (const MatrixC &op)		{
	#ifdef DEBUG_MATRIX
		if (data==NULL)							Debug.Print (DEBUG_MATRIX, "MatrixF::m*=op: Matrix data is null\n");
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixF::m*=op: Operand matrix (op) data is null\n");
		if (rows!=op.rows || cols!=op.cols)		Debug.Print (DEBUG_MATRIX, "MatrixF::m*=op: Matricies must be same size\n");
    #endif	 
	VTYPE *n, *ne;
	unsigned char *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne;) *n++ *= (VTYPE) *b++;	
    return *this;
}
MatrixF &MatrixF::operator*= (const MatrixI &op)		{
	#ifdef DEBUG_MATRIX
		if (data==NULL)							Debug.Print (DEBUG_MATRIX, "MatrixF::m*=op: Matrix data is null\n");
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixF::m*=op: Operand matrix (op) data is null\n");
		if (rows!=op.rows || cols!=op.cols)		Debug.Print (DEBUG_MATRIX, "MatrixF::m*=op: Matricies must be same size\n");
    #endif	
	VTYPE *n, *ne;
	int *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne;) *n++ *= (VTYPE) *b++;
    return *this;
}
MatrixF &MatrixF::operator*= (const MatrixF &op)		{
	#ifdef DEBUG_MATRIX
		if (data==NULL)							Debug.Print (DEBUG_MATRIX, "MatrixF::m*=op: Matrix data is null\n");
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixF::m*=op: Operand matrix (op) data is null\n");
        if (rows!=op.rows || cols!=op.cols)		Debug.Print (DEBUG_MATRIX, "MatrixF::m*=op: Matricies must be the same size\n");		
    #endif
	VTYPE *n, *ne;
	double *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne;) *n++ *= (VTYPE) *b++;
	return *this;
}

MatrixF &MatrixF::operator/= (const unsigned char op)	{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ /= (VTYPE) op; return *this;}
MatrixF &MatrixF::operator/= (const int op)				{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ /= (VTYPE) op; return *this;}
MatrixF &MatrixF::operator/= (const double op)			{VTYPE *n = data, *nlen = data + len; for (;n<nlen;) *n++ /= (VTYPE) op; return *this;}
MatrixF &MatrixF::operator/= (const MatrixC &op)		{
	#ifdef DEBUG_MATRIX
		if (data==NULL)							Debug.Print (DEBUG_MATRIX, "MatrixF::m/=op: Matrix data is null\n");
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixF::m/=op: Operand matrix (op) data is null\n");
		if (rows!=op.rows || cols!=op.cols)		Debug.Print (DEBUG_MATRIX, "MatrixF::m/=op: Matricies must be same size\n");
    #endif	 
	VTYPE *n, *ne;
	unsigned char *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne;) if (*b!=(VTYPE) 0) {*n++ /= (VTYPE) *b++;} else {*n++ = (VTYPE) 0; b++;}	
    return *this;
}
MatrixF &MatrixF::operator/= (const MatrixI &op)		{
	#ifdef DEBUG_MATRIX
		if (data==NULL)							Debug.Print (DEBUG_MATRIX, "MatrixF::m/=op: Matrix data is null\n");
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixF::m/=op: Operand matrix (op) data is null\n");
		if (rows!=op.rows || cols!=op.cols)		Debug.Print (DEBUG_MATRIX, "MatrixF::m/=op: Matricies must be same size\n");
    #endif	
	VTYPE *n, *ne;
	int *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne;) if (*b!=(VTYPE) 0) {*n++ /= (VTYPE) *b++;} else {*n++ = (VTYPE) 0; b++;}
    return *this;
}
MatrixF &MatrixF::operator/= (const MatrixF &op)		{
	#ifdef DEBUG_MATRIX
		if (data==NULL)							Debug.Print (DEBUG_MATRIX, "MatrixF::m/=op: Matrix data is null\n");
		if (op.data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixF::m/=op: Operand matrix (op) data is null\n");
        if (rows!=op.rows || cols!=op.cols)		Debug.Print (DEBUG_MATRIX, "MatrixF::m/=op: Matricies must be the same size\n");		
    #endif
	VTYPE *n, *ne;
	double *b;
	n = data; ne = data + len; b = op.data;
	for (; n<ne;) 
		if (*b!=(VTYPE) 0) {
			*n++ /= (VTYPE) *b++;
		} else {
			*n++ = (VTYPE) 0; b++;
		}
	return *this;
}

MatrixF &MatrixF::Multiply (const MatrixF &op) {
	#ifdef DEBUG_MATRIX 
		if (data==NULL)						debug.Print (DEBUG_MATRIX, "MatrixF::m*=op: Matrix data is null\n");
		if (op.data==NULL)					debug.Print (DEBUG_MATRIX, "MatrixF::m*=op: Operand matrix (op) data is null\n");
        if (cols!=op.rows)					debug.Print (DEBUG_MATRIX, "MatrixF::m*=op: Matricies not compatible (m.cols != op.rows)\n");
    #endif
	if (cols==op.rows) {
		VTYPE *newdata, *n, *ne, *a, *as;		// Pointers into A and new A matricies
		double *b, *bs, *bce, *be;				// Pointers into B matrix
		int newr = rows, newc = op.cols;		// Set new rows and columns
		int newlen = newr * newc;				// Determine new matrix size
		newdata = new VTYPE[newlen];			// Allocate new matrix to hold multiplication
//		if (newdata==NULL)						{debug.Print ( (char*) "MatrixF::m*=op: Cannot allocate new matrix.\n"); exit(-1);}
		ne = newdata + newlen;					// Calculate end of new matrix
		int bskip = op.cols;					// Calculate row increment for B matrix	
		bce = op.data + bskip;					// Calculate end of first row in B matrix
		be = op.data + op.rows*op.cols;			// Calculate end of B matrix	
		as = data; bs = op.data;				// Goto start of A and B matricies
		for (n=newdata ;n<ne;) {				// Compute C = A*B		
			a = as; b = bs;						// Goto beginning of row in A, top of col in B
			*n = (VTYPE) 0;						// Initialize n element in C
			for (; b<be;) {*n += (*a++) * (*b); b += bskip;}	// Compute n element in C
			if (++bs >= bce) {					// If last col in B..
				bs = op.data;					// Go back to first column in B
				as += cols;					// Goto next row in A
			}
			n++;								// Goto next element in C
		}	
		delete[] data;							// Destroy old A matrix
		data = newdata; rows = newr; cols = newc; len = newlen;		// Replace with new A matrix	
	}
	return *this;
}

MatrixF &MatrixF::Multiply4x4 (const MatrixF &op) {
	#ifdef DEBUG_MATRIX 
		if (data==NULL)						Debug.Print (DEBUG_MATRIX, "MatrixF::Multiply4x4 m*=op: Matrix data is null\n");
		if (op.data==NULL)					Debug.Print (DEBUG_MATRIX, "MatrixF::Multiply4x4 m*=op: Operand matrix (op) data is null\n");
        if (rows!=4 || cols!=4)				Debug.Print (DEBUG_MATRIX, "MatrixF::Multiply4x4 m*=op: Matrix m is not 4x4");
		if (op.rows!=4 || op.cols!=4)		Debug.Print (DEBUG_MATRIX, "MatrixF::Multiply4x4 m*=op: Matrix op is not 4x4");
    #endif
	register double c1, c2, c3, c4;					// Temporary storage
	VTYPE *n, *a, *b1, *b2, *b3, *b4;
	a = data;	n = data; 
	b1 = op.data; b2 = op.data + 4; b3 = op.data + 8; b4 = op.data + 12;

	c1 = *a++;	c2 = *a++;	c3 = *a++; c4 = *a++;					// Calculate First Row
	*n++ = c1*(*b1++) + c2*(*b2++) + c3*(*b3++) + c4*(*b4++);	
	*n++ = c1*(*b1++) + c2*(*b2++) + c3*(*b3++) + c4*(*b4++);
	*n++ = c1*(*b1++) + c2*(*b2++) + c3*(*b3++) + c4*(*b4++);
	*n++ = c1*(*b1) + c2*(*b2) + c3*(*b3) + c4*(*b4);
	b1 -= 3 ; b2 -= 3; b3 -= 3; b4 -= 3;

	c1 = *a++;	c2 = *a++;	c3 = *a++; c4 = *a++;					// Calculate Second Row
	*n++ = c1*(*b1++) + c2*(*b2++) + c3*(*b3++) + c4*(*b4++);	
	*n++ = c1*(*b1++) + c2*(*b2++) + c3*(*b3++) + c4*(*b4++);
	*n++ = c1*(*b1++) + c2*(*b2++) + c3*(*b3++) + c4*(*b4++);
	*n++ = c1*(*b1) + c2*(*b2) + c3*(*b3) + c4*(*b4);
	b1 -= 3 ; b2 -= 3; b3 -= 3; b4 -= 3;

	c1 = *a++;	c2 = *a++;	c3 = *a++; c4 = *a++;					// Calculate Third Row
	*n++ = c1*(*b1++) + c2*(*b2++) + c3*(*b3++) + c4*(*b4++);	
	*n++ = c1*(*b1++) + c2*(*b2++) + c3*(*b3++) + c4*(*b4++);
	*n++ = c1*(*b1++) + c2*(*b2++) + c3*(*b3++) + c4*(*b4++);
	*n++ = c1*(*b1) + c2*(*b2) + c3*(*b3) + c4*(*b4);
	b1 -= 3 ; b2 -= 3; b3 -= 3; b4 -= 3;

	c1 = *a++;	c2 = *a++;	c3 = *a++; c4 = *a;						// Calculate Four Row
	*n++ = c1*(*b1++) + c2*(*b2++) + c3*(*b3++) + c4*(*b4++);	
	*n++ = c1*(*b1++) + c2*(*b2++) + c3*(*b3++) + c4*(*b4++);
	*n++ = c1*(*b1++) + c2*(*b2++) + c3*(*b3++) + c4*(*b4++);
	*n = c1*(*b1) + c2*(*b2) + c3*(*b3) + c4*(*b4);	

	return *this;
}


MatrixF &MatrixF::Resize (const int x, const int y)
{
	if (data!=NULL) {
		if (rows==y && cols==x) return *this;
		delete[] data;
	}
	rows = y; cols = x;
	if (y>0 && x>0) {
		len = rows * cols; 
		if (len!=0) {
			data = new VTYPE[len];
			#ifdef DEBUG_MATRIX
				if (data==NULL) Debug.Print (DEBUG_MATRIX, "MatrixF::Size: Out of memory for construction.\n");
			#endif
		}
	}	
	
	#ifdef MATRIX_INITIALIZE
		if (data!=NULL) memset (data, 0, sizeof(VTYPE)*len);		
	#endif		
	return *this;
}
MatrixF &MatrixF::ResizeSafe (const int x, const int y)
{
	VTYPE *newdata;
	int newlen;
	VTYPE *n, *ne;
	VTYPE *b, *be;
	int bskip;
					
	if (data!=NULL) {
		newlen = x*y;
		newdata = new VTYPE[newlen];
		#ifdef DEBUG_MATRIX
			if (newdata==NULL)
				Debug.Print (DEBUG_MATRIX, "MatrixF::SizeSafe: Out of memory for construction.\n");
		#endif		
		if (y>=rows && x>=cols) {			// New size is larger (in both r and c)			
			memset (newdata, 0, newlen*sizeof(VTYPE));	// Clear new matrix
			ne = data + len;					// Calculate end of current matrix
			b = newdata;						// Start of new matrix
			be = newdata + cols;				// Last filled column+1 in new matrix
			bskip = x-cols;
			for (n = data; n<ne;) {				// Fill new matrix with old
				for (; b<be;) *b++ = *n++;
				b += bskip; 
				be += x;
			}
		} else if (y<rows && x<cols) {		// New size is smaller (in both r and c)
			ne = newdata + newlen;			// Calculate end of new matrix
			b = data;						// Start of old matrix
			be = data + x;					// Last retrieved column+1 in old matrix
			bskip = cols-x;
			for (n = newdata; n<ne;) {		// Fill new matrix with old
				for (; b<be;) *n++ = *b++;
				b += bskip;
				be += x;
			}
		} else {							// Asymetrical resize
			#ifdef DEBUG_MATRIX
				Debug.Print (DEBUG_MATRIX, "MatrixF::SizeSafe: Asymetrical resize NOT YET IMPLEMENTED.\n");
			#endif
			exit (202);
		}
		delete[] data;
		rows = y; cols = x;
		data = newdata; len = newlen;
	} else {
		len = (rows = y) * (cols = x);
		data = new VTYPE[len];
		#ifdef DEBUG_MATRIX
			if (data==NULL)
				Debug.Print (DEBUG_MATRIX, "MatrixF::SizeSafe: Out of memory for construction.\n");
		#endif
	}	
	return *this;
}
MatrixF &MatrixF::InsertRow (const int r)
{
	VTYPE *newdata;
	VTYPE *r_src, *r_dest;
	int newlen;

	if (data!=NULL) {
		newlen = (rows+1)*cols;
		newdata = new VTYPE[newlen];
		#ifdef DEBUG_MATRIX
			if (newdata==NULL)
				Debug.Print (DEBUG_MATRIX, "MatrixF::InsertRow: Out of memory for construction.\n");
		#endif
		memcpy (newdata, data, r*cols*sizeof(VTYPE));
		if (r<rows) {
			r_src = data + r*cols;
			r_dest = newdata + (r+1)*cols;		
			if (r<rows) memcpy (r_dest, r_src, (rows-r)*cols*sizeof(VTYPE));		
		}
		r_dest = newdata + r*cols;
		memset (r_dest, 0, cols*sizeof(VTYPE));
		rows++;
		delete[] data;
		data = newdata; len = newlen;	
	} else {		
		#ifdef DEBUG_MATRIX
			Debug.Print (DEBUG_MATRIX, "MatrixF::InsertRow: Cannot insert row in a null matrix.\n");
		#endif
	}
	return *this;
}
MatrixF &MatrixF::InsertCol (const int c)
{
	VTYPE *newdata;
	int newlen;

	if (data!=NULL) {		
		newlen = rows*(cols+1);
		newdata = new VTYPE[newlen];
		#ifdef DEBUG_MATRIX
			if (newdata==NULL)
				Debug.Print (DEBUG_MATRIX, "MatrixF::InsertCol: Out of memory for construction.\n");
		#endif
		VTYPE *n, *ne;
		VTYPE *b, *be;
		int bskip, nskip;
		
		if (c>0) {				
			n = data;							// Copy columns to left of c
			ne = data + len;
			nskip = (cols-c);
			b = newdata;
			be = newdata + c;
			bskip = (cols-c)+1;
			for (; n<ne;) {
				for (; b<be; ) *b++ = *n++;
				b += bskip;
				be += (cols+1);
				n += nskip;
			}
		}
		if (c<cols) {
			n = data + c;						// Copy columns to right of c
			ne = data + len;
			nskip = c;
			b = newdata + (c+1);
			be = newdata + (cols+1);
			bskip = c+1;
			for (; n<ne;) {
				for (; b<be; ) *b++ = *n++;
				b += bskip;
				be += (cols+1);
				n += nskip;
			}
		}
		cols++;
		for (n=newdata+c, ne=newdata+len; n<ne; n+=cols) *n = (VTYPE) 0;		
		delete[] data;
		data = newdata; len = newlen;					
	} else {		
		#ifdef DEBUG_MATRIX
			Debug.Print (DEBUG_MATRIX, "MatrixF::InsertCol: Cannot insert col in a null matrix.\n");
		#endif
	}
	return *this;
}
MatrixF &MatrixF::Transpose (void)
{
	VTYPE *newdata;
	int r = rows;
	
	if (data!=NULL) {
		if (rows==1) {
			rows = cols; cols = 1;
		} else if (cols==1) {
			cols = rows; rows = 1;	
		} else {		
			newdata = new VTYPE[len];
			#ifdef DEBUG_MATRIX
				if (newdata==NULL)
					Debug.Print (DEBUG_MATRIX, "MatrixF::Transpose: Out of memory for construction.\n");
			#endif	
			VTYPE *n, *ne;
			VTYPE *b, *be;			
			n = data;						// Goto start of old matrix
			ne = data + len;
			b = newdata;					// Goto start of new matrix
			be = newdata + len;					
			for (; n<ne; ) {				// Copy rows of old to columns of new
				for (; b<be; b+=r) *b  = *n++;
				b -= len;
				b++;
			}
		}		
		delete[] data;
		data = newdata;
		rows = cols; cols = r;
	} else {
		#ifdef DEBUG_MATRIX
			Debug.Print (DEBUG_MATRIX, "MatrixF::Transpose: Cannot transpose a null matrix.\n");
		#endif
	}
	return *this;
}
MatrixF &MatrixF::Identity (const int order)
{
 	Resize (order, order);
	VTYPE *n, *ne;	
	memset (data, 0, len*sizeof(VTYPE));	// Fill matrix with zeros
	n = data;
	ne = data + len;
	for (; n<ne; ) {
		*n = 1;								// Set diagonal element to 1
		n+= cols;							
		n++;								// Next diagonal element
	}
	return *this;
}

// rotates points >>counter-clockwise<< when looking down the X+ axis toward the origin
MatrixF &MatrixF::RotateX (const double ang)
{
	Resize (4,4);
	VTYPE *n = data;
	double c,s;
	c = cos(ang * 3.141592/180);
	s = sin(ang * 3.141592/180);	
	*n = 1; n += 5;
	*n++ = (VTYPE) c;	*n = (VTYPE) s; n+=3;
	*n++ = (VTYPE) -s;	*n = (VTYPE) c; n+=5;
	*n = 1; 
	return *this;
}

// rotates points >>counter-clockwise<< when looking down the Y+ axis toward the origin
MatrixF &MatrixF::RotateY (const double ang)
{
	Resize (4,4);
	VTYPE *n = data;
	double c,s;
	c = cos(ang * 3.141592/180);
	s = sin(ang * 3.141592/180);	
	*n = (VTYPE) c;		n+=2;
	*n = (VTYPE) -s;	n+=3;
	*n = 1;				n+=3;
	*n = (VTYPE) s;		n+=2;
	*n = (VTYPE) c;		n+=5;
	*n = 1;
	return *this;
}

// rotates points >>counter-clockwise<< when looking down the Z+ axis toward the origin
MatrixF &MatrixF::RotateZ (const double ang)
{
	Resize (4,4);		
	VTYPE *n = data;
	double c,s;
	c = cos(ang * 3.141592/180);
	s = sin(ang * 3.141592/180);
	*n++ = (VTYPE) c;	*n = (VTYPE) s; n+=3;
	*n++ = (VTYPE) -s;	*n = (VTYPE) c; n+=5; 
	*n = 1; n+=5; *n = 1;
	return *this;
}
MatrixF &MatrixF::Ortho (double sx, double sy, double vn, double vf)
{
	// simplified version of OpenGL's glOrtho function
	VTYPE *n = data;	
	*n++ = (VTYPE) (1.0/sx); *n++ = (VTYPE) 0.0; *n++ = (VTYPE) 0.0; *n++ = (VTYPE) 0.0;
	*n++ = (VTYPE) 0.0; *n++ = (VTYPE) (1.0/sy); *n++ = (VTYPE) 0.0; *n++ = (VTYPE) 0.0;
	*n++ = (VTYPE) 0.0; *n++ = (VTYPE) 0.0; *n++ = (VTYPE) (-2.0/(vf-vn)); *n++ = (VTYPE) (-(vf+vn)/(vf-vn));
	*n++ = (VTYPE) 0.0; *n++ = (VTYPE) 0.0; *n++ = (VTYPE) 0; *n++ = (VTYPE) 1.0;
	return *this;
}

MatrixF &MatrixF::Translate (double tx, double ty, double tz)
{
	Resize (4,4);
	VTYPE *n = data;	
	*n++ = (VTYPE) 1.0; *n++ = (VTYPE) 0.0; *n++ = (VTYPE) 0.0; *n++ = (VTYPE) 0.0;
	*n++ = (VTYPE) 0.0; *n++ = (VTYPE) 1.0; *n++ = (VTYPE) 0.0; *n++ = (VTYPE) 0.0;
	*n++ = (VTYPE) 0.0; *n++ = (VTYPE) 0.0; *n++ = (VTYPE) 1.0; *n++ = (VTYPE) 0.0;
	*n++ = (VTYPE) tx; *n++ = (VTYPE) ty; *n++ = (VTYPE) tz; *n++ = (VTYPE) 1.0;	
	return *this;
}

MatrixF &MatrixF::Basis (const Vector3DF &c1, const Vector3DF &c2, const Vector3DF &c3)
{
	Resize (4,4);
	VTYPE *n = data;	
	*n++ = (VTYPE) c1.x; *n++ = (VTYPE) c2.x; *n++ = (VTYPE) c3.x; *n++ = (VTYPE) 0;
	*n++ = (VTYPE) c1.y; *n++ = (VTYPE) c2.y; *n++ = (VTYPE) c3.y; *n++ = (VTYPE) 0;
	*n++ = (VTYPE) c1.z; *n++ = (VTYPE) c2.z; *n++ = (VTYPE) c3.z; *n++ = (VTYPE) 0;
	*n++ = (VTYPE) 0; *n++ = (VTYPE) 0; *n++ = (VTYPE) 0; *n++ = (VTYPE) 0;
	return *this;
}

#define		SWAP(a, b)		{temp=(a); (a)=(b); (b)=temp;}

MatrixF &MatrixF::GaussJordan (MatrixF &b)
{
	// Gauss-Jordan solves the matrix equation Ax = b
	// Given the problem:
	//		A*x = b		(where A is 'this' matrix and b is provided)
	// The solution is:
	//		Ainv*b = x
	// This function returns Ainv in A and x in b... that is:
	//		A (this) -> Ainv
	//		b -> solution x
	//
	
	MatrixI index_col, index_row;	
	MatrixI piv_flag;
	int r, c, c2, rs, cs;
	double piv_val;
	int piv_row, piv_col;
	double pivinv, dummy, temp;

	#ifdef DEBUG_MATRIX
		if (rows!=cols) Debug.Print (DEBUG_MATRIX, "MatrixF::GaussJordan: Number of rows and cols of A must be equal.\n");
		if (rows!=b.rows) Debug.Print (DEBUG_MATRIX, "MatrixF::GaussJordan: Number of rows of A and rows of b must be equal.\n");
		if (b.cols!=1) Debug.Print ( DEBUG_MATRIX, "MatrixF::GaussJordan: Number of cols of b must be 1.\n");
	#endif

	index_col.Resize (cols, 1);
	index_row.Resize (cols, 1);
	piv_flag.Resize (cols, 1);
	piv_flag = 0;
	for (c = 0; c < cols; c++) {
		piv_val = 0.0;		
		for (rs = 0; rs < rows; rs++) {
			if (piv_flag(rs, 0) != 1 )				
				for (cs = 0; cs < cols; cs++) {
					if (piv_flag(cs, 0) == 0) {
						if (fabs((*this) (cs, rs)) >= piv_val) {
							piv_val = fabs((*this) (cs, rs));
							piv_row = rs;
							piv_col = cs;
						}
					} else if (piv_flag(cs, 0)>1) {
						#ifdef DEBUG_MATRIX
							Debug.Print (DEBUG_MATRIX, "MatrixF::GaussJordan: Singular matrix (dbl pivs).\n");
							//Print ();
						#endif
					}
				}
		}
		piv_flag(piv_col, 0)++;
		if (piv_row != piv_col) {
			for (c2 = 0; c2 < cols; c2++) SWAP ((*this) (c2, piv_row), (*this) (c2, piv_col));
			for (c2 = 0; c2 < b.cols; c2++) SWAP (b(c2, piv_row), b(c2, piv_col));
		}
		index_row (c, 0) = piv_row;
		index_col (c, 0) = piv_col;
		if ((*this) (piv_col, piv_col) == 0.0) {
			#ifdef DEBUG_MATRIX
				Debug.Print (DEBUG_MATRIX, "MatrixF::GaussJordan: Singular matrix (0 piv).\n");
				//Print ();
			#endif
		}
		pivinv = 1.0 / ((*this) (piv_col, piv_col));
		(*this) (piv_col, piv_col) = 1.0;
		for (c2 = 0; c2 < cols; c2++) (*this) (c2, piv_col) *= pivinv;
		for (c2 = 0; c2 < b.cols; c2++) b(c2, piv_col) *= pivinv;
		for (r = 0; r < rows; r++) {
			if (r != piv_col) {
				dummy = (*this) (piv_col, r);
				(*this) (piv_col, r) = 0.0;
				for (c2 = 0; c2 < cols; c2++) (*this) (c2, r) -= (*this) (c2, piv_col)*dummy;
				for (c2 = 0; c2 < b.cols; c2++) b(c2, r) -= b(c2, piv_col)*dummy;
			}
		}
	}	
	for (c = cols-1; c >= 0; c--) {
		if (index_row(c, 0) != index_col(c, 0))
			for (r = 0; r < rows; r++)
				SWAP ((*this) (index_row(c,0), r), (*this) (index_col(c,0), r) );
	}
	return *this;
}
MatrixF &MatrixF::Submatrix ( MatrixF& b, int mx, int my)
{
	VTYPE* pEnd = data + rows*cols;		// end of matrix
	VTYPE* pVal = data;
	VTYPE* pNewVal = b.data;
	VTYPE* pNewEnd = pNewVal + mx;
	int pNewSkip = cols - mx;

	for (pVal = data; pVal < pEnd;) {
		for (; pNewVal < pNewEnd;) *pVal++ = *pNewVal++;
		pNewVal += pNewSkip;
		pNewEnd += mx;
	}
	return *this;
}

// N-Vector Dot Product
// Elements may be in rows or columns, but:
// - If in rows, number of columns must be one and number of rows must match.
// - If in cols, number of rows must be one and number of cols must match.
double MatrixF::Dot ( MatrixF& b )
{
	double d = 0.0;
	VTYPE* pA = data;
	VTYPE* pB = b.data;
	
	if ( rows==1 && b.rows==1 && cols == b.cols ) {
		VTYPE* pAEnd = data + cols;
		d = 0.0;
		for (; pA < pAEnd;)
			d += (*pA++) * (*pB++);
	} else if ( cols==1 && b.cols==1 && rows == b.rows) {
		VTYPE* pAEnd = data + rows;
		d = 0.0;
		for (; pA < pAEnd;)
			d += (*pA++) * (*pB++);
	}	
	return d;
}

#define I(x, y)		( (y*xres) + x )
#define Ix(r)		( r % xres )			// X coordinate from row	
#define Iy(r)		( r / xres )			// Y coordinate from row

MatrixF &MatrixF::MatrixVector5 (MatrixF& x, int mrows, MatrixF& b)
{
	double v;	

	// A( 2, r ) * B ( r ) + A(1,r)*B(r-1) + A(3,r)*B(r+1) + A(0, r)*B( R-( r ) ) + A(4, r)*B( R+( r ) )
	for (int r = 0; r < mrows; r++) {
		v = GetVal(2, r) * x(0,r);
		if ( r > 0 ) v += GetVal(1,r) * x(0,r-1);
		if ( r < mrows-1) v += GetVal(3,r) * x(0,r+1);
		if ( (int) GetVal(5, r) >= 0) v += GetVal(0,r) * x(0, (int) GetVal(5,r));
		if ( (int) GetVal(6, r) >= 0) v += GetVal(4,r) * x(0, (int) GetVal(6,r));
		b(0,r) = v;
	}
	return *this;
}

MatrixF &MatrixF::ConjugateGradient (MatrixF &b)
{
	return *this;
}

// Sparse Conjugate Gradient 2D (special case)
// This compute conjugate gradients on a 
// sparse "5-7" x N positive definite matrix. 
// Only 'mrows' subset of the row-size of A and b will be used.
MatrixF &MatrixF::ConjugateGradient5 (MatrixF &b, int mrows )
{
	double a, g, rdot;
	int i, imax;
	MatrixF x, xnew;				// solution vector
	MatrixF r, rnew;				// residual
	MatrixF p, ptemp;				// search direction
	MatrixF v;

	x.Resize ( 1, mrows );
	xnew.Resize ( 1, mrows );
	r.Resize ( 1, mrows );
	rnew.Resize ( 1, mrows );
	p.Resize ( 1, mrows );
	ptemp.Resize ( 1, mrows );
	v.Resize ( 1, mrows );
	
	r.Submatrix ( b, 1, mrows);
	MatrixVector5 ( x, mrows, v );				// (Ax -> v)
	r -= v;										// r = b - Ax
	p = r;
	
	imax = 20;
	for (i=0; i < imax; i++) {
		MatrixVector5 ( p, mrows, v );			// v = Ap
		rdot = r.Dot ( r );
		a = rdot / p.Dot ( v );					// a = (r . r) / (p . v)		
		xnew = p;
		xnew *= a;
		xnew += x;								// x = x + p*a
		v *= a;
		rnew = r;								// rnew = r - v*a
		rnew -= v;								
		g = rnew.Dot ( rnew ) / rdot;			// g = (rnew . rnew) / (r . r)
		p *= g;
		p += rnew;								// p = rnew + p*g
		r = rnew;
		x = xnew;
	}	
	for (int rx=0; rx < mrows; rx++) 
		b(0, rx) = x(0, rx);
	return *this;
}

int MatrixF::GetX()						{return cols;}
int MatrixF::GetY()						{return rows;}
int MatrixF::GetRows(void)				{return rows;}
int MatrixF::GetCols(void)				{return cols;}
int MatrixF::GetLength(void)				{return len;}
VTYPE *MatrixF::GetData(void)			{return data;}

double MatrixF::GetF (const int r, const int c)		{return (double) (*(data + r*cols + c));}

void MatrixF::GetRowVec (int r, Vector3DF &v)
{ 
	VTYPE *n = data + r*cols;
	v.x = (float) *n++; v.y = (float) *n++; v.z= (float) *n++;
}

void MatrixF::Print ( char* fname )
{
	char buf[2000];
	
	#ifdef _MSC_VER
		FILE* fp;
		fopen_s (&fp, fname, "w+" );
	#else
		FILE* fp = fopen ( fname, "w+" );
	#endif
	
	for (int r=0; r < rows; r++) {
		buf[0] = '\0';
		for (int c =0; c < cols; c++) {
			#ifdef _MSC_VER
				sprintf_s ( buf, "%s %04.3f", buf, GetVal(c, r) );
			#else
				sprintf ( buf, "%s %04.3f", buf, GetVal(c, r) );
			#endif
		}
		fprintf ( fp, "%s\n", buf);
	}
	fprintf ( fp, "---------------------------------------\n", buf);
	fflush ( fp );
	fclose ( fp );	
}


// MatrixF Code Definition
#undef VTYPE
#define VNAME		F
#define VTYPE		float

// Constructors/Destructors

Matrix4F::Matrix4F ( float* src )	{ for (int n=0; n < 16; n++) data[n] = src[n]; }
Matrix4F::Matrix4F ( float f0, float f1, float f2, float f3, 
							float f4, float f5, float f6, float f7, 
							float f8, float f9, float f10, float f11,
							float f12, float f13, float f14, float f15 )
{
	data[0] = f0;	data[1] = f1;	data[2] = f2;	data[3] = f3;
	data[4] = f4;	data[5] = f5;	data[6] = f6;	data[7] = f7;
	data[8] = f8;	data[9] = f9;	data[10] = f10;	data[11] = f11;
	data[12] = f12;	data[13] = f13;	data[14] = f14;	data[15] = f15;
}

Matrix4F Matrix4F::operator* (const float &op)
{
	return Matrix4F ( data[0]*op,	data[1]*op, data[2]*op, data[3],
					  data[4]*op,	data[5]*op, data[6]*op,	data[7],
					  data[8]*op,	data[9]*op, data[10]*op, data[11],
					  data[12],		data[13],	data[14],	data[15] );
}

Matrix4F Matrix4F::operator* (const Vector3DF &op)
{
	return Matrix4F ( data[0]*op.x, data[1]*op.y, data[2]*op.z, data[3],
					  data[4]*op.x, data[5]*op.y, data[6]*op.z, data[7],
					  data[8]*op.x, data[9]*op.y, data[10]*op.z, data[11],
					  data[12]*op.x, data[13]*op.y, data[14]*op.z, data[15] );
}

Matrix4F &Matrix4F::operator= (const unsigned char op)	{for ( int n=0; n<16; n++) data[n] = (VTYPE) op; return *this;}
Matrix4F &Matrix4F::operator= (const int op)				{for ( int n=0; n<16; n++) data[n] = (VTYPE) op; return *this;}
Matrix4F &Matrix4F::operator= (const double op)			{for ( int n=0; n<16; n++) data[n] = (VTYPE) op; return *this;}
Matrix4F &Matrix4F::operator+= (const unsigned char op)	{for ( int n=0; n<16; n++) data[n] += (VTYPE) op; return *this;}
Matrix4F &Matrix4F::operator+= (const int op)				{for ( int n=0; n<16; n++) data[n] += (VTYPE) op; return *this;}
Matrix4F &Matrix4F::operator+= (const double op)			{for ( int n=0; n<16; n++) data[n] += (VTYPE) op; return *this;}
Matrix4F &Matrix4F::operator-= (const unsigned char op)	{for ( int n=0; n<16; n++) data[n] -= (VTYPE) op; return *this;}
Matrix4F &Matrix4F::operator-= (const int op)				{for ( int n=0; n<16; n++) data[n] -= (VTYPE) op; return *this;}
Matrix4F &Matrix4F::operator-= (const double op)			{for ( int n=0; n<16; n++) data[n] -= (VTYPE) op; return *this;}
Matrix4F &Matrix4F::operator*= (const unsigned char op)	{for ( int n=0; n<16; n++) data[n] *= (VTYPE) op; return *this;}
Matrix4F &Matrix4F::operator*= (const int op)				{for ( int n=0; n<16; n++) data[n] *= (VTYPE) op; return *this;}
Matrix4F &Matrix4F::operator*= (const double op)			{for ( int n=0; n<16; n++) data[n] *= (VTYPE) op; return *this;}
Matrix4F &Matrix4F::operator/= (const unsigned char op)	{for ( int n=0; n<16; n++) data[n] /= (VTYPE) op; return *this;}
Matrix4F &Matrix4F::operator/= (const int op)				{for ( int n=0; n<16; n++) data[n] /= (VTYPE) op; return *this;}
Matrix4F &Matrix4F::operator/= (const double op)			{for ( int n=0; n<16; n++) data[n] /= (VTYPE) op; return *this;}

// column-major multiply (like OpenGL)
Matrix4F &Matrix4F::operator*= (const Matrix4F &op) {
	register float orig[16];				// Temporary storage
	memcpy ( orig, data, 16*sizeof(float) );

	// Calculate First Row
	data[0] = op.data[0]*orig[0] + op.data[1]*orig[4] + op.data[2]*orig[8] + op.data[3]*orig[12];
	data[1] = op.data[0]*orig[1] + op.data[1]*orig[5] + op.data[2]*orig[9] + op.data[3]*orig[13];
	data[2] = op.data[0]*orig[2] + op.data[1]*orig[6] + op.data[2]*orig[10] + op.data[3]*orig[14];
	data[3] = op.data[0]*orig[3] + op.data[1]*orig[7] + op.data[2]*orig[11] + op.data[3]*orig[15];

	// Calculate Second Row
	data[4] = op.data[4]*orig[0] + op.data[5]*orig[4] + op.data[6]*orig[8] + op.data[7]*orig[12];
	data[5] = op.data[4]*orig[1] + op.data[5]*orig[5] + op.data[6]*orig[9] + op.data[7]*orig[13];
	data[6] = op.data[4]*orig[2] + op.data[5]*orig[6] + op.data[6]*orig[10] + op.data[7]*orig[14];
	data[7] = op.data[4]*orig[3] + op.data[5]*orig[7] + op.data[6]*orig[11] + op.data[7]*orig[15];
	
	// Calculate Third Row
	data[8] = op.data[8]*orig[0] + op.data[9]*orig[4] + op.data[10]*orig[8] + op.data[11]*orig[12];
	data[9] = op.data[8]*orig[1] + op.data[9]*orig[5] + op.data[10]*orig[9] + op.data[11]*orig[13];
	data[10] = op.data[8]*orig[2] + op.data[9]*orig[6] + op.data[10]*orig[10] + op.data[11]*orig[14];
	data[11] = op.data[8]*orig[3] + op.data[9]*orig[7] + op.data[10]*orig[11] + op.data[11]*orig[15];

	// Calculate Four Row
	data[12] = op.data[12]*orig[0] + op.data[13]*orig[4] + op.data[14]*orig[8] + op.data[15]*orig[12];
	data[13] = op.data[12]*orig[1] + op.data[13]*orig[5] + op.data[14]*orig[9] + op.data[15]*orig[13];
	data[14] = op.data[12]*orig[2] + op.data[13]*orig[6] + op.data[14]*orig[10] + op.data[15]*orig[14];
	data[15] = op.data[12]*orig[3] + op.data[13]*orig[7] + op.data[14]*orig[11] + op.data[15]*orig[15];

	return *this;
}

Matrix4F &Matrix4F::operator= (const float* op) 
{
	for (int n=0; n < 16; n++ )
		data[n] = op[n];
	return *this;
}

Matrix4F &Matrix4F::operator*= (const float* op) {
	register float orig[16];				// Temporary storage
	memcpy ( orig, data, 16*sizeof(float) );

	// Calculate First Row
	data[0] = op[0]*orig[0] + op[1]*orig[4] + op[2]*orig[8] + op[3]*orig[12];
	data[1] = op[0]*orig[1] + op[1]*orig[5] + op[2]*orig[9] + op[3]*orig[13];
	data[2] = op[0]*orig[2] + op[1]*orig[6] + op[2]*orig[10] + op[3]*orig[14];
	data[3] = op[0]*orig[3] + op[1]*orig[7] + op[2]*orig[11] + op[3]*orig[15];

	// Calculate Second Row
	data[4] = op[4]*orig[0] + op[5]*orig[4] + op[6]*orig[8] + op[7]*orig[12];
	data[5] = op[4]*orig[1] + op[5]*orig[5] + op[6]*orig[9] + op[7]*orig[13];
	data[6] = op[4]*orig[2] + op[5]*orig[6] + op[6]*orig[10] + op[7]*orig[14];
	data[7] = op[4]*orig[3] + op[5]*orig[7] + op[6]*orig[11] + op[7]*orig[15];
	
	// Calculate Third Row
	data[8] = op[8]*orig[0] + op[9]*orig[4] + op[10]*orig[8] + op[11]*orig[12];
	data[9] = op[8]*orig[1] + op[9]*orig[5] + op[10]*orig[9] + op[11]*orig[13];
	data[10] = op[8]*orig[2] + op[9]*orig[6] + op[10]*orig[10] + op[11]*orig[14];
	data[11] = op[8]*orig[3] + op[9]*orig[7] + op[10]*orig[11] + op[11]*orig[15];

	// Calculate Four Row
	data[12] = op[12]*orig[0] + op[13]*orig[4] + op[14]*orig[8] + op[15]*orig[12];
	data[13] = op[12]*orig[1] + op[13]*orig[5] + op[14]*orig[9] + op[15]*orig[13];
	data[14] = op[12]*orig[2] + op[13]*orig[6] + op[14]*orig[10] + op[15]*orig[14];
	data[15] = op[12]*orig[3] + op[13]*orig[7] + op[14]*orig[11] + op[15]*orig[15];

	return *this;
}


Matrix4F &Matrix4F::Transpose (void)
{
	register float orig[16];				// Temporary storage
	memcpy ( orig, data, 16*sizeof(VTYPE) );
	
	data[0] = orig[0];	data[1] = orig[4];	data[2] = orig[8];	data[3] = orig[12];
	data[4] = orig[1];	data[5] = orig[5];	data[6] = orig[9];	data[7] = orig[13];
	data[8] = orig[2];	data[9] = orig[6];	data[10] = orig[10];data[11] = orig[14];
	data[12] = orig[3];	data[13] = orig[7];	data[14] = orig[11];data[15] = orig[15];
	return *this;	
}

Matrix4F &Matrix4F::Identity ()
{
	memset (data, 0, 16*sizeof(VTYPE));	
	data[0] = 1.0;
	data[5] = 1.0;
	data[10] = 1.0;
	data[15] = 1.0;	
	return *this;
}

// Pre-multiply (left side multiply ZYX) = Euler rotation about X, then Y, then Z
//
Matrix4F &Matrix4F::RotateZYX (const Vector3DF& angs)
{	
	float cx,sx,cy,sy,cz,sz;
	cx = (float) cos(angs.x * 3.141592/180);
	sx = (float) sin(angs.x * 3.141592/180);	
	cy = (float) cos(angs.y * 3.141592/180);
	sy = (float) sin(angs.y * 3.141592/180);	
	cz = (float) cos(angs.z * 3.141592/180);
	sz = (float) sin(angs.z * 3.141592/180);	
	data[0] = (VTYPE) cz * cy;
	data[1] = (VTYPE) sz * cy;
	data[2] = (VTYPE) -sy;
	data[3] = (VTYPE) 0;
	data[4] = (VTYPE) -sz * cx + cz*sy*sx;
	data[5] = (VTYPE)  cz * cx - sz*sy*sz;
	data[6] = (VTYPE) -cy * sx;
	data[7] = (VTYPE) 0 ;
	data[8] = (VTYPE) -sz * sx + cz*sy*cx;
	data[9] = (VTYPE)  cz * sx + sz*sy*cx;
	data[10] = (VTYPE) cy * cx;
	data[11] = 0;
	data[12] = 0;
	data[13] = 0;
	data[14] = 0;
	data[15] = 1;
	return *this;
}
Matrix4F &Matrix4F::RotateZYXT (const Vector3DF& angs, const Vector3DF& t)
{	
	float cx,sx,cy,sy,cz,sz;
	cx = (float) cos(angs.x * 3.141592/180);
	sx = (float) sin(angs.x * 3.141592/180);	
	cy = (float) cos(angs.y * 3.141592/180);
	sy = (float) sin(angs.y * 3.141592/180);	
	cz = (float) cos(angs.z * 3.141592/180);
	sz = (float) sin(angs.z * 3.141592/180);	
	data[0] = (VTYPE) cy * cz;				// See Diebel 2006, "Representing Attitude"
	data[1] = (VTYPE) cy * sz;
	data[2] = (VTYPE) -sy;
	data[3] = (VTYPE) 0;
	data[4] = (VTYPE) sx*sy*cz - cx*sz;
	data[5] = (VTYPE) sx*sy*sz + cx*cz;
	data[6] = (VTYPE) sx * cy;
	data[7] = (VTYPE) 0 ;
	data[8] = (VTYPE) cx*sy*cz + sx*sz;
	data[9] = (VTYPE) cx*sy*sz - sx*cz;
	data[10] = (VTYPE) cx * cy;
	data[11] = 0;
	data[12] = (VTYPE) data[0]*t.x + data[4]*t.y + data[8]*t.z;
	data[13] = (VTYPE) data[1]*t.x + data[5]*t.y + data[9]*t.z;
	data[14] = (VTYPE) data[2]*t.x + data[6]*t.y + data[10]*t.z;
	data[15] = 1;
	return *this;
}
Matrix4F &Matrix4F::RotateTZYX (const Vector3DF& angs, const Vector3DF& t)
{	
	float cx,sx,cy,sy,cz,sz;
	cx = (float) cos(angs.x * 3.141592/180);
	sx = (float) sin(angs.x * 3.141592/180);	
	cy = (float) cos(angs.y * 3.141592/180);
	sy = (float) sin(angs.y * 3.141592/180);	
	cz = (float) cos(angs.z * 3.141592/180);
	sz = (float) sin(angs.z * 3.141592/180);	
	data[0] = (VTYPE) cz * cy;
	data[1] = (VTYPE) sz * cy;
	data[2] = (VTYPE) -sy;
	data[3] = (VTYPE) 0;
	data[4] = (VTYPE) -sz * cx + cz*sy*sx;
	data[5] = (VTYPE)  cz * cx + sz*sy*sz;
	data[6] = (VTYPE)  cy * sx;
	data[7] = (VTYPE) 0 ;
	data[8] = (VTYPE)  sz * sx + cz*sy*cx;
	data[9] = (VTYPE) -cz * sx + sz*sy*cx;
	data[10] = (VTYPE) cy * cx;
	data[11] = 0;
	data[12] = (VTYPE) t.x;
	data[13] = (VTYPE) t.y;
	data[14] = (VTYPE) t.z;
	data[15] = 1;
	return *this;
}



// rotates points >>counter-clockwise<< when looking down the Y+ axis toward the origin
Matrix4F &Matrix4F::RotateY (const double ang)
{
	memset (data, 0, 16*sizeof(VTYPE));			
	double c,s;
	c = cos(ang * 3.141592/180);
	s = sin(ang * 3.141592/180);
	data[0] = (VTYPE) c;
	data[2] = (VTYPE) -s;
	data[5] = 1;		
	data[8] = (VTYPE) s;
	data[10] = (VTYPE) c;
	data[15] = 1;
	return *this;
}

// rotates points >>counter-clockwise<< when looking down the Z+ axis toward the origin
Matrix4F &Matrix4F::RotateZ (const double ang)
{
	memset (data, 0, 16*sizeof(VTYPE));			
	double c,s;
	c = cos(ang * 3.141592/180);
	s = sin(ang * 3.141592/180);
	data[0] = (VTYPE) c;	data[1] = (VTYPE) s;
	data[4] = (VTYPE) -s;	data[5] = (VTYPE) c;
	data[10] = 1; 
	data[15] = 1;
	return *this;
}

Matrix4F &Matrix4F::Ortho (double sx, double sy, double vn, double vf)
{
	// simplified version of OpenGL's glOrtho function	
	data[ 0] = (VTYPE) (1.0/sx);data[ 1] = (VTYPE) 0.0;		data[ 2] = (VTYPE) 0.0;				data[ 3]= (VTYPE) 0.0;
	data[ 4] = (VTYPE) 0.0;		data[ 5] = (VTYPE) (1.0/sy);data[ 6] = (VTYPE) 0.0;				data[ 7] = (VTYPE) 0.0;
	data[ 8] = (VTYPE) 0.0;		data[ 9] = (VTYPE) 0.0;		data[10]= (VTYPE) (-2.0/(vf-vn));	data[11] = (VTYPE) (-(vf+vn)/(vf-vn));
	data[12] = (VTYPE) 0.0;		data[13] = (VTYPE) 0.0;		data[14] = (VTYPE) 0;				data[15] = (VTYPE) 1.0;
	return *this;
}

Matrix4F &Matrix4F::Translate (double tx, double ty, double tz)
{
	data[ 0] = (VTYPE) 1.0; data[ 1] = (VTYPE) 0.0;	data[ 2] = (VTYPE) 0.0; data[ 3] = (VTYPE) 0.0;
	data[ 4] = (VTYPE) 0.0; data[ 5] = (VTYPE) 1.0; data[ 6] = (VTYPE) 0.0; data[ 7] = (VTYPE) 0.0;
	data[ 8] = (VTYPE) 0.0; data[ 9] = (VTYPE) 0.0; data[10] = (VTYPE) 1.0; data[11] = (VTYPE) 0.0;
	data[12] = (VTYPE) tx;	data[13] = (VTYPE) ty;	data[14] = (VTYPE) tz;	data[15] = (VTYPE) 1.0;	
	return *this;
}

Matrix4F &Matrix4F::Scale (double sx, double sy, double sz)
{
	data[ 0] = (VTYPE) sx; data[ 1] = (VTYPE) 0.0;	data[ 2] = (VTYPE) 0.0; data[ 3] = (VTYPE) 0.0;
	data[ 4] = (VTYPE) 0.0; data[ 5] = (VTYPE) sy; data[ 6] = (VTYPE) 0.0; data[ 7] = (VTYPE) 0.0;
	data[ 8] = (VTYPE) 0.0; data[ 9] = (VTYPE) 0.0; data[10] = (VTYPE) sz; data[11] = (VTYPE) 0.0;
	data[12] = (VTYPE) 0.0;	data[13] = (VTYPE) 0.0;	data[14] = (VTYPE) 0.0;	data[15] = (VTYPE) 1.0;	
	return *this;
}

Matrix4F &Matrix4F::Basis (const Vector3DF &norm)
{
	Vector3DF binorm, tang;
	binorm.Set ( 0.0, 1.0, 0 );		// up vector
	binorm.Cross ( norm );	
	binorm.Normalize ();
	tang = binorm;
	tang.Cross ( norm );	
	//tang *= -1;
	tang.Normalize ();
	
	data[ 0] = (VTYPE) binorm.x; data[ 1] = (VTYPE) binorm.y; data[ 2] = (VTYPE) binorm.z; data[ 3] = (VTYPE) 0.0;
	data[ 4] = (VTYPE) norm.x; data[ 5] = (VTYPE) norm.y; data[ 6] = (VTYPE) norm.z; data[ 7] = (VTYPE) 0.0;
	data[ 8] = (VTYPE) tang.x; data[ 9] = (VTYPE) tang.y; data[10] = (VTYPE) tang.z; data[11] = (VTYPE) 0.0;
	data[12] = (VTYPE) 0.0;	 data[13] = (VTYPE) 0.0;  data[14] = (VTYPE) 0.0;  data[15] = (VTYPE) 1.0;	
	return *this;
	
}

Matrix4F &Matrix4F::Basis (const Vector3DF &c1, const Vector3DF &c2, const Vector3DF &c3)
{
	data[ 0] = (VTYPE) c1.x; data[ 1] = (VTYPE) c2.x; data[ 2] = (VTYPE) c3.x; data[ 3] = (VTYPE) 0.0;
	data[ 4] = (VTYPE) c1.y; data[ 5] = (VTYPE) c2.y; data[ 6] = (VTYPE) c3.y; data[ 7] = (VTYPE) 0.0;
	data[ 8] = (VTYPE) c1.z; data[ 9] = (VTYPE) c2.z; data[10] = (VTYPE) c3.z; data[11] = (VTYPE) 0.0;
	data[12] = (VTYPE)  0.0; data[13] = (VTYPE)  0.0; data[14] = (VTYPE)  0.0; data[15] = (VTYPE) 1.0;
	return *this;
}
Matrix4F &Matrix4F::TransSRT (const Vector3DF &c1, const Vector3DF &c2, const Vector3DF &c3, const Vector3DF& t, const Vector3DF& s)
{
	data[ 0] = (VTYPE) c1.x*s.x; data[ 4] = (VTYPE) c2.x*s.x; data[ 8] = (VTYPE) c3.x*s.x;  data[12] = (VTYPE) 0.0;
	data[ 1] = (VTYPE) c1.y*s.y; data[ 5] = (VTYPE) c2.y*s.y; data[ 9] = (VTYPE) c3.y*s.y;  data[13] = (VTYPE) 0.0;
	data[ 2] = (VTYPE) c1.z*s.z; data[ 6] = (VTYPE) c2.z*s.z; data[10] = (VTYPE) c3.z*s.z;  data[14] = (VTYPE) 0.0;
	data[ 3] = (VTYPE) t.x;		 data[ 7] = (VTYPE) t.y;	  data[11] = (VTYPE) t.z;		data[15] = (VTYPE) 1.0;	
	return *this;
}

Matrix4F &Matrix4F::SRT (const Vector3DF &c1, const Vector3DF &c2, const Vector3DF &c3, const Vector3DF& t, const Vector3DF& s)
{
	data[ 0] = (VTYPE) c1.x*s.x; data[ 1] = (VTYPE) c2.x*s.x; data[ 2] = (VTYPE) c3.x*s.x;  data[ 3] = (VTYPE) 0.0;
	data[ 4] = (VTYPE) c1.y*s.y; data[ 5] = (VTYPE) c2.y*s.y; data[ 6] = (VTYPE) c3.y*s.y;  data[ 7] = (VTYPE) 0.0;
	data[ 8] = (VTYPE) c1.z*s.z; data[ 9] = (VTYPE) c2.z*s.z; data[10] = (VTYPE) c3.z*s.z;  data[11] = (VTYPE) 0.0;
	data[12] = (VTYPE) t.x;		 data[13] = (VTYPE) t.y;	  data[14] = (VTYPE) t.z;		data[15] = (VTYPE) 1.0;	
	return *this;
}
Matrix4F &Matrix4F::SRT (const Vector3DF &c1, const Vector3DF &c2, const Vector3DF &c3, const Vector3DF& t, const float s)
{
	data[ 0] = (VTYPE) c1.x*s; data[ 1] = (VTYPE) c1.y*s; data[ 2] = (VTYPE) c1.z*s;  data[ 3] = (VTYPE) 0.0;
	data[ 4] = (VTYPE) c2.x*s; data[ 5] = (VTYPE) c2.y*s; data[ 6] = (VTYPE) c2.z*s;  data[ 7] = (VTYPE) 0.0;
	data[ 8] = (VTYPE) c3.x*s; data[ 9] = (VTYPE) c3.y*s; data[10] = (VTYPE) c3.z*s;  data[11] = (VTYPE) 0.0;
	data[12] = (VTYPE) t.x;		 data[13] = (VTYPE) t.y;	  data[14] = (VTYPE) t.z;		data[15] = (VTYPE) 1.0;	
	return *this;
}

Matrix4F &Matrix4F::InvTRS (const Vector3DF &c1, const Vector3DF &c2, const Vector3DF &c3, const Vector3DF& t, const Vector3DF& s)
{
	data[ 0] = (VTYPE) c1.x/s.x; data[ 1] = (VTYPE) c1.y/s.y; data[ 2] = (VTYPE) c1.z/s.z;  data[ 3] = (VTYPE) 0.0;
	data[ 4] = (VTYPE) c2.x/s.x; data[ 5] = (VTYPE) c2.y/s.y; data[ 6] = (VTYPE) c2.z/s.z;  data[ 7] = (VTYPE) 0.0;
	data[ 8] = (VTYPE) c3.x/s.x; data[ 9] = (VTYPE) c3.y/s.y; data[10] = (VTYPE) c3.z/s.z;  data[11] = (VTYPE) 0.0;
	data[12] = (VTYPE) -t.x/s.x; data[13] = (VTYPE) -t.y/s.y; data[14] = (VTYPE) -t.z/s.z;  data[15] = (VTYPE) 1.0;	
	return *this;
}

Matrix4F &Matrix4F::InvTRS (const Vector3DF &c1, const Vector3DF &c2, const Vector3DF &c3, const Vector3DF& t, const float s)
{
	data[ 0] = (VTYPE) c1.x/s; data[ 1] = (VTYPE) c1.y/s; data[ 2] = (VTYPE) c1.z/s;  data[ 3] = (VTYPE) 0.0;
	data[ 4] = (VTYPE) c2.x/s; data[ 5] = (VTYPE) c2.y/s; data[ 6] = (VTYPE) c2.z/s;  data[ 7] = (VTYPE) 0.0;
	data[ 8] = (VTYPE) c3.x/s; data[ 9] = (VTYPE) c3.y/s; data[10] = (VTYPE) c3.z/s;  data[11] = (VTYPE) 0.0;
	data[12] = (VTYPE) -t.x/s; data[13] = (VTYPE) -t.y/s; data[14] = (VTYPE) -t.z/s;  data[15] = (VTYPE) 1.0;	
	return *this;
}

Matrix4F &Matrix4F::InvertTRS ()
{
	double inv[16], det;
	// mult: 16 *  13 + 4 	= 212
	// add:   16 * 5 + 3 	=   83
	int i;
	inv[0] =   data[5]*data[10]*data[15] - data[5]*data[11]*data[14] - data[9]*data[6]*data[15] + data[9]*data[7]*data[14] + data[13]*data[6]*data[11] - data[13]*data[7]*data[10];
	inv[4] =  -data[4]*data[10]*data[15] + data[4]*data[11]*data[14] + data[8]*data[6]*data[15]- data[8]*data[7]*data[14] - data[12]*data[6]*data[11] + data[12]*data[7]*data[10];
	inv[8] =   data[4]*data[9]*data[15] - data[4]*data[11]*data[13] - data[8]*data[5]*data[15]+ data[8]*data[7]*data[13] + data[12]*data[5]*data[11] - data[12]*data[7]*data[9];
	inv[12] = -data[4]*data[9]*data[14] + data[4]*data[10]*data[13] + data[8]*data[5]*data[14]- data[8]*data[6]*data[13] - data[12]*data[5]*data[10] + data[12]*data[6]*data[9];
	inv[1] =  -data[1]*data[10]*data[15] + data[1]*data[11]*data[14] + data[9]*data[2]*data[15]- data[9]*data[3]*data[14] - data[13]*data[2]*data[11] + data[13]*data[3]*data[10];
	inv[5] =   data[0]*data[10]*data[15] - data[0]*data[11]*data[14] - data[8]*data[2]*data[15]+ data[8]*data[3]*data[14] + data[12]*data[2]*data[11] - data[12]*data[3]*data[10];
	inv[9] =  -data[0]*data[9]*data[15] + data[0]*data[11]*data[13] + data[8]*data[1]*data[15]- data[8]*data[3]*data[13] - data[12]*data[1]*data[11] + data[12]*data[3]*data[9];
	inv[13] =  data[0]*data[9]*data[14] - data[0]*data[10]*data[13] - data[8]*data[1]*data[14]+ data[8]*data[2]*data[13] + data[12]*data[1]*data[10] - data[12]*data[2]*data[9];
	inv[2] =   data[1]*data[6]*data[15] - data[1]*data[7]*data[14] - data[5]*data[2]*data[15]+ data[5]*data[3]*data[14] + data[13]*data[2]*data[7] - data[13]*data[3]*data[6];
	inv[6] =  -data[0]*data[6]*data[15] + data[0]*data[7]*data[14] + data[4]*data[2]*data[15]- data[4]*data[3]*data[14] - data[12]*data[2]*data[7] + data[12]*data[3]*data[6];
	inv[10] =  data[0]*data[5]*data[15] - data[0]*data[7]*data[13] - data[4]*data[1]*data[15]+ data[4]*data[3]*data[13] + data[12]*data[1]*data[7] - data[12]*data[3]*data[5];
	inv[14] = -data[0]*data[5]*data[14] + data[0]*data[6]*data[13] + data[4]*data[1]*data[14]- data[4]*data[2]*data[13] - data[12]*data[1]*data[6] + data[12]*data[2]*data[5];
	inv[3] =  -data[1]*data[6]*data[11] + data[1]*data[7]*data[10] + data[5]*data[2]*data[11]- data[5]*data[3]*data[10] - data[9]*data[2]*data[7] + data[9]*data[3]*data[6];
	inv[7] =   data[0]*data[6]*data[11] - data[0]*data[7]*data[10] - data[4]*data[2]*data[11]+ data[4]*data[3]*data[10] + data[8]*data[2]*data[7] - data[8]*data[3]*data[6];
	inv[11] = -data[0]*data[5]*data[11] + data[0]*data[7]*data[9] + data[4]*data[1]*data[11]- data[4]*data[3]*data[9] - data[8]*data[1]*data[7] + data[8]*data[3]*data[5];
	inv[15] =  data[0]*data[5]*data[10] - data[0]*data[6]*data[9] - data[4]*data[1]*data[10]+ data[4]*data[2]*data[9] + data[8]*data[1]*data[6] - data[8]*data[2]*data[5];
	
	det = data[0]*inv[0] + data[1]*inv[4] + data[2]*inv[8] + data[3]*inv[12];
	if (det == 0)    return *this;
	det = 1.0f / det;

	for (i = 0; i < 16; i++)  
		data[i] = (float) (inv[i] * det);
	
	return *this;
}
float Matrix4F::GetF (const int r, const int c)		{return (float) data[ (r<<2) + c];}

void Matrix4F::GetRowVec (int r, Vector3DF &v)
{ 
	v.x = data[ (r<<2) ]; 
	v.y = data[ (r<<2)+1 ]; 
	v.z = data[ (r<<2)+2 ];
}

Matrix4F &Matrix4F::operator= ( float* mat )
{
	for (int n=0; n < 16; n++) 
		data[n] = mat[n];	
	return *this;
}

// Translate after (post-translate)
// Computes: M' = T*M
//
Matrix4F &Matrix4F::operator+= (const Vector3DF& t)
{	
	data[12] += (VTYPE) t.x;
	data[13] += (VTYPE) t.y;
	data[14] += (VTYPE) t.z;
	return *this;
}

// Translate first (pre-translate)
// Computes: M' = M*T
Matrix4F &Matrix4F::PreTranslate (const Vector3DF& t)
{	
	data[12] += (VTYPE) data[0]*t.x + data[4]*t.y + data[8]*t.z;
	data[13] += (VTYPE) data[1]*t.x + data[5]*t.y + data[9]*t.z;
	data[14] += (VTYPE) data[2]*t.x + data[6]*t.y + data[10]*t.z;
	return *this;
}

Matrix4F &Matrix4F::operator*= (const Vector3DF& t)		// quick scale
{	
	data[0] *= (VTYPE) t.x;	data[1] *= (VTYPE) t.x;	data[2] *= (VTYPE) t.x;	data[3] *= (VTYPE) t.x;
	data[4] *= (VTYPE) t.y;	data[5] *= (VTYPE) t.y;	data[6] *= (VTYPE) t.y;	data[7] *= (VTYPE) t.y;
	data[8] *= (VTYPE) t.z;	data[9] *= (VTYPE) t.z;	data[10] *= (VTYPE) t.z; data[11] *= (VTYPE) t.z;
	return *this;
}

Matrix4F &Matrix4F::InverseProj ( float* mat )
{
	data[0] = 1.0f/mat[0];	data[1] = 0.0f;			data[2] = 0.0f;						data[3] = 0.0f;
	data[4] = 0.0f;			data[5] = 1.0f/mat[5];	data[6] = 0.0f;						data[7] = 0.0f;
	data[8] = 0.0f;			data[9] = 0.0f;			data[10] = 0.0f;					data[11] = 1.0f/mat[14];
	data[12] = mat[8]/mat[0];		data[13] = mat[9]/mat[5];		data[14] = -1.0f;	data[15] = mat[10]/mat[14];
	return *this;
}

Matrix4F &Matrix4F::InverseView ( float* mat, Vector3DF& pos)
{
	// NOTE: Assumes there is no scaling in input matrix.
	// Although there can be translation (typical of a view matrix)
	data[0] = mat[0];	data[1] = mat[4];	data[2] = mat[8];	data[3] = 0.0f;
	data[4] = mat[1];	data[5] = mat[5];	data[6] = mat[9];	data[7] = 0.0f;
	data[8] = mat[2];	data[9] = mat[6];	data[10] = mat[10];	data[11] = 0.0f;
	data[12] = pos.x;	data[13] = pos.y;	data[14] =  pos.z;	data[15] = 1.0f;
	return *this;
}

Vector4DF Matrix4F::GetT ( float* mat )
{
	return Vector4DF ( mat[12], mat[13], mat[14], 1.0 );
}

void Matrix4F::Print ()
{
	printf ( (char*) "%04.3f %04.3f %04.3f %04.3f\n", data[0], data[1], data[2], data[3] );
	printf ( (char*) "%04.3f %04.3f %04.3f %04.3f\n", data[4], data[5], data[6], data[7] );
	printf ( (char*) "%04.3f %04.3f %04.3f %04.3f\n", data[8], data[9], data[10], data[11] );
	printf ( (char*) "%04.3f %04.3f %04.3f %04.3f\n\n", data[12], data[13], data[14], data[15] );
}
std::string Matrix4F::WriteToStr ()
{
	char buf[4096];
	std::string str;
	sprintf_s ( buf, 4096, "   %f %f %f %f\n", data[0], data[1], data[2], data[3] ); str = buf;
	sprintf_s ( buf, 4096, "   %f %f %f %f\n", data[4], data[5], data[6], data[7] ); str += buf;
	sprintf_s ( buf, 4096, "   %f %f %f %f\n", data[8], data[9], data[10], data[11] ); str += buf;
	sprintf_s ( buf, 4096, "   %f %f %f %f\n", data[12], data[13], data[14], data[15] ); str += buf;
	return str;
}

#undef VTYPE
#undef VNAME




// CGL - Complex Graphics Language
// Camera3D Class
//
// 3D Camera3D IMPLEMENTATION 
// 
// The Camera3D transformation of an arbitrary point is:
//
//		Q' = Q * T * R * P
//
// where Q  = 3D point
//		 Q' = Screen point
//		 T = Camera3D Translation (moves Camera3D to origin)
//		 R = Camera3D Rotation (rotates Camera3D to point 'up' along Z axis)
//       P = Projection (projects points onto XY plane)
// 
// T is a unit-coordinate system translated to origin from Camera3D:
//		[1	0  0  0]
//		[0	1  0  0]
//		[0	0  1  0]
//		[-cx -cy -cz 0]		where c is Camera3D location
// R is a basis matrix:	
//
// P is a projection matrix:
//

Camera3D::Camera3D ()
{	
	
	mProjType = Perspective;
	mWire = 0;

	up_dir.Set ( 0.0, 1.0, 0 );				// frustum params
	mAspect = (float) 800.0f/600.0f;
	mDolly = 5.0;
	mFov = 40.0;	
	mNear = (float) 0.1;
	mFar = (float) 6000.0;
	mTile.Set ( 0, 0, 1, 1 );

	for (int n=0; n < 8; n++ ) mOps[n] = false;	
	mOps[0] = false;

//	mOps[0] = true;		
//	mOps[1] = true;

	setOrbit ( 0, 45, 0, Vector3DF(0,0,0), 120.0, 5.0 );
	updateMatricies ();

}

/*void Camera3D::draw_gl ()
{
	Vector3DF pnt; 
	int va, vb;
	
	if ( !mOps[0] ) return;

	// Box testing
	//
	// NOTES: This demonstrates AABB box testing against the frustum 
	// Boxes tested are 10x10x10 size, spaced apart from each other so we can see them.
	if ( mOps[5] ) {
		glPushMatrix ();
		glEnable ( GL_LIGHTING );
		glColor3f ( 1, 1, 1 );	
		Vector3DF bmin, bmax, vmin, vmax;
		int lod;
		for (float y=0; y < 100; y += 10.0 ) {
		for (float z=-100; z < 100; z += 10.0 ) {
			for (float x=-100; x < 100; x += 10.0 ) {
				bmin.Set ( x, y, z );
				bmax.Set ( x+8, y+8, z+8 );
				if ( boxInFrustum ( bmin, bmax ) ) {				
					lod = (int) calculateLOD ( bmin, 1, 5, 300.0 );
					//rendGL->drawCube ( bmin, bmax, Vector3DF(1,1,1) );
				}
			}
		}
		}
		glPopMatrix ();
	}

	glDisable ( GL_LIGHTING );	
	glLoadMatrixf ( getViewMatrix().GetDataF() );

	// Frustum planes (world space)
	//
	// NOTE: The frustum planes are drawn as discs because
	// they are boundless (infinite). The minimum information contained in the
	// plane equation is normal direction and distance from plane to origin.
	// This sufficiently defines infinite planes for inside/outside testing,
	// but cannot be used to draw the view frustum without more information.
	// Drawing is done as discs here to verify the frustum plane equations.
	if ( mOps[2] ) {
		glBegin ( GL_POINTS );
		glColor3f ( 1, 1, 0 );
		Vector3DF norm;
		Vector3DF side, up;
		for (int n=0; n < 6; n++ ) {
			norm.Set ( frustum[n][0], frustum[n][1], frustum[n][2] );
			glColor3f ( n/6.0, 1.0- (n/6.0), 0.5 );
			side = Vector3DF(0,1,0); side.Cross ( norm ); side.Normalize ();	
			up = side; up.Cross ( norm ); up.Normalize();
			norm *= frustum[n][3];
			for (float y=-50; y < 50; y += 1.0 ) {
				for (float x=-50; x < 50; x += 1.0 ) {
					if ( x*x+y*y < 1000 ) {
						//pnt = side * x + up * y - norm; 
                        pnt = side;
                        Vector3DF tv = up;

                        tv *= y;
                        pnt *= x;
                        pnt += tv;
                        pnt -= norm;

						glVertex3f ( pnt.x, pnt.y, pnt.z );
					}
				}
			}
		}
		glEnd (); 
	}

	// Inside/outside testing
	//
	// NOTES: This code demonstrates frustum clipping 
	// tests on individual points.
	if ( mOps[4] ) {
		glColor3f ( 1, 1, 1 );
		glBegin ( GL_POINTS );
		for (float z=-100; z < 100; z += 4.0 ) {
			for (float y=0; y < 100; y += 4.0 ) {
				for (float x=-100; x < 100; x += 4.0 ) {
					if ( pointInFrustum ( x, y, z) ) {
						glVertex3f ( x, y, z );
					}
				}
			}
		}
		glEnd ();
	}
	
	// Inverse rays (world space)
	//
	// NOTES: This code demonstrates drawing 
	// inverse camera rays, as might be needed for raytracing or hit testing.
	if ( mOps[3] ) {
		glBegin ( GL_LINES );
		glColor3f ( 0, 1, 0);
		for (float x = 0; x <= 1.0; x+= 0.5 ) {
			for (float y = 0; y <= 1.0; y+= 0.5 ) {
				pnt = inverseRay ( x, y, mFar );
				pnt += from_pos;
				glVertex3f ( from_pos.x, from_pos.y, from_pos.z );		// all inverse rays originate at the camera center
				glVertex3f ( pnt.x, pnt.y, pnt.z );
			}
		}
		glEnd ();
	}

	// Projection
	//
	// NOTES: This code demonstrates 
	// perspective projection _without_ using the OpenGL pipeline.
	// Projection is done by the camera class. A cube is drawn on the near plane.
	
	// Cube geometry
	Vector3DF pnts[8];
	Vector3DI edge[12];
	pnts[0].Set (  0,  0,  0 );	pnts[1].Set ( 10,  0,  0 ); pnts[2].Set ( 10,  0, 10 ); pnts[3].Set (  0,  0, 10 );		// lower points (y=0)
	pnts[4].Set (  0, 10,  0 );	pnts[5].Set ( 10, 10,  0 ); pnts[6].Set ( 10, 10, 10 ); pnts[7].Set (  0, 10, 10 );		// upper points (y=10)
	edge[0].Set ( 0, 1, 0 ); edge[1].Set ( 1, 2, 0 ); edge[2].Set ( 2, 3, 0 ); edge[3].Set ( 3, 0, 0 );					// 4 lower edges
	edge[4].Set ( 4, 5, 0 ); edge[5].Set ( 5, 6, 0 ); edge[6].Set ( 6, 7, 0 ); edge[7].Set ( 7, 4, 0 );					// 4 upper edges
	edge[8].Set ( 0, 4, 0 ); edge[9].Set ( 1, 5, 0 ); edge[10].Set ( 2, 6, 0 ); edge[11].Set ( 3, 7, 0 );				// 4 vertical edges
	
	// -- White cube is drawn using OpenGL projection
	if ( mOps[6] ) {
		glBegin ( GL_LINES );
		glColor3f ( 1, 1, 1);
		for (int e = 0; e < 12; e++ ) {
			va = edge[e].x;
			vb = edge[e].y;
			glVertex3f ( pnts[va].x, pnts[va].y, pnts[va].z );
			glVertex3f ( pnts[vb].x, pnts[vb].y, pnts[vb].z );
		}
		glEnd ();	
	}

	//---- Draw the following in camera space..
	// NOTES:
	// The remainder drawing steps are done in 
	// camera space. This is done by multiplying by the
	// inverse_rotation matrix, which transforms from camera to world space.
	// The camera axes, near, and far planes can now be drawn in camera space.
	glPushMatrix ();
	glLoadMatrixf ( getViewMatrix().GetDataF() );
	glTranslatef ( from_pos.x, from_pos.y, from_pos.z );
	glMultMatrixf ( invrot_matrix.GetDataF() );				// camera space --to--> world space

	// -- Red cube is drawn on the near plane using software projection pipeline. See Camera3D::project
	if ( mOps[6] ) {
		glBegin ( GL_LINES );
		glColor3f ( 1, 0, 0);
		Vector4DF proja, projb;
		for (int e = 0; e < 12; e++ ) {
			va = edge[e].x;
			vb = edge[e].y;
			proja = project ( pnts[va] );
			projb = project ( pnts[vb] );
			if ( proja.w > 0 && projb.w > 0 && proja.w < 1 && projb.w < 1) {	// Very simple Z clipping  (try commenting this out and see what happens)
				glVertex3f ( proja.x, proja.y, proja.z );
				glVertex3f ( projb.x, projb.y, projb.z );
			}
		}
		glEnd ();
	}
	// Camera axes
	glBegin ( GL_LINES );
	float to_d = (from_pos - to_pos).Length();
	glColor3f ( .8,.8,.8); glVertex3f ( 0, 0, 0 );	glVertex3f ( 0, 0, -to_d );
	glColor3f ( 1,0,0); glVertex3f ( 0, 0, 0 );		glVertex3f ( 10, 0, 0 );
	glColor3f ( 0,1,0); glVertex3f ( 0, 0, 0 );		glVertex3f ( 0, 10, 0 );
	glColor3f ( 0,0,1); glVertex3f ( 0, 0, 0 );		glVertex3f ( 0, 0, 10 );
	glEnd ();

	if ( mOps[1] ) {
		// Near plane
		float sy = tan ( mFov * DEGtoRAD / 2.0);
		float sx = sy * mAspect;
		glColor3f ( 0.8, 0.8, 0.8 );
		glBegin ( GL_LINE_LOOP );
		glVertex3f ( -mNear*sx,  mNear*sy, -mNear );
		glVertex3f (  mNear*sx,  mNear*sy, -mNear );
		glVertex3f (  mNear*sx, -mNear*sy, -mNear );
		glVertex3f ( -mNear*sx, -mNear*sy, -mNear );
		glEnd ();
		// Far plane
		glBegin ( GL_LINE_LOOP );
		glVertex3f ( -mFar*sx,  mFar*sy, -mFar );
		glVertex3f (  mFar*sx,  mFar*sy, -mFar );
		glVertex3f (  mFar*sx, -mFar*sy, -mFar );
		glVertex3f ( -mFar*sx, -mFar*sy, -mFar );
		glEnd ();

		// Subview Near plane
		float l, r, t, b;
		l = -sx + 2.0*sx*mTile.x;						// Tile is in range 0 <= x,y <= 1
		r = -sx + 2.0*sx*mTile.z;
		t =  sy - 2.0*sy*mTile.y;
		b =  sy - 2.0*sy*mTile.w;
		glColor3f ( 0.8, 0.8, 0.0 );
		glBegin ( GL_LINE_LOOP );
		glVertex3f ( l * mNear, t * mNear, -mNear );
		glVertex3f ( r * mNear, t * mNear, -mNear );
		glVertex3f ( r * mNear, b * mNear, -mNear );
		glVertex3f ( l * mNear, b * mNear, -mNear );		
		glEnd ();
		// Subview Far plane
		glBegin ( GL_LINE_LOOP );
		glVertex3f ( l * mFar, t * mFar, -mFar );
		glVertex3f ( r * mFar, t * mFar, -mFar );
		glVertex3f ( r * mFar, b * mFar, -mFar );
		glVertex3f ( l * mFar, b * mFar, -mFar );		
		glEnd ();
	}

	glPopMatrix ();
}
*/

bool Camera3D::pointInFrustum ( float x, float y, float z )
{
	int p;
	for ( p = 0; p < 6; p++ )
		if( frustum[p][0] * x + frustum[p][1] * y + frustum[p][2] * z + frustum[p][3] <= 0 )
			return false;
	return true;
}

bool Camera3D::boxInFrustum ( Vector3DF bmin, Vector3DF bmax)
{
	Vector3DF vmin, vmax;
	int p;
	bool ret = true;	
	for ( p = 0; p < 6; p++ ) {
		vmin.x = ( frustum[p][0] > 0 ) ? bmin.x : bmax.x;		// Determine nearest and farthest point to plane
		vmax.x = ( frustum[p][0] > 0 ) ? bmax.x : bmin.x;		
		vmin.y = ( frustum[p][1] > 0 ) ? bmin.y : bmax.y;
		vmax.y = ( frustum[p][1] > 0 ) ? bmax.y : bmin.y;
		vmin.z = ( frustum[p][2] > 0 ) ? bmin.z : bmax.z;
		vmax.z = ( frustum[p][2] > 0 ) ? bmax.z : bmin.z;
		if ( frustum[p][0]*vmax.x + frustum[p][1]*vmax.y + frustum[p][2]*vmax.z + frustum[p][3] <= 0 ) return false;		// If nearest point is outside, Box is outside
		else if ( frustum[p][0]*vmin.x + frustum[p][1]*vmin.y + frustum[p][2]*vmin.z + frustum[p][3] <= 0 ) ret = true;		// If nearest inside and farthest point is outside, Box intersects
	}
	return ret;			// No points found outside. Box must be inside.
	
	/* --- Original method - Slow yet simpler.
	int p;
	for ( p = 0; p < 6; p++ ) {
		if( frustum[p][0] * bmin.x + frustum[p][1] * bmin.y + frustum[p][2] * bmin.z + frustum[p][3] > 0 ) continue;
		if( frustum[p][0] * bmax.x + frustum[p][1] * bmin.y + frustum[p][2] * bmin.z + frustum[p][3] > 0 ) continue;
		if( frustum[p][0] * bmax.x + frustum[p][1] * bmin.y + frustum[p][2] * bmax.z + frustum[p][3] > 0 ) continue;
		if( frustum[p][0] * bmin.x + frustum[p][1] * bmin.y + frustum[p][2] * bmax.z + frustum[p][3] > 0 ) continue;
		if( frustum[p][0] * bmin.x + frustum[p][1] * bmax.y + frustum[p][2] * bmin.z + frustum[p][3] > 0 ) continue;
		if( frustum[p][0] * bmax.x + frustum[p][1] * bmax.y + frustum[p][2] * bmin.z + frustum[p][3] > 0 ) continue;
		if( frustum[p][0] * bmax.x + frustum[p][1] * bmax.y + frustum[p][2] * bmax.z + frustum[p][3] > 0 ) continue;
		if( frustum[p][0] * bmin.x + frustum[p][1] * bmax.y + frustum[p][2] * bmax.z + frustum[p][3] > 0 ) continue;
		return false;
	}
	return true;*/
}

void Camera3D::setOrbit  ( Vector3DF angs, Vector3DF tp, float dist, float dolly )
{
	setOrbit ( angs.x, angs.y, angs.z, tp, dist, dolly );
}

void Camera3D::setOrbit ( float ax, float ay, float az, Vector3DF tp, float dist, float dolly )
{
	ang_euler.Set ( ax, ay, az );
	mOrbitDist = dist;
	mDolly = dolly;
	double dx, dy, dz;
	dx = cos ( ang_euler.y * DEGtoRAD ) * sin ( ang_euler.x * DEGtoRAD ) ;
	dy = sin ( ang_euler.y * DEGtoRAD );
	dz = cos ( ang_euler.y * DEGtoRAD ) * cos ( ang_euler.x * DEGtoRAD );
	from_pos.x = tp.x + (float) dx * mOrbitDist;
	from_pos.y = tp.y + (float) dy * mOrbitDist;
	from_pos.z = tp.z + (float) dz * mOrbitDist;
	to_pos.x = from_pos.x - (float) dx * mDolly;
	to_pos.y = from_pos.y - (float) dy * mDolly;
	to_pos.z = from_pos.z - (float) dz * mDolly;
	updateMatricies ();
}

void Camera3D::moveOrbit ( float ax, float ay, float az, float dd )
{
	ang_euler += Vector3DF(ax,ay,az);
	mOrbitDist += dd;
	
	double dx, dy, dz;
	dx = cos ( ang_euler.y * DEGtoRAD ) * sin ( ang_euler.x * DEGtoRAD ) ;
	dy = sin ( ang_euler.y * DEGtoRAD );
	dz = cos ( ang_euler.y * DEGtoRAD ) * cos ( ang_euler.x * DEGtoRAD );
	from_pos.x = to_pos.x + (float) dx * mOrbitDist;
	from_pos.y = to_pos.y + (float) dy * mOrbitDist;
	from_pos.z = to_pos.z + (float) dz * mOrbitDist;
	updateMatricies ();
}

void Camera3D::moveToPos ( float tx, float ty, float tz )
{
	to_pos += Vector3DF(tx,ty,tz);

	double dx, dy, dz;
	dx = cos ( ang_euler.y * DEGtoRAD ) * sin ( ang_euler.x * DEGtoRAD ) ;
	dy = sin ( ang_euler.y * DEGtoRAD );
	dz = cos ( ang_euler.y * DEGtoRAD ) * cos ( ang_euler.x * DEGtoRAD );
	from_pos.x = to_pos.x + (float) dx * mOrbitDist;
	from_pos.y = to_pos.y + (float) dy * mOrbitDist;
	from_pos.z = to_pos.z + (float) dz * mOrbitDist;
	updateMatricies ();
}

void Camera3D::setAngles ( float ax, float ay, float az )
{
	ang_euler = Vector3DF(ax,ay,az);
	to_pos.x = from_pos.x - (float) (cos ( ang_euler.y * DEGtoRAD ) * sin ( ang_euler.x * DEGtoRAD ) * mDolly);
	to_pos.y = from_pos.y - (float) (sin ( ang_euler.y * DEGtoRAD ) * mDolly);
	to_pos.z = from_pos.z - (float) (cos ( ang_euler.y * DEGtoRAD ) * cos ( ang_euler.x * DEGtoRAD ) * mDolly);
	updateMatricies ();
}


void Camera3D::moveRelative ( float dx, float dy, float dz )
{
	Vector3DF vec ( dx, dy, dz );
	vec *= invrot_matrix;
	to_pos += vec;
	from_pos += vec;
	updateMatricies ();
}

void Camera3D::setProjection (eProjection proj_type)
{
	mProjType = proj_type;
}

void Camera3D::updateMatricies ()
{
	Matrix4F basis;
	Vector3DF temp;	
	
	// compute camera direction vectors	--- MATCHES OpenGL's gluLookAt function (DO NOT MODIFY)
	dir_vec = to_pos;					// f vector in gluLookAt docs						
	dir_vec -= from_pos;				// eye = from_pos in gluLookAt docs
	dir_vec.Normalize ();
	side_vec = dir_vec;
	side_vec.Cross ( up_dir );
	side_vec.Normalize ();
	up_vec = side_vec;
	up_vec.Cross ( dir_vec );
	up_vec.Normalize();
	dir_vec *= -1;
	
	// construct view matrix
	rotate_matrix.Basis (side_vec, up_vec, dir_vec );
	view_matrix = rotate_matrix;
	//Matrix4F trans;
	//trans.Translate ( -from_pos.x, -from_pos.y, -from_pos.z );		// !efficiency
	view_matrix.PreTranslate ( Vector3DF(-from_pos.x, -from_pos.y, -from_pos.z ) );

	// construct projection matrix  --- MATCHES OpenGL's gluPerspective function (DO NOT MODIFY)
	float sx = (float) tan ( mFov * 0.5f * DEGtoRAD ) * mNear;
	float sy = sx / mAspect;
	proj_matrix = 0.0f;
	proj_matrix(0,0) = 2.0f*mNear / sx;				// matches OpenGL definition
	proj_matrix(1,1) = 2.0f*mNear / sy;
	proj_matrix(2,2) = -(mFar + mNear)/(mFar - mNear);			// C
	proj_matrix(2,3) = -(2.0f*mFar * mNear)/(mFar - mNear);		// D
	proj_matrix(3,2) = -1.0f;

	// construct tile projection matrix --- MATCHES OpenGL's glFrustum function (DO NOT MODIFY) 
	float l, r, t, b;
	l = -sx + 2.0f*sx*mTile.x;						// Tile is in range 0 <= x,y <= 1
	r = -sx + 2.0f*sx*mTile.z;
	t =  sy - 2.0f*sy*mTile.y;
	b =  sy - 2.0f*sy*mTile.w;
	tileproj_matrix = 0.0f;
	tileproj_matrix(0,0) = 2.0f*mNear / (r - l);
	tileproj_matrix(1,1) = 2.0f*mNear / (t - b);
	tileproj_matrix(0,2) = (r + l) / (r - l);		// A
	tileproj_matrix(1,2) = (t + b) / (t - b);		// B
	tileproj_matrix(2,2) = proj_matrix(2,2);		// C
	tileproj_matrix(2,3) = proj_matrix(2,3);		// D
	tileproj_matrix(3,2) = -1.0f; 

	/*float sx = (float) tan ( mFov * 0.5f * DEGtoRAD ) * mNear;
	float sy = sx / mAspect;
	proj_matrix = 0.0f;
	proj_matrix(0,0) = 2.0f*mNear / sx;				// matches OpenGL definition
	proj_matrix(1,1) = 2.0f*mNear / sy;
	proj_matrix(2,2) = (mFar)/(mFar - mNear);			// C
	proj_matrix(2,3) = -(mFar * mNear)/(mFar - mNear);		// D
	proj_matrix(3,2) = 1.0f;

	// construct tile projection matrix --- MATCHES OpenGL's glFrustum function (DO NOT MODIFY) 
	float l, r, t, b;
	l = -sx + 2.0f*sx*mTile.x;						// Tile is in range 0 <= x,y <= 1
	r = -sx + 2.0f*sx*mTile.z;
	t =  sy - 2.0f*sy*mTile.y;
	b =  sy - 2.0f*sy*mTile.w;
	tileproj_matrix = 0.0f;
	tileproj_matrix(0,0) = 2.0f*mNear / (r - l);
	tileproj_matrix(1,1) = 2.0f*mNear / (t - b);
	tileproj_matrix(0,2) = -(r + l) / (r - l);		// A
	tileproj_matrix(1,2) = -(t + b) / (t - b);		// B
	tileproj_matrix(2,2) = proj_matrix(2,2);		// C
	tileproj_matrix(2,3) = proj_matrix(2,3);		// D
	tileproj_matrix(3,2) = 1.0f; */

	// construct inverse rotate and inverse projection matrix
    Vector3DF tvz(0, 0, 0);
	//invrot_matrix.InverseView ( rotate_matrix.GetDataF(), Vector3DF(0,0,0) );		// Computed using rule: "Inverse of a basis rotation matrix is its transpose." (So long as translation is taken out)
	invrot_matrix.InverseView ( rotate_matrix.GetDataF(), tvz );		// Computed using rule: "Inverse of a basis rotation matrix is its transpose." (So long as translation is taken out)
	invproj_matrix.InverseProj ( tileproj_matrix.GetDataF() );							// Computed using rule: 

	updateFrustum ();
}

void Camera3D::updateFrustum ()
{
	Matrix4F mv;
	mv = tileproj_matrix;					// Compute the model-view-projection matrix
	mv *= view_matrix;
	float* mvm = mv.GetDataF();
	float t;

	// Right plane
   frustum[0][0] = mvm[ 3] - mvm[ 0];
   frustum[0][1] = mvm[ 7] - mvm[ 4];
   frustum[0][2] = mvm[11] - mvm[ 8];
   frustum[0][3] = mvm[15] - mvm[12];
   t = sqrt( frustum[0][0] * frustum[0][0] + frustum[0][1] * frustum[0][1] + frustum[0][2] * frustum[0][2] );
   frustum[0][0] /= t; frustum[0][1] /= t; frustum[0][2] /= t; frustum[0][3] /= t;
	// Left plane
   frustum[1][0] = mvm[ 3] + mvm[ 0];
   frustum[1][1] = mvm[ 7] + mvm[ 4];
   frustum[1][2] = mvm[11] + mvm[ 8];
   frustum[1][3] = mvm[15] + mvm[12];
   t = sqrt( frustum[1][0] * frustum[1][0] + frustum[1][1] * frustum[1][1] + frustum[1][2]    * frustum[1][2] );
   frustum[1][0] /= t; frustum[1][1] /= t; frustum[1][2] /= t; frustum[1][3] /= t;
	// Bottom plane
   frustum[2][0] = mvm[ 3] + mvm[ 1];
   frustum[2][1] = mvm[ 7] + mvm[ 5];
   frustum[2][2] = mvm[11] + mvm[ 9];
   frustum[2][3] = mvm[15] + mvm[13];
   t = sqrt( frustum[2][0] * frustum[2][0] + frustum[2][1] * frustum[2][1] + frustum[2][2]    * frustum[2][2] );
   frustum[2][0] /= t; frustum[2][1] /= t; frustum[2][2] /= t; frustum[2][3] /= t;
	// Top plane
   frustum[3][0] = mvm[ 3] - mvm[ 1];
   frustum[3][1] = mvm[ 7] - mvm[ 5];
   frustum[3][2] = mvm[11] - mvm[ 9];
   frustum[3][3] = mvm[15] - mvm[13];
   t = sqrt( frustum[3][0] * frustum[3][0] + frustum[3][1] * frustum[3][1] + frustum[3][2]    * frustum[3][2] );
   frustum[3][0] /= t; frustum[3][1] /= t; frustum[3][2] /= t; frustum[3][3] /= t;
	// Far plane
   frustum[4][0] = mvm[ 3] - mvm[ 2];
   frustum[4][1] = mvm[ 7] - mvm[ 6];
   frustum[4][2] = mvm[11] - mvm[10];
   frustum[4][3] = mvm[15] - mvm[14];
   t = sqrt( frustum[4][0] * frustum[4][0] + frustum[4][1] * frustum[4][1] + frustum[4][2]    * frustum[4][2] );
   frustum[4][0] /= t; frustum[4][1] /= t; frustum[4][2] /= t; frustum[4][3] /= t;
	// Near plane
   frustum[5][0] = mvm[ 3] + mvm[ 2];
   frustum[5][1] = mvm[ 7] + mvm[ 6];
   frustum[5][2] = mvm[11] + mvm[10];
   frustum[5][3] = mvm[15] + mvm[14];
   t = sqrt( frustum[5][0] * frustum[5][0] + frustum[5][1] * frustum[5][1] + frustum[5][2]    * frustum[5][2] );
   frustum[5][0] /= t; frustum[5][1] /= t; frustum[5][2] /= t; frustum[5][3] /= t;
}

float Camera3D::calculateLOD ( Vector3DF pnt, float minlod, float maxlod, float maxdist )
{
	Vector3DF vec = pnt;
	vec -= from_pos;
	float lod = minlod + ((float) vec.Length() * (maxlod-minlod) / maxdist );	
	lod = (lod < minlod) ? minlod : lod;
	lod = (lod > maxlod) ? maxlod : lod;
	return lod;
}

/*void Camera3D::setModelMatrix ()
{
	glGetFloatv ( GL_MODELVIEW_MATRIX, model_matrix.GetDataF() );
}
void Camera3D::setModelMatrix ( Matrix4F& model )
{
	model_matrix = model;
	mv_matrix = model;
	mv_matrix *= view_matrix;
	#ifdef USE_DX

	#else
		glLoadMatrixf ( mv_matrix.GetDataF() );
	#endif
}
*/

Vector3DF Camera3D::inverseRay (float x, float y, float z)
{	
	float sx = (float) tan ( mFov * 0.5f * DEGtoRAD);
	float sy = sx / mAspect;
	float tu, tv;
	tu = mTile.x + x * (mTile.z-mTile.x);
	tv = mTile.y + y * (mTile.w-mTile.y);
	Vector4DF pnt ( (tu*2.0f-1.0f) * z*sx, (1.0f-tv*2.0f) * z*sy, -z, 1 );
	pnt *= invrot_matrix;
	return pnt;
}

Vector4DF Camera3D::project ( Vector3DF& p, Matrix4F& vm )
{
	Vector4DF q = p;								// World coordinates
	
	q *= vm;										// Eye coordinates
	
	q *= proj_matrix;								// Projection 

	q /= q.w;										// Normalized Device Coordinates (w-divide)
	
	q.x *= 0.5f;
	q.y *= -0.5f;
	q.z = q.z*0.5f + 0.5f;							// Stored depth buffer value
		
	return q;
}

Vector4DF Camera3D::project ( Vector3DF& p )
{
	Vector4DF q = p;								// World coordinates
	q *= view_matrix;								// Eye coordinates

	q *= proj_matrix;								// Clip coordinates
	
	q /= q.w;										// Normalized Device Coordinates (w-divide)

	q.x *= 0.5f;
	q.y *= -0.5f;
	q.z = q.z*0.5f + 0.5f;							// Stored depth buffer value
		
	return q;
}

void PivotX::setPivot ( float x, float y, float z, float rx, float ry, float rz )
{
	from_pos.Set ( x,y,z);
	ang_euler.Set ( rx,ry,rz );
}

void PivotX::updateTform ()
{
	trans.RotateZYXT ( ang_euler, from_pos );
}



