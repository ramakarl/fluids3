//----------------------------------------------------------------------------------
// File:   app_util.h
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

#ifndef APP_UTIL
	#define APP_UTIL

	#define DEGtoRAD		(3.141592/180.0)


	typedef unsigned int			uint;
	typedef unsigned short int		ushort;

	#define CLRVAL			uint
	#define COLOR(r,g,b)	( (uint(r*255.0f)<<24) | (uint(g*255.0f)<<16) | (uint(b*255.0f)<<8) )
	#define COLORA(r,g,b,a)	( (uint(a*255.0f)<<24) | (uint(b*255.0f)<<16) | (uint(g*255.0f)<<8) | uint(r*255.0f) )
	#define ALPH(c)			(float((c>>24) & 0xFF)/255.0)
	#define BLUE(c)			(float((c>>16) & 0xFF)/255.0)
	#define GRN(c)			(float((c>>8)  & 0xFF)/255.0)
	#define RED(c)			(float( c      & 0xFF)/255.0)


	// string manip
	#include <string>
	#include <vector>
	int strToI (std::string s);
	float strToF (std::string s);
	unsigned char strToC ( std::string s );
	std::string strParse ( std::string& str, std::string lsep, std::string rsep );
	bool strGet ( std::string str, std::string& result, std::string lsep, std::string rsep );
	std::string strSplit ( std::string& str, std::string sep );
	bool strSub ( std::string str, int first, int cnt, std::string cmp );
	std::string strReplace ( std::string str, std::string delim, std::string ins );
	std::string strLTrim ( std::string str );
	std::string strRTrim ( std::string str );
	std::string strTrim ( std::string str );
	std::string strLeft ( std::string str, int n );
	int strExtract ( std::string& str, std::vector<std::string>& list );
	unsigned long getFileSize ( char* fname );
	unsigned long getFilePos ( FILE* fp );

	// From app_opengl.h or app_directx.h
	// DX/GL Device and Context are provided outside of app_util.	
	extern void app_printf ( char* format, ... );
	extern char app_getch ();
	#ifdef USE_DX
		#include <d3d11.h>
		#include <dxgi.h>
		#include <d3dcompiler.h>
		typedef	ID3D11Buffer*		BUF;
		typedef	ID3D11Texture2D*	TEX;
		extern ID3D11Device*			g_pDevice;
		extern ID3D11DeviceContext*		g_pContext;
		extern ID3D11DepthStencilState*  g_pDepthOffState;
		extern ID3D11DepthStencilState*  g_pDepthStencilState;
		extern ID3D11SamplerState*		g_pSamplerState;
		extern bool checkHR( HRESULT hr );
		extern int checkSHADER ( HRESULT hr, ID3D10Blob* blob );			
		#define IDX_NULL	0xFFFFFFFF		// DirectX - cut index
	#else
		#include "GLEW\glew.h"
		typedef	GLuint			BUF;		
		typedef	GLuint			TEX;
		extern void checkGL( char* msg );
		#define IDX_NULL	0xFF000000		// OpenGL - primitive restart index
	#endif

	#include <vector>

	//-------------------------------------- PLY FORMAT
	struct PlyProperty {            // PLY Format structures
        char                        type;
        std::string                 name;
    };
    struct PlyElement {
        int							num;
        char                        type;        // 0 = vert, 1 = face
        std::vector<PlyProperty>    prop_list;
    };
	typedef unsigned long			xref;

	#define PLY_UINT            0                    // PLY Format constants
    #define PLY_INT             1
    #define PLY_FLOAT           2
    #define PLY_LIST            3
    #define PLY_VERTS           4
    #define PLY_FACES           5

	struct Vertex {
		float	x, y, z;
		float	nx, ny, nz;
		float	tx, ty;
	};

	//------------------------------------ MESHES
	class nvMesh {
	public:
		nvMesh();

		void Clear ();
		void AddPlyElement ( char typ, int n );
		void AddPlyProperty ( char typ, std::string name );
		int AddVert ( float x, float y, float z, float tx, float ty );
		void SetNormal ( int n, float x, float y, float z );
		int AddFace ( int v0, int v1, int v2 );
		int AddFace4 ( int v0, int v1, int v2, int v3 );
		int FindPlyElem ( char typ );
		int FindPlyProp ( int elem, std::string name );
		bool LoadPly ( char* fname, float scal );	// Load PLY
		void ComputeNormals ();						// Compute surface normals
		void UpdateVBO ();							// Update GPU buffers
		void SelectVBO ();
		void Draw ( int inst );						// Draw
		void DrawPatches ( int inst );

		int getNumFaces ()		{ return (int) mNumFaces; }		

	private:

		std::vector<Vertex>		mVertices;			// Vertices
		std::vector<unsigned int> mFaceVN;			// Face data (must be uniformly 3 or 4 sided)
		int						mNumFaces;			// Num faces		
		int						mNumSides;			// Must be 3 or 4		
		std::vector<BUF>		mVBO;				// Buffers
		BUF						mVAO;

        std::vector< PlyElement* >    m_Ply;		// Ply loading
        int						m_PlyCurrElem;
		int		localPos, localNorm, localUV;
	};

	//------------------------------------ IMAGES
	#define IMG_RGB			0
	#define IMG_RGBA		1
	#define IMG_LUM			2
	
	class nvImg {
	public:
		void	Create ( int x, int y, int fmt );
		void	Fill ( float r, float g, float b, float a );
		bool	LoadPng ( char* fname );			// Load PNG
		bool	LoadTga ( char* fname );			// Load TGA
		void	UpdateTex ();						// Update GPU texture
		void	BindTex ();							// Bind GPU texture
		TEX		getTex()	{ return mTex;}
		int		getWidth()	{ return mXres;}
		int		getHeight()	{ return mYres;}

	private:
		int						mXres, mYres;
		int						mSize, mFmt;
		unsigned char*			mData;

		TEX						mTex;
		#ifdef USE_DX
			ID3D11ShaderResourceView*	mTexIV;
		#endif
	};

	//-------------------------------------- FONTS - from Tristan Lorach (OpenGLText)	
	// Read from .bin files	
    struct GlyphInfo {
       struct Pix {			// pixel oriented data       
           int u, v;
           int width, height;
           int advance;
           int offX, offY;
       };
       struct Norm {		// normalized data       
           float u, v;		// position in the map in normalized coords
           float width, height;
           float advance;
           float offX, offY;
       };
       Pix  pix;
       Norm norm;
    };
    struct FileHeader {
       int texwidth, texheight;
       struct Pix {
           int ascent;
           int descent;
           int linegap;
       };
       struct Norm {
           float ascent;
           float descent;
           float linegap;
       };
       Pix  pix;
       Norm norm;
       GlyphInfo glyphs[256];
    };

	#define GRP_TRI			0
	#define GRP_TRITEX		1
	#define GRP_LINES		2	
	#define GRP_NUM			3

	typedef unsigned long		ulong;

	//-------------------------------------- 2D DRAW SETS
	struct Vert2D {
		float	x, y, z;
		float	r, g, b, a;
		float	tx, ty;
	};
	struct Set2D {
		Set2D () { mVBO[0]=0; mVBO[1]=0; mVBO[2]=0; mVBOI[0]=0; mVBOI[1]=0; mVBOI[2]=0;}
		float	model[16];
		float	view[16];
		float	proj[16];		
		float	zfactor;
		ulong	mNum[GRP_NUM];
		ulong	mMax[GRP_NUM];			
		Vert2D*	mGeom[GRP_NUM];			// Geom data		
		ulong	mNumI[GRP_NUM];
		ulong	mMaxI[GRP_NUM];		
		uint*	mIdx[GRP_NUM];			// Index data		
		BUF		mVBO[GRP_NUM];			// GPU handles
		BUF		mVBOI[GRP_NUM];
	};

	//-------------------------------------- 2D DRAWING
	class nvDraw {
	public:		
		nvDraw ();
		void setView2D ( float w, float h );
		void setView2D ( float* model, float* view, float* proj );
		void setOrder2D ( bool zt, float zfactor );
		void updateStatic2D ( int n );
		void start2D () { start2D (false); }
		int  start2D ( bool bStatic );
		//void assignMVP ( float* model, float* view, float* proj );		
		Vert2D* allocGeom ( int cnt, int grp, Set2D* set, int& ndx );
		uint*    allocIdx  ( int cnt, int grp, Set2D* set );
		void end2D ();
		void remove2D ( int id );
		void setText ( float scale, float kern )		{ mTextScale = scale; mTextKern = kern; }
		void drawLine ( float x1, float y1, float x2, float y2, float r, float g, float b, float a );
		void drawRect ( float x1, float y1, float x2, float y2, float r, float g, float b, float a );
		void drawFill ( float x1, float y1, float x2, float y2, float r, float g, float b, float a );
		void drawTri ( float x1, float y1, float x2, float y2, float x3, float y3, float r, float g, float b, float a );
		void drawCircle ( float x1, float y1, float radius, float r, float g, float b, float a );
		void drawCircleDash ( float x1, float y1, float radius, float r, float g, float b, float a );
		void drawCircleFill ( float x1, float y1, float radius, float r, float g, float b, float a );
		void drawText ( float x1, float y1, char* msg, float r, float g, float b, float a );		
		float getTextX ( char* msg );
		float getTextY ( char* msg );
		
		void Initialize ( const char* fontName );
		bool LoadFont (const char * fontName );
		void MakeShaders2D ();		
		void UpdateVBOs ( Set2D& s );
		void SetDefaultView ( Set2D& s, float w, float h, float zf );
		void SetMatrixView ( Set2D& s, float* model, float* view, float* proj, float zf );
		void Draw ( Set2D& buf );
		void Draw ();		// do actual draw
	private:
		std::vector<Set2D>	mStatic;		// static 2D draw - saved frame-to-frame
		std::vector<Set2D>	mDynamic;		// dynamic 2D draw - discarded each frame
		int					mCurrZ;
		int					mCurr, mDynNum;
		Set2D*				mCurrSet;
		float				mWidth, mHeight;
		float				mTextScale, mTextKern;
		float				mModelMtx[16], mViewMtx[16], mProjMtx[16];		
		double				mZFactor;

		int					localPos, localClr, localUV;
		
	#ifdef USE_DX
		ID3D11Buffer*		mpMatrixBuffer[3];		// 2D model/view/proj
		ID3D11VertexShader* mVS;		// 2D shader
		ID3D11PixelShader*  mPS;
		ID3D11InputLayout*  mLO;		// 2D layout


	#else
		GLuint				mSH2D;							// 2D shader 
		GLint				mModel, mProj, mView, mFont;	// 2D shader parameters
		GLuint				mVAO;
	#endif

		nvImg				mFontImg, mWhiteImg;
		FileHeader			mGlyphInfos;
	};
	
	//-------------------------------------- 2D DRAWING - UTILITY FUNCTIONS
	extern nvDraw	g_2D;
	extern void init2D ( const char* fontName );
	extern void start2D ();
	extern void start2D (bool bStatic);
	extern void static2D ();
	extern void end2D ();
	extern void draw2D ();
	extern void setview2D ( float w, float h );
	extern void setview2D ( float* model, float* view, float* proj );
	extern void setorder2D ( bool zt, float zfactor );
	extern void updatestatic2D ( int n );
	extern void setText ( float scale, float kern );
	extern float getTextX ( char* msg );
	extern float getTextY ( char* msg );
	extern void drawLine ( float x1, float y1, float x2, float y2, float r, float g, float b, float a );
	extern void drawRect ( float x1, float y1, float x2, float y2, float r, float g, float b, float a );
	extern void drawFill ( float x1, float y1, float x2, float y2, float r, float g, float b, float a );
	extern void drawTri ( float x1, float y1, float x2, float y2, float x3, float y3, float r, float g, float b, float a );
	extern void drawCircle ( float x1, float y1, float radius, float r, float g, float b, float a );
	extern void drawCircleDash ( float x1, float y1, float radius, float r, float g, float b, float a );
	extern void drawCircleFill ( float x1, float y1, float radius, float r, float g, float b, float a );
	extern void drawText ( float x1, float y1, char* msg, float r, float g, float b, float a );
	extern void drawText ( float x1, float y1, char* msg );

	//-------------------------------------- 2D GUIS

	#define GUI_PRINT		0
	#define GUI_SLIDER		1
	#define GUI_CHECK		2
	#define GUI_TEXT		3

	#define GUI_BOOL		0
	#define GUI_INT			1
	#define GUI_FLOAT		2
	#define GUI_STR			3	
	#define GUI_INTLOG		4

	struct Gui {
		float			x, y, w, h;
		int				gtype;
		std::string		name;
		int				dtype;
		void*			data;
		float			vmin, vmax;
		int				size;
		bool			changed;
	};

	class nvGui {
	public:
		nvGui ();
		int		AddGui ( float x, float y, float w, float h, char* name, int gtype, int dtype, void* data, float vmin, float vmax );
		bool	guiChanged ( int n );
		bool	MouseDown ( float x, float y );
		bool	MouseDrag ( float x, float y );
		void	Draw ();

	private:
		std::vector<Gui>		mGui;
		int		mActiveGui;
	};

	extern nvGui	g_Gui;
	extern void		drawGui ();	
	extern int		addGui ( float x, float y, float w, float h, char* name, int gtype, int dtype, void* data, float vmin, float vmax );
	extern bool		guiMouseDown ( float x, float y );
	extern bool		guiMouseDrag ( float x, float y );
	extern bool		guiChanged ( int n );
	


	//-------------------------------------- LOAD PNG 
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

#ifndef LODEPNG_H
#define LODEPNG_H

#include <string.h> /*for size_t*/

#ifdef __cplusplus
#include <vector>
#include <string>
#endif /*__cplusplus*/

/*
The following #defines are used to create code sections. They can be disabled
to disable code sections, which can give faster compile time and smaller binary.
The "NO_COMPILE" defines are designed to be used to pass as defines to the
compiler command to disable them without modifying this header, e.g.
-DLODEPNG_NO_COMPILE_ZLIB for gcc.
*/
/*deflate & zlib. If disabled, you must specify alternative zlib functions in
the custom_zlib field of the compress and decompress settings*/
#ifndef LODEPNG_NO_COMPILE_ZLIB
#define LODEPNG_COMPILE_ZLIB
#endif
/*png encoder and png decoder*/
#ifndef LODEPNG_NO_COMPILE_PNG
#define LODEPNG_COMPILE_PNG
#endif
/*deflate&zlib decoder and png decoder*/
#ifndef LODEPNG_NO_COMPILE_DECODER
#define LODEPNG_COMPILE_DECODER
#endif
/*deflate&zlib encoder and png encoder*/
#ifndef LODEPNG_NO_COMPILE_ENCODER
#define LODEPNG_COMPILE_ENCODER
#endif
/*the optional built in harddisk file loading and saving functions*/
#ifndef LODEPNG_NO_COMPILE_DISK
#define LODEPNG_COMPILE_DISK
#endif
/*support for chunks other than IHDR, IDAT, PLTE, tRNS, IEND: ancillary and unknown chunks*/
#ifndef LODEPNG_NO_COMPILE_ANCILLARY_CHUNKS
#define LODEPNG_COMPILE_ANCILLARY_CHUNKS
#endif
/*ability to convert error numerical codes to English text string*/
#ifndef LODEPNG_NO_COMPILE_ERROR_TEXT
#define LODEPNG_COMPILE_ERROR_TEXT
#endif
/*compile the C++ version (you can disable the C++ wrapper here even when compiling for C++)*/
#ifdef __cplusplus
#ifndef LODEPNG_NO_COMPILE_CPP
#define LODEPNG_COMPILE_CPP
#endif
#endif

#ifdef LODEPNG_COMPILE_PNG
/*The PNG color types (also used for raw).*/
typedef enum LodePNGColorType
{
  LCT_GREY = 0, /*greyscale: 1,2,4,8,16 bit*/
  LCT_RGB = 2, /*RGB: 8,16 bit*/
  LCT_PALETTE = 3, /*palette: 1,2,4,8 bit*/
  LCT_GREY_ALPHA = 4, /*greyscale with alpha: 8,16 bit*/
  LCT_RGBA = 6 /*RGB with alpha: 8,16 bit*/
} LodePNGColorType;

#ifdef LODEPNG_COMPILE_DECODER
/*
Converts PNG data in memory to raw pixel data.
out: Output parameter. Pointer to buffer that will contain the raw pixel data.
     After decoding, its size is w * h * (bytes per pixel) bytes larger than
     initially. Bytes per pixel depends on colortype and bitdepth.
     Must be freed after usage with free(*out).
     Note: for 16-bit per channel colors, uses big endian format like PNG does.
w: Output parameter. Pointer to width of pixel data.
h: Output parameter. Pointer to height of pixel data.
in: Memory buffer with the PNG file.
insize: size of the in buffer.
colortype: the desired color type for the raw output image. See explanation on PNG color types.
bitdepth: the desired bit depth for the raw output image. See explanation on PNG color types.
Return value: LodePNG error code (0 means no error).
*/
unsigned lodepng_decode_memory(unsigned char** out, unsigned* w, unsigned* h,
                               const unsigned char* in, size_t insize,
                               LodePNGColorType colortype, unsigned bitdepth);

/*Same as lodepng_decode_memory, but always decodes to 32-bit RGBA raw image*/
unsigned lodepng_decode32(unsigned char** out, unsigned* w, unsigned* h,
                          const unsigned char* in, size_t insize);

/*Same as lodepng_decode_memory, but always decodes to 24-bit RGB raw image*/
unsigned lodepng_decode24(unsigned char** out, unsigned* w, unsigned* h,
                          const unsigned char* in, size_t insize);

#ifdef LODEPNG_COMPILE_DISK
/*
Load PNG from disk, from file with given name.
Same as the other decode functions, but instead takes a filename as input.
*/
unsigned lodepng_decode_file(unsigned char** out, unsigned* w, unsigned* h,
                             const char* filename,
                             LodePNGColorType colortype, unsigned bitdepth);

/*Same as lodepng_decode_file, but always decodes to 32-bit RGBA raw image.*/
unsigned lodepng_decode32_file(unsigned char** out, unsigned* w, unsigned* h,
                               const char* filename);

/*Same as lodepng_decode_file, but always decodes to 24-bit RGB raw image.*/
unsigned lodepng_decode24_file(unsigned char** out, unsigned* w, unsigned* h,
                               const char* filename);
#endif /*LODEPNG_COMPILE_DISK*/
#endif /*LODEPNG_COMPILE_DECODER*/


#ifdef LODEPNG_COMPILE_ENCODER
/*
Converts raw pixel data into a PNG image in memory. The colortype and bitdepth
  of the output PNG image cannot be chosen, they are automatically determined
  by the colortype, bitdepth and content of the input pixel data.
  Note: for 16-bit per channel colors, needs big endian format like PNG does.
out: Output parameter. Pointer to buffer that will contain the PNG image data.
     Must be freed after usage with free(*out).
outsize: Output parameter. Pointer to the size in bytes of the out buffer.
image: The raw pixel data to encode. The size of this buffer should be
       w * h * (bytes per pixel), bytes per pixel depends on colortype and bitdepth.
w: width of the raw pixel data in pixels.
h: height of the raw pixel data in pixels.
colortype: the color type of the raw input image. See explanation on PNG color types.
bitdepth: the bit depth of the raw input image. See explanation on PNG color types.
Return value: LodePNG error code (0 means no error).
*/
unsigned lodepng_encode_memory(unsigned char** out, size_t* outsize,
                               const unsigned char* image, unsigned w, unsigned h,
                               LodePNGColorType colortype, unsigned bitdepth);

/*Same as lodepng_encode_memory, but always encodes from 32-bit RGBA raw image.*/
unsigned lodepng_encode32(unsigned char** out, size_t* outsize,
                          const unsigned char* image, unsigned w, unsigned h);

/*Same as lodepng_encode_memory, but always encodes from 24-bit RGB raw image.*/
unsigned lodepng_encode24(unsigned char** out, size_t* outsize,
                          const unsigned char* image, unsigned w, unsigned h);

#ifdef LODEPNG_COMPILE_DISK
/*
Converts raw pixel data into a PNG file on disk.
Same as the other encode functions, but instead takes a filename as output.
NOTE: This overwrites existing files without warning!
*/
unsigned lodepng_encode_file(const char* filename,
                             const unsigned char* image, unsigned w, unsigned h,
                             LodePNGColorType colortype, unsigned bitdepth);

/*Same as lodepng_encode_file, but always encodes from 32-bit RGBA raw image.*/
unsigned lodepng_encode32_file(const char* filename,
                               const unsigned char* image, unsigned w, unsigned h);

/*Same as lodepng_encode_file, but always encodes from 24-bit RGB raw image.*/
unsigned lodepng_encode24_file(const char* filename,
                               const unsigned char* image, unsigned w, unsigned h);
#endif /*LODEPNG_COMPILE_DISK*/
#endif /*LODEPNG_COMPILE_ENCODER*/


#ifdef LODEPNG_COMPILE_CPP
namespace lodepng
{
#ifdef LODEPNG_COMPILE_DECODER
/*Same as lodepng_decode_memory, but decodes to an std::vector.*/
unsigned decode(std::vector<unsigned char>& out, unsigned& w, unsigned& h,
                const unsigned char* in, size_t insize,
                LodePNGColorType colortype = LCT_RGBA, unsigned bitdepth = 8);
unsigned decode(std::vector<unsigned char>& out, unsigned& w, unsigned& h,
                const std::vector<unsigned char>& in,
                LodePNGColorType colortype = LCT_RGBA, unsigned bitdepth = 8);
#ifdef LODEPNG_COMPILE_DISK
/*
Converts PNG file from disk to raw pixel data in memory.
Same as the other decode functions, but instead takes a filename as input.
*/
unsigned decode(std::vector<unsigned char>& out, unsigned& w, unsigned& h,
                const std::string& filename,
                LodePNGColorType colortype = LCT_RGBA, unsigned bitdepth = 8);
#endif //LODEPNG_COMPILE_DISK
#endif //LODEPNG_COMPILE_DECODER

#ifdef LODEPNG_COMPILE_ENCODER
/*Same as lodepng_encode_memory, but encodes to an std::vector.*/
unsigned encode(std::vector<unsigned char>& out,
                const unsigned char* in, unsigned w, unsigned h,
                LodePNGColorType colortype = LCT_RGBA, unsigned bitdepth = 8);
unsigned encode(std::vector<unsigned char>& out,
                const std::vector<unsigned char>& in, unsigned w, unsigned h,
                LodePNGColorType colortype = LCT_RGBA, unsigned bitdepth = 8);
#ifdef LODEPNG_COMPILE_DISK
/*
Converts 32-bit RGBA raw pixel data into a PNG file on disk.
Same as the other encode functions, but instead takes a filename as output.
NOTE: This overwrites existing files without warning!
*/
unsigned encode(const std::string& filename,
                const unsigned char* in, unsigned w, unsigned h,
                LodePNGColorType colortype = LCT_RGBA, unsigned bitdepth = 8);
unsigned encode(const std::string& filename,
                const std::vector<unsigned char>& in, unsigned w, unsigned h,
                LodePNGColorType colortype = LCT_RGBA, unsigned bitdepth = 8);
#endif //LODEPNG_COMPILE_DISK
#endif //LODEPNG_COMPILE_ENCODER
} //namespace lodepng
#endif /*LODEPNG_COMPILE_CPP*/
#endif /*LODEPNG_COMPILE_PNG*/

#ifdef LODEPNG_COMPILE_ERROR_TEXT
/*Returns an English description of the numerical error code.*/
const char* lodepng_error_text(unsigned code);
#endif /*LODEPNG_COMPILE_ERROR_TEXT*/

#ifdef LODEPNG_COMPILE_DECODER
/*Settings for zlib decompression*/
typedef struct LodePNGDecompressSettings LodePNGDecompressSettings;
struct LodePNGDecompressSettings
{
  unsigned ignore_adler32; /*if 1, continue and don't give an error message if the Adler32 checksum is corrupted*/

  /*use custom zlib decoder instead of built in one (default: null)*/
  unsigned (*custom_zlib)(unsigned char**, size_t*,
                          const unsigned char*, size_t,
                          const LodePNGDecompressSettings*);
  /*use custom deflate decoder instead of built in one (default: null)
  if custom_zlib is used, custom_deflate is ignored since only the built in
  zlib function will call custom_deflate*/
  unsigned (*custom_inflate)(unsigned char**, size_t*,
                             const unsigned char*, size_t,
                             const LodePNGDecompressSettings*);

  void* custom_context; /*optional custom settings for custom functions*/
};

extern const LodePNGDecompressSettings lodepng_default_decompress_settings;
void lodepng_decompress_settings_init(LodePNGDecompressSettings* settings);
#endif /*LODEPNG_COMPILE_DECODER*/

#ifdef LODEPNG_COMPILE_ENCODER
/*
Settings for zlib compression. Tweaking these settings tweaks the balance
between speed and compression ratio.
*/
typedef struct LodePNGCompressSettings LodePNGCompressSettings;
struct LodePNGCompressSettings /*deflate = compress*/
{
  /*LZ77 related settings*/
  unsigned btype; /*the block type for LZ (0, 1, 2 or 3, see zlib standard). Should be 2 for proper compression.*/
  unsigned use_lz77; /*whether or not to use LZ77. Should be 1 for proper compression.*/
  unsigned windowsize; /*the maximum is 32768, higher gives more compression but is slower. Typical value: 2048.*/
  unsigned minmatch; /*mininum lz77 length. 3 is normally best, 6 can be better for some PNGs. Default: 0*/
  unsigned nicematch; /*stop searching if >= this length found. Set to 258 for best compression. Default: 128*/
  unsigned lazymatching; /*use lazy matching: better compression but a bit slower. Default: true*/

  /*use custom zlib encoder instead of built in one (default: null)*/
  unsigned (*custom_zlib)(unsigned char**, size_t*,
                          const unsigned char*, size_t,
                          const LodePNGCompressSettings*);
  /*use custom deflate encoder instead of built in one (default: null)
  if custom_zlib is used, custom_deflate is ignored since only the built in
  zlib function will call custom_deflate*/
  unsigned (*custom_deflate)(unsigned char**, size_t*,
                             const unsigned char*, size_t,
                             const LodePNGCompressSettings*);

  void* custom_context; /*optional custom settings for custom functions*/
};

extern const LodePNGCompressSettings lodepng_default_compress_settings;
void lodepng_compress_settings_init(LodePNGCompressSettings* settings);
#endif /*LODEPNG_COMPILE_ENCODER*/

#ifdef LODEPNG_COMPILE_PNG
/*
Color mode of an image. Contains all information required to decode the pixel
bits to RGBA colors. This information is the same as used in the PNG file
format, and is used both for PNG and raw image data in LodePNG.
*/
typedef struct LodePNGColorMode
{
  /*header (IHDR)*/
  LodePNGColorType colortype; /*color type, see PNG standard or documentation further in this header file*/
  unsigned bitdepth;  /*bits per sample, see PNG standard or documentation further in this header file*/

  /*
  palette (PLTE and tRNS)

  Dynamically allocated with the colors of the palette, including alpha.
  When encoding a PNG, to store your colors in the palette of the LodePNGColorMode, first use
  lodepng_palette_clear, then for each color use lodepng_palette_add.
  If you encode an image without alpha with palette, don't forget to put value 255 in each A byte of the palette.

  When decoding, by default you can ignore this palette, since LodePNG already
  fills the palette colors in the pixels of the raw RGBA output.

  The palette is only supported for color type 3.
  */
  unsigned char* palette; /*palette in RGBARGBA... order*/
  size_t palettesize; /*palette size in number of colors (amount of bytes is 4 * palettesize)*/

  /*
  transparent color key (tRNS)

  This color uses the same bit depth as the bitdepth value in this struct, which can be 1-bit to 16-bit.
  For greyscale PNGs, r, g and b will all 3 be set to the same.

  When decoding, by default you can ignore this information, since LodePNG sets
  pixels with this key to transparent already in the raw RGBA output.

  The color key is only supported for color types 0 and 2.
  */
  unsigned key_defined; /*is a transparent color key given? 0 = false, 1 = true*/
  unsigned key_r;       /*red/greyscale component of color key*/
  unsigned key_g;       /*green component of color key*/
  unsigned key_b;       /*blue component of color key*/
} LodePNGColorMode;

/*init, cleanup and copy functions to use with this struct*/
void lodepng_color_mode_init(LodePNGColorMode* info);
void lodepng_color_mode_cleanup(LodePNGColorMode* info);
/*return value is error code (0 means no error)*/
unsigned lodepng_color_mode_copy(LodePNGColorMode* dest, const LodePNGColorMode* source);

void lodepng_palette_clear(LodePNGColorMode* info);
/*add 1 color to the palette*/
unsigned lodepng_palette_add(LodePNGColorMode* info,
                             unsigned char r, unsigned char g, unsigned char b, unsigned char a);

/*get the total amount of bits per pixel, based on colortype and bitdepth in the struct*/
unsigned lodepng_get_bpp(const LodePNGColorMode* info);
/*get the amount of color channels used, based on colortype in the struct.
If a palette is used, it counts as 1 channel.*/
unsigned lodepng_get_channels(const LodePNGColorMode* info);
/*is it a greyscale type? (only colortype 0 or 4)*/
unsigned lodepng_is_greyscale_type(const LodePNGColorMode* info);
/*has it got an alpha channel? (only colortype 2 or 6)*/
unsigned lodepng_is_alpha_type(const LodePNGColorMode* info);
/*has it got a palette? (only colortype 3)*/
unsigned lodepng_is_palette_type(const LodePNGColorMode* info);
/*only returns true if there is a palette and there is a value in the palette with alpha < 255.
Loops through the palette to check this.*/
unsigned lodepng_has_palette_alpha(const LodePNGColorMode* info);
/*
Check if the given color info indicates the possibility of having non-opaque pixels in the PNG image.
Returns true if the image can have translucent or invisible pixels (it still be opaque if it doesn't use such pixels).
Returns false if the image can only have opaque pixels.
In detail, it returns true only if it's a color type with alpha, or has a palette with non-opaque values,
or if "key_defined" is true.
*/
unsigned lodepng_can_have_alpha(const LodePNGColorMode* info);
/*Returns the byte size of a raw image buffer with given width, height and color mode*/
size_t lodepng_get_raw_size(unsigned w, unsigned h, const LodePNGColorMode* color);

#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
/*The information of a Time chunk in PNG.*/
typedef struct LodePNGTime
{
  unsigned year;    /*2 bytes used (0-65535)*/
  unsigned month;   /*1-12*/
  unsigned day;     /*1-31*/
  unsigned hour;    /*0-23*/
  unsigned minute;  /*0-59*/
  unsigned second;  /*0-60 (to allow for leap seconds)*/
} LodePNGTime;
#endif /*LODEPNG_COMPILE_ANCILLARY_CHUNKS*/

/*Information about the PNG image, except pixels, width and height.*/
typedef struct LodePNGInfo
{
  /*header (IHDR), palette (PLTE) and transparency (tRNS) chunks*/
  unsigned compression_method;/*compression method of the original file. Always 0.*/
  unsigned filter_method;     /*filter method of the original file*/
  unsigned interlace_method;  /*interlace method of the original file*/
  LodePNGColorMode color;     /*color type and bits, palette and transparency of the PNG file*/

#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
  /*
  suggested background color chunk (bKGD)
  This color uses the same color mode as the PNG (except alpha channel), which can be 1-bit to 16-bit.

  For greyscale PNGs, r, g and b will all 3 be set to the same. When encoding
  the encoder writes the red one. For palette PNGs: When decoding, the RGB value
  will be stored, not a palette index. But when encoding, specify the index of
  the palette in background_r, the other two are then ignored.

  The decoder does not use this background color to edit the color of pixels.
  */
  unsigned background_defined; /*is a suggested background color given?*/
  unsigned background_r;       /*red component of suggested background color*/
  unsigned background_g;       /*green component of suggested background color*/
  unsigned background_b;       /*blue component of suggested background color*/

  /*
  non-international text chunks (tEXt and zTXt)

  The char** arrays each contain num strings. The actual messages are in
  text_strings, while text_keys are keywords that give a short description what
  the actual text represents, e.g. Title, Author, Description, or anything else.

  A keyword is minimum 1 character and maximum 79 characters long. It's
  discouraged to use a single line length longer than 79 characters for texts.

  Don't allocate these text buffers yourself. Use the init/cleanup functions
  correctly and use lodepng_add_text and lodepng_clear_text.
  */
  size_t text_num; /*the amount of texts in these char** buffers (there may be more texts in itext)*/
  char** text_keys; /*the keyword of a text chunk (e.g. "Comment")*/
  char** text_strings; /*the actual text*/

  /*
  international text chunks (iTXt)
  Similar to the non-international text chunks, but with additional strings
  "langtags" and "transkeys".
  */
  size_t itext_num; /*the amount of international texts in this PNG*/
  char** itext_keys; /*the English keyword of the text chunk (e.g. "Comment")*/
  char** itext_langtags; /*language tag for this text's language, ISO/IEC 646 string, e.g. ISO 639 language tag*/
  char** itext_transkeys; /*keyword translated to the international language - UTF-8 string*/
  char** itext_strings; /*the actual international text - UTF-8 string*/

  /*time chunk (tIME)*/
  unsigned time_defined; /*set to 1 to make the encoder generate a tIME chunk*/
  LodePNGTime time;

  /*phys chunk (pHYs)*/
  unsigned phys_defined; /*if 0, there is no pHYs chunk and the values below are undefined, if 1 else there is one*/
  unsigned phys_x; /*pixels per unit in x direction*/
  unsigned phys_y; /*pixels per unit in y direction*/
  unsigned phys_unit; /*may be 0 (unknown unit) or 1 (metre)*/

  /*
  unknown chunks
  There are 3 buffers, one for each position in the PNG where unknown chunks can appear
  each buffer contains all unknown chunks for that position consecutively
  The 3 buffers are the unknown chunks between certain critical chunks:
  0: IHDR-PLTE, 1: PLTE-IDAT, 2: IDAT-IEND
  Do not allocate or traverse this data yourself. Use the chunk traversing functions declared
  later, such as lodepng_chunk_next and lodepng_chunk_append, to read/write this struct.
  */
  unsigned char* unknown_chunks_data[3];
  size_t unknown_chunks_size[3]; /*size in bytes of the unknown chunks, given for protection*/
#endif /*LODEPNG_COMPILE_ANCILLARY_CHUNKS*/
} LodePNGInfo;

/*init, cleanup and copy functions to use with this struct*/
void lodepng_info_init(LodePNGInfo* info);
void lodepng_info_cleanup(LodePNGInfo* info);
/*return value is error code (0 means no error)*/
unsigned lodepng_info_copy(LodePNGInfo* dest, const LodePNGInfo* source);

#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
void lodepng_clear_text(LodePNGInfo* info); /*use this to clear the texts again after you filled them in*/
unsigned lodepng_add_text(LodePNGInfo* info, const char* key, const char* str); /*push back both texts at once*/

void lodepng_clear_itext(LodePNGInfo* info); /*use this to clear the itexts again after you filled them in*/
unsigned lodepng_add_itext(LodePNGInfo* info, const char* key, const char* langtag,
                           const char* transkey, const char* str); /*push back the 4 texts of 1 chunk at once*/
#endif /*LODEPNG_COMPILE_ANCILLARY_CHUNKS*/

/*
Converts raw buffer from one color type to another color type, based on
LodePNGColorMode structs to describe the input and output color type.
See the reference manual at the end of this header file to see which color conversions are supported.
return value = LodePNG error code (0 if all went ok, an error if the conversion isn't supported)
The out buffer must have size (w * h * bpp + 7) / 8, where bpp is the bits per pixel
of the output color type (lodepng_get_bpp)
Note: for 16-bit per channel colors, uses big endian format like PNG does.
*/
unsigned lodepng_convert(unsigned char* out, const unsigned char* in,
                         LodePNGColorMode* mode_out, LodePNGColorMode* mode_in,
                         unsigned w, unsigned h);


#ifdef LODEPNG_COMPILE_DECODER
/*
Settings for the decoder. This contains settings for the PNG and the Zlib
decoder, but not the Info settings from the Info structs.
*/
typedef struct LodePNGDecoderSettings
{
  LodePNGDecompressSettings zlibsettings; /*in here is the setting to ignore Adler32 checksums*/

  unsigned ignore_crc; /*ignore CRC checksums*/
  unsigned color_convert; /*whether to convert the PNG to the color type you want. Default: yes*/

#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
  unsigned read_text_chunks; /*if false but remember_unknown_chunks is true, they're stored in the unknown chunks*/
  /*store all bytes from unknown chunks in the LodePNGInfo (off by default, useful for a png editor)*/
  unsigned remember_unknown_chunks;
#endif /*LODEPNG_COMPILE_ANCILLARY_CHUNKS*/
} LodePNGDecoderSettings;

void lodepng_decoder_settings_init(LodePNGDecoderSettings* settings);
#endif /*LODEPNG_COMPILE_DECODER*/

#ifdef LODEPNG_COMPILE_ENCODER
/*automatically use color type with less bits per pixel if losslessly possible. Default: AUTO*/
typedef enum LodePNGFilterStrategy
{
  /*every filter at zero*/
  LFS_ZERO,
  /*Use filter that gives minumum sum, as described in the official PNG filter heuristic.*/
  LFS_MINSUM,
  /*Use the filter type that gives smallest Shannon entropy for this scanline. Depending
  on the image, this is better or worse than minsum.*/
  LFS_ENTROPY,
  /*
  Brute-force-search PNG filters by compressing each filter for each scanline.
  Experimental, very slow, and only rarely gives better compression than MINSUM.
  */
  LFS_BRUTE_FORCE,
  /*use predefined_filters buffer: you specify the filter type for each scanline*/
  LFS_PREDEFINED
} LodePNGFilterStrategy;

/*automatically use color type with less bits per pixel if losslessly possible. Default: LAC_AUTO*/
typedef enum LodePNGAutoConvert
{
  LAC_NO, /*use color type user requested*/
  LAC_ALPHA, /*use color type user requested, but if only opaque pixels and RGBA or grey+alpha, use RGB or grey*/
  LAC_AUTO, /*use PNG color type that can losslessly represent the uncompressed image the smallest possible*/
  /*
  like AUTO, but do not choose 1, 2 or 4 bit per pixel types.
  sometimes a PNG image compresses worse if less than 8 bits per pixels.
  */
  LAC_AUTO_NO_NIBBLES,
  /*
  like AUTO, but never choose palette color type. For small images, encoding
  the palette may take more bytes than what is gained. Note that AUTO also
  already prevents encoding the palette for extremely small images, but that may
  not be sufficient because due to the compression it cannot predict when to
  switch.
  */
  LAC_AUTO_NO_PALETTE,
  LAC_AUTO_NO_NIBBLES_NO_PALETTE
} LodePNGAutoConvert;


/*Settings for the encoder.*/
typedef struct LodePNGEncoderSettings
{
  LodePNGCompressSettings zlibsettings; /*settings for the zlib encoder, such as window size, ...*/

  LodePNGAutoConvert auto_convert; /*how to automatically choose output PNG color type, if at all*/

  /*If true, follows the official PNG heuristic: if the PNG uses a palette or lower than
  8 bit depth, set all filters to zero. Otherwise use the filter_strategy. Note that to
  completely follow the official PNG heuristic, filter_palette_zero must be true and
  filter_strategy must be LFS_MINSUM*/
  unsigned filter_palette_zero;
  /*Which filter strategy to use when not using zeroes due to filter_palette_zero.
  Set filter_palette_zero to 0 to ensure always using your chosen strategy. Default: LFS_MINSUM*/
  LodePNGFilterStrategy filter_strategy;
  /*used if filter_strategy is LFS_PREDEFINED. In that case, this must point to a buffer with
  the same length as the amount of scanlines in the image, and each value must <= 5. You
  have to cleanup this buffer, LodePNG will never free it. Don't forget that filter_palette_zero
  must be set to 0 to ensure this is also used on palette or low bitdepth images.*/
  unsigned char* predefined_filters;

  /*force creating a PLTE chunk if colortype is 2 or 6 (= a suggested palette).
  If colortype is 3, PLTE is _always_ created.*/
  unsigned force_palette;
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
  /*add LodePNG identifier and version as a text chunk, for debugging*/
  unsigned add_id;
  /*encode text chunks as zTXt chunks instead of tEXt chunks, and use compression in iTXt chunks*/
  unsigned text_compression;
#endif /*LODEPNG_COMPILE_ANCILLARY_CHUNKS*/
} LodePNGEncoderSettings;

void lodepng_encoder_settings_init(LodePNGEncoderSettings* settings);
#endif /*LODEPNG_COMPILE_ENCODER*/


#if defined(LODEPNG_COMPILE_DECODER) || defined(LODEPNG_COMPILE_ENCODER)
/*The settings, state and information for extended encoding and decoding.*/
typedef struct LodePNGState
{
#ifdef LODEPNG_COMPILE_DECODER
  LodePNGDecoderSettings decoder; /*the decoding settings*/
#endif /*LODEPNG_COMPILE_DECODER*/
#ifdef LODEPNG_COMPILE_ENCODER
  LodePNGEncoderSettings encoder; /*the encoding settings*/
#endif /*LODEPNG_COMPILE_ENCODER*/
  LodePNGColorMode info_raw; /*specifies the format in which you would like to get the raw pixel buffer*/
  LodePNGInfo info_png; /*info of the PNG image obtained after decoding*/
  unsigned error;
#ifdef LODEPNG_COMPILE_CPP
  //For the lodepng::State subclass.
  virtual ~LodePNGState(){}
#endif
} LodePNGState;

/*init, cleanup and copy functions to use with this struct*/
void lodepng_state_init(LodePNGState* state);
void lodepng_state_cleanup(LodePNGState* state);
void lodepng_state_copy(LodePNGState* dest, const LodePNGState* source);
#endif /* defined(LODEPNG_COMPILE_DECODER) || defined(LODEPNG_COMPILE_ENCODER) */

#ifdef LODEPNG_COMPILE_DECODER
/*
Same as lodepng_decode_memory, but uses a LodePNGState to allow custom settings and
getting much more information about the PNG image and color mode.
*/
unsigned lodepng_decode(unsigned char** out, unsigned* w, unsigned* h,
                        LodePNGState* state,
                        const unsigned char* in, size_t insize);

/*
Read the PNG header, but not the actual data. This returns only the information
that is in the header chunk of the PNG, such as width, height and color type. The
information is placed in the info_png field of the LodePNGState.
*/
unsigned lodepng_inspect(unsigned* w, unsigned* h,
                         LodePNGState* state,
                         const unsigned char* in, size_t insize);
#endif /*LODEPNG_COMPILE_DECODER*/


#ifdef LODEPNG_COMPILE_ENCODER
/*This function allocates the out buffer with standard malloc and stores the size in *outsize.*/
unsigned lodepng_encode(unsigned char** out, size_t* outsize,
                        const unsigned char* image, unsigned w, unsigned h,
                        LodePNGState* state);
#endif /*LODEPNG_COMPILE_ENCODER*/

/*
The lodepng_chunk functions are normally not needed, except to traverse the
unknown chunks stored in the LodePNGInfo struct, or add new ones to it.
It also allows traversing the chunks of an encoded PNG file yourself.

PNG standard chunk naming conventions:
First byte: uppercase = critical, lowercase = ancillary
Second byte: uppercase = public, lowercase = private
Third byte: must be uppercase
Fourth byte: uppercase = unsafe to copy, lowercase = safe to copy
*/

/*get the length of the data of the chunk. Total chunk length has 12 bytes more.*/
unsigned lodepng_chunk_length(const unsigned char* chunk);

/*puts the 4-byte type in null terminated string*/
void lodepng_chunk_type(char type[5], const unsigned char* chunk);

/*check if the type is the given type*/
unsigned char lodepng_chunk_type_equals(const unsigned char* chunk, const char* type);

/*0: it's one of the critical chunk types, 1: it's an ancillary chunk (see PNG standard)*/
unsigned char lodepng_chunk_ancillary(const unsigned char* chunk);

/*0: public, 1: private (see PNG standard)*/
unsigned char lodepng_chunk_private(const unsigned char* chunk);

/*0: the chunk is unsafe to copy, 1: the chunk is safe to copy (see PNG standard)*/
unsigned char lodepng_chunk_safetocopy(const unsigned char* chunk);

/*get pointer to the data of the chunk, where the input points to the header of the chunk*/
unsigned char* lodepng_chunk_data(unsigned char* chunk);
const unsigned char* lodepng_chunk_data_const(const unsigned char* chunk);

/*returns 0 if the crc is correct, 1 if it's incorrect (0 for OK as usual!)*/
unsigned lodepng_chunk_check_crc(const unsigned char* chunk);

/*generates the correct CRC from the data and puts it in the last 4 bytes of the chunk*/
void lodepng_chunk_generate_crc(unsigned char* chunk);

/*iterate to next chunks. don't use on IEND chunk, as there is no next chunk then*/
unsigned char* lodepng_chunk_next(unsigned char* chunk);
const unsigned char* lodepng_chunk_next_const(const unsigned char* chunk);

/*
Appends chunk to the data in out. The given chunk should already have its chunk header.
The out variable and outlength are updated to reflect the new reallocated buffer.
Returns error code (0 if it went ok)
*/
unsigned lodepng_chunk_append(unsigned char** out, size_t* outlength, const unsigned char* chunk);

/*
Appends new chunk to out. The chunk to append is given by giving its length, type
and data separately. The type is a 4-letter string.
The out variable and outlength are updated to reflect the new reallocated buffer.
Returne error code (0 if it went ok)
*/
unsigned lodepng_chunk_create(unsigned char** out, size_t* outlength, unsigned length,
                              const char* type, const unsigned char* data);


/*Calculate CRC32 of buffer*/
unsigned lodepng_crc32(const unsigned char* buf, size_t len);
#endif /*LODEPNG_COMPILE_PNG*/


#ifdef LODEPNG_COMPILE_ZLIB
/*
This zlib part can be used independently to zlib compress and decompress a
buffer. It cannot be used to create gzip files however, and it only supports the
part of zlib that is required for PNG, it does not support dictionaries.
*/

#ifdef LODEPNG_COMPILE_DECODER
/*Inflate a buffer. Inflate is the decompression step of deflate. Out buffer must be freed after use.*/
unsigned lodepng_inflate(unsigned char** out, size_t* outsize,
                         const unsigned char* in, size_t insize,
                         const LodePNGDecompressSettings* settings);

/*
Decompresses Zlib data. Reallocates the out buffer and appends the data. The
data must be according to the zlib specification.
Either, *out must be NULL and *outsize must be 0, or, *out must be a valid
buffer and *outsize its size in bytes. out must be freed by user after usage.
*/
unsigned lodepng_zlib_decompress(unsigned char** out, size_t* outsize,
                                 const unsigned char* in, size_t insize,
                                 const LodePNGDecompressSettings* settings);
#endif /*LODEPNG_COMPILE_DECODER*/

#ifdef LODEPNG_COMPILE_ENCODER
/*
Compresses data with Zlib. Reallocates the out buffer and appends the data.
Zlib adds a small header and trailer around the deflate data.
The data is output in the format of the zlib specification.
Either, *out must be NULL and *outsize must be 0, or, *out must be a valid
buffer and *outsize its size in bytes. out must be freed by user after usage.
*/
unsigned lodepng_zlib_compress(unsigned char** out, size_t* outsize,
                               const unsigned char* in, size_t insize,
                               const LodePNGCompressSettings* settings);

/*
Find length-limited Huffman code for given frequencies. This function is in the
public interface only for tests, it's used internally by lodepng_deflate.
*/
unsigned lodepng_huffman_code_lengths(unsigned* lengths, const unsigned* frequencies,
                                      size_t numcodes, unsigned maxbitlen);

/*Compress a buffer with deflate. See RFC 1951. Out buffer must be freed after use.*/
unsigned lodepng_deflate(unsigned char** out, size_t* outsize,
                         const unsigned char* in, size_t insize,
                         const LodePNGCompressSettings* settings);

#endif /*LODEPNG_COMPILE_ENCODER*/
#endif /*LODEPNG_COMPILE_ZLIB*/

#ifdef LODEPNG_COMPILE_DISK
/*
Load a file from disk into buffer. The function allocates the out buffer, and
after usage you should free it.
out: output parameter, contains pointer to loaded buffer.
outsize: output parameter, size of the allocated out buffer
filename: the path to the file to load
return value: error code (0 means ok)
*/
unsigned lodepng_load_file(unsigned char** out, size_t* outsize, const char* filename);

/*
Save a file from buffer to disk. Warning, if it exists, this function overwrites
the file without warning!
buffer: the buffer to write
buffersize: size of the buffer to write
filename: the path to the file to save to
return value: error code (0 means ok)
*/
unsigned lodepng_save_file(const unsigned char* buffer, size_t buffersize, const char* filename);
#endif /*LODEPNG_COMPILE_DISK*/

#ifdef LODEPNG_COMPILE_CPP
//The LodePNG C++ wrapper uses std::vectors instead of manually allocated memory buffers.
namespace lodepng
{
#ifdef LODEPNG_COMPILE_PNG
class State : public LodePNGState
{
  public:
    State();
    State(const State& other);
    virtual ~State();
    State& operator=(const State& other);
};

#ifdef LODEPNG_COMPILE_DECODER
//Same as other lodepng::decode, but using a State for more settings and information.
unsigned decode(std::vector<unsigned char>& out, unsigned& w, unsigned& h,
                State& state,
                const unsigned char* in, size_t insize);
unsigned decode(std::vector<unsigned char>& out, unsigned& w, unsigned& h,
                State& state,
                const std::vector<unsigned char>& in);
#endif /*LODEPNG_COMPILE_DECODER*/

#ifdef LODEPNG_COMPILE_ENCODER
//Same as other lodepng::encode, but using a State for more settings and information.
unsigned encode(std::vector<unsigned char>& out,
                const unsigned char* in, unsigned w, unsigned h,
                State& state);
unsigned encode(std::vector<unsigned char>& out,
                const std::vector<unsigned char>& in, unsigned w, unsigned h,
                State& state);
#endif /*LODEPNG_COMPILE_ENCODER*/

#ifdef LODEPNG_COMPILE_DISK
/*
Load a file from disk into an std::vector. If the vector is empty, then either
the file doesn't exist or is an empty file.
*/
void load_file(std::vector<unsigned char>& buffer, const std::string& filename);

/*
Save the binary data in an std::vector to a file on disk. The file is overwritten
without warning.
*/
void save_file(const std::vector<unsigned char>& buffer, const std::string& filename);
#endif //LODEPNG_COMPILE_DISK
#endif //LODEPNG_COMPILE_PNG

#ifdef LODEPNG_COMPILE_ZLIB
#ifdef LODEPNG_COMPILE_DECODER
//Zlib-decompress an unsigned char buffer
unsigned decompress(std::vector<unsigned char>& out, const unsigned char* in, size_t insize,
                    const LodePNGDecompressSettings& settings = lodepng_default_decompress_settings);

//Zlib-decompress an std::vector
unsigned decompress(std::vector<unsigned char>& out, const std::vector<unsigned char>& in,
                    const LodePNGDecompressSettings& settings = lodepng_default_decompress_settings);
#endif //LODEPNG_COMPILE_DECODER

#ifdef LODEPNG_COMPILE_ENCODER
//Zlib-compress an unsigned char buffer
unsigned compress(std::vector<unsigned char>& out, const unsigned char* in, size_t insize,
                  const LodePNGCompressSettings& settings = lodepng_default_compress_settings);

//Zlib-compress an std::vector
unsigned compress(std::vector<unsigned char>& out, const std::vector<unsigned char>& in,
                  const LodePNGCompressSettings& settings = lodepng_default_compress_settings);
#endif //LODEPNG_COMPILE_ENCODER
#endif //LODEPNG_COMPILE_ZLIB
} //namespace lodepng
#endif /*LODEPNG_COMPILE_CPP*/


// RAMA
// ---- helper functions
extern "C" void save_png ( char* fname, unsigned char* img, int w, int h );


/*
TODO:
[.] test if there are no memory leaks or security exploits - done a lot but needs to be checked often
[.] check compatibility with vareous compilers  - done but needs to be redone for every newer version
[X] converting color to 16-bit per channel types
[ ] read all public PNG chunk types (but never let the color profile and gamma ones touch RGB values)
[ ] make sure encoder generates no chunks with size > (2^31)-1
[ ] partial decoding (stream processing)
[X] let the "isFullyOpaque" function check color keys and transparent palettes too
[X] better name for the variables "codes", "codesD", "codelengthcodes", "clcl" and "lldl"
[ ] don't stop decoding on errors like 69, 57, 58 (make warnings)
[ ] make option to choose if the raw image with non multiple of 8 bits per scanline should have padding bits or not
[ ] let the C++ wrapper catch exceptions coming from the standard library and return LodePNG error codes
*/

#endif /*LODEPNG_H inclusion guard*/

/*
LodePNG Documentation
---------------------

0. table of contents
--------------------

  1. about
   1.1. supported features
   1.2. features not supported
  2. C and C++ version
  3. security
  4. decoding
  5. encoding
  6. color conversions
    6.1. PNG color types
    6.2. color conversions
    6.3. padding bits
    6.4. A note about 16-bits per channel and endianness
  7. error values
  8. chunks and PNG editing
  9. compiler support
  10. examples
   10.1. decoder C++ example
   10.2. decoder C example
  11. changes
  12. contact information


1. about
--------

PNG is a file format to store raster images losslessly with good compression,
supporting different color types and alpha channel.

LodePNG is a PNG codec according to the Portable Network Graphics (PNG)
Specification (Second Edition) - W3C Recommendation 10 November 2003.

The specifications used are:

*) Portable Network Graphics (PNG) Specification (Second Edition):
     http://www.w3.org/TR/2003/REC-PNG-20031110
*) RFC 1950 ZLIB Compressed Data Format version 3.3:
     http://www.gzip.org/zlib/rfc-zlib.html
*) RFC 1951 DEFLATE Compressed Data Format Specification ver 1.3:
     http://www.gzip.org/zlib/rfc-deflate.html

The most recent version of LodePNG can currently be found at
http://lodev.org/lodepng/

LodePNG works both in C (ISO C90) and C++, with a C++ wrapper that adds
extra functionality.

LodePNG exists out of two files:
-lodepng.h: the header file for both C and C++
-lodepng.c(pp): give it the name lodepng.c or lodepng.cpp (or .cc) depending on your usage

If you want to start using LodePNG right away without reading this doc, get the
examples from the LodePNG website to see how to use it in code, or check the
smaller examples in chapter 13 here.

LodePNG is simple but only supports the basic requirements. To achieve
simplicity, the following design choices were made: There are no dependencies
on any external library. There are functions to decode and encode a PNG with
a single function call, and extended versions of these functions taking a
LodePNGState struct allowing to specify or get more information. By default
the colors of the raw image are always RGB or RGBA, no matter what color type
the PNG file uses. To read and write files, there are simple functions to
convert the files to/from buffers in memory.

This all makes LodePNG suitable for loading textures in games, demos and small
programs, ... It's less suitable for full fledged image editors, loading PNGs
over network (it requires all the image data to be available before decoding can
begin), life-critical systems, ...

1.1. supported features
-----------------------

The following features are supported by the decoder:

*) decoding of PNGs with any color type, bit depth and interlace mode, to a 24- or 32-bit color raw image,
   or the same color type as the PNG
*) encoding of PNGs, from any raw image to 24- or 32-bit color, or the same color type as the raw image
*) Adam7 interlace and deinterlace for any color type
*) loading the image from harddisk or decoding it from a buffer from other sources than harddisk
*) support for alpha channels, including RGBA color model, translucent palettes and color keying
*) zlib decompression (inflate)
*) zlib compression (deflate)
*) CRC32 and ADLER32 checksums
*) handling of unknown chunks, allowing making a PNG editor that stores custom and unknown chunks.
*) the following chunks are supported (generated/interpreted) by both encoder and decoder:
    IHDR: header information
    PLTE: color palette
    IDAT: pixel data
    IEND: the final chunk
    tRNS: transparency for palettized images
    tEXt: textual information
    zTXt: compressed textual information
    iTXt: international textual information
    bKGD: suggested background color
    pHYs: physical dimensions
    tIME: modification time

1.2. features not supported
---------------------------

The following features are _not_ supported:

*) some features needed to make a conformant PNG-Editor might be still missing.
*) partial loading/stream processing. All data must be available and is processed in one call.
*) The following public chunks are not supported but treated as unknown chunks by LodePNG
    cHRM, gAMA, iCCP, sRGB, sBIT, hIST, sPLT
   Some of these are not supported on purpose: LodePNG wants to provide the RGB values
   stored in the pixels, not values modified by system dependent gamma or color models.


2. C and C++ version
--------------------

The C version uses buffers allocated with alloc that you need to free()
yourself. You need to use init and cleanup functions for each struct whenever
using a struct from the C version to avoid exploits and memory leaks.

The C++ version has extra functions with std::vectors in the interface and the
lodepng::State class which is a LodePNGState with constructor and destructor.

These files work without modification for both C and C++ compilers because all
the additional C++ code is in "#ifdef __cplusplus" blocks that make C-compilers
ignore it, and the C code is made to compile both with strict ISO C90 and C++.

To use the C++ version, you need to rename the source file to lodepng.cpp
(instead of lodepng.c), and compile it with a C++ compiler.

To use the C version, you need to rename the source file to lodepng.c (instead
of lodepng.cpp), and compile it with a C compiler.


3. Security
-----------

Even if carefully designed, it's always possible that LodePNG contains possible
exploits. If you discover one, please let me know, and it will be fixed.

When using LodePNG, care has to be taken with the C version of LodePNG, as well
as the C-style structs when working with C++. The following conventions are used
for all C-style structs:

-if a struct has a corresponding init function, always call the init function when making a new one
-if a struct has a corresponding cleanup function, call it before the struct disappears to avoid memory leaks
-if a struct has a corresponding copy function, use the copy function instead of "=".
 The destination must also be inited already.


4. Decoding
-----------

Decoding converts a PNG compressed image to a raw pixel buffer.

Most documentation on using the decoder is at its declarations in the header
above. For C, simple decoding can be done with functions such as
lodepng_decode32, and more advanced decoding can be done with the struct
LodePNGState and lodepng_decode. For C++, all decoding can be done with the
various lodepng::decode functions, and lodepng::State can be used for advanced
features.

When using the LodePNGState, it uses the following fields for decoding:
*) LodePNGInfo info_png: it stores extra information about the PNG (the input) in here
*) LodePNGColorMode info_raw: here you can say what color mode of the raw image (the output) you want to get
*) LodePNGDecoderSettings decoder: you can specify a few extra settings for the decoder to use

LodePNGInfo info_png
--------------------

After decoding, this contains extra information of the PNG image, except the actual
pixels, width and height because these are already gotten directly from the decoder
functions.

It contains for example the original color type of the PNG image, text comments,
suggested background color, etc... More details about the LodePNGInfo struct are
at its declaration documentation.

LodePNGColorMode info_raw
-------------------------

When decoding, here you can specify which color type you want
the resulting raw image to be. If this is different from the colortype of the
PNG, then the decoder will automatically convert the result. This conversion
always works, except if you want it to convert a color PNG to greyscale or to
a palette with missing colors.

By default, 32-bit color is used for the result.

LodePNGDecoderSettings decoder
------------------------------

The settings can be used to ignore the errors created by invalid CRC and Adler32
chunks, and to disable the decoding of tEXt chunks.

There's also a setting color_convert, true by default. If false, no conversion
is done, the resulting data will be as it was in the PNG (after decompression)
and you'll have to puzzle the colors of the pixels together yourself using the
color type information in the LodePNGInfo.


5. Encoding
-----------

Encoding converts a raw pixel buffer to a PNG compressed image.

Most documentation on using the encoder is at its declarations in the header
above. For C, simple encoding can be done with functions such as
lodepng_encode32, and more advanced decoding can be done with the struct
LodePNGState and lodepng_encode. For C++, all encoding can be done with the
various lodepng::encode functions, and lodepng::State can be used for advanced
features.

Like the decoder, the encoder can also give errors. However it gives less errors
since the encoder input is trusted, the decoder input (a PNG image that could
be forged by anyone) is not trusted.

When using the LodePNGState, it uses the following fields for encoding:
*) LodePNGInfo info_png: here you specify how you want the PNG (the output) to be.
*) LodePNGColorMode info_raw: here you say what color type of the raw image (the input) has
*) LodePNGEncoderSettings encoder: you can specify a few settings for the encoder to use

LodePNGInfo info_png
--------------------

When encoding, you use this the opposite way as when decoding: for encoding,
you fill in the values you want the PNG to have before encoding. By default it's
not needed to specify a color type for the PNG since it's automatically chosen,
but it's possible to choose it yourself given the right settings.

The encoder will not always exactly match the LodePNGInfo struct you give,
it tries as close as possible. Some things are ignored by the encoder. The
encoder uses, for example, the following settings from it when applicable:
colortype and bitdepth, text chunks, time chunk, the color key, the palette, the
background color, the interlace method, unknown chunks, ...

When encoding to a PNG with colortype 3, the encoder will generate a PLTE chunk.
If the palette contains any colors for which the alpha channel is not 255 (so
there are translucent colors in the palette), it'll add a tRNS chunk.

LodePNGColorMode info_raw
-------------------------

You specify the color type of the raw image that you give to the input here,
including a possible transparent color key and palette you happen to be using in
your raw image data.

By default, 32-bit color is assumed, meaning your input has to be in RGBA
format with 4 bytes (unsigned chars) per pixel.

LodePNGEncoderSettings encoder
------------------------------

The following settings are supported (some are in sub-structs):
*) auto_convert: when this option is enabled, the encoder will
automatically choose the smallest possible color mode (including color key) that
can encode the colors of all pixels without information loss.
*) btype: the block type for LZ77. 0 = uncompressed, 1 = fixed huffman tree,
   2 = dynamic huffman tree (best compression). Should be 2 for proper
   compression.
*) use_lz77: whether or not to use LZ77 for compressed block types. Should be
   true for proper compression.
*) windowsize: the window size used by the LZ77 encoder (1 - 32768). Has value
   2048 by default, but can be set to 32768 for better, but slow, compression.
*) force_palette: if colortype is 2 or 6, you can make the encoder write a PLTE
   chunk if force_palette is true. This can used as suggested palette to convert
   to by viewers that don't support more than 256 colors (if those still exist)
*) add_id: add text chunk "Encoder: LodePNG <version>" to the image.
*) text_compression: default 1. If 1, it'll store texts as zTXt instead of tEXt chunks.
  zTXt chunks use zlib compression on the text. This gives a smaller result on
  large texts but a larger result on small texts (such as a single program name).
  It's all tEXt or all zTXt though, there's no separate setting per text yet.


6. color conversions
--------------------

An important thing to note about LodePNG, is that the color type of the PNG, and
the color type of the raw image, are completely independent. By default, when
you decode a PNG, you get the result as a raw image in the color type you want,
no matter whether the PNG was encoded with a palette, greyscale or RGBA color.
And if you encode an image, by default LodePNG will automatically choose the PNG
color type that gives good compression based on the values of colors and amount
of colors in the image. It can be configured to let you control it instead as
well, though.

To be able to do this, LodePNG does conversions from one color mode to another.
It can convert from almost any color type to any other color type, except the
following conversions: RGB to greyscale is not supported, and converting to a
palette when the palette doesn't have a required color is not supported. This is
not supported on purpose: this is information loss which requires a color
reduction algorithm that is beyong the scope of a PNG encoder (yes, RGB to grey
is easy, but there are multiple ways if you want to give some channels more
weight).

By default, when decoding, you get the raw image in 32-bit RGBA or 24-bit RGB
color, no matter what color type the PNG has. And by default when encoding,
LodePNG automatically picks the best color model for the output PNG, and expects
the input image to be 32-bit RGBA or 24-bit RGB. So, unless you want to control
the color format of the images yourself, you can skip this chapter.

6.1. PNG color types
--------------------

A PNG image can have many color types, ranging from 1-bit color to 64-bit color,
as well as palettized color modes. After the zlib decompression and unfiltering
in the PNG image is done, the raw pixel data will have that color type and thus
a certain amount of bits per pixel. If you want the output raw image after
decoding to have another color type, a conversion is done by LodePNG.

The PNG specification gives the following color types:

0: greyscale, bit depths 1, 2, 4, 8, 16
2: RGB, bit depths 8 and 16
3: palette, bit depths 1, 2, 4 and 8
4: greyscale with alpha, bit depths 8 and 16
6: RGBA, bit depths 8 and 16

Bit depth is the amount of bits per pixel per color channel. So the total amount
of bits per pixel is: amount of channels * bitdepth.

6.2. color conversions
----------------------

As explained in the sections about the encoder and decoder, you can specify
color types and bit depths in info_png and info_raw to change the default
behaviour.

If, when decoding, you want the raw image to be something else than the default,
you need to set the color type and bit depth you want in the LodePNGColorMode,
or the parameters of the simple function of LodePNG you're using.

If, when encoding, you use another color type than the default in the input
image, you need to specify its color type and bit depth in the LodePNGColorMode
of the raw image, or use the parameters of the simplefunction of LodePNG you're
using.

If, when encoding, you don't want LodePNG to choose the output PNG color type
but control it yourself, you need to set auto_convert in the encoder settings
to LAC_NONE, and specify the color type you want in the LodePNGInfo of the
encoder.

If you do any of the above, LodePNG may need to do a color conversion, which
follows the rules below, and may sometimes not be allowed.

To avoid some confusion:
-the decoder converts from PNG to raw image
-the encoder converts from raw image to PNG
-the colortype and bitdepth in LodePNGColorMode info_raw, are those of the raw image
-the colortype and bitdepth in the color field of LodePNGInfo info_png, are those of the PNG
-when encoding, the color type in LodePNGInfo is ignored if auto_convert
 is enabled, it is automatically generated instead
-when decoding, the color type in LodePNGInfo is set by the decoder to that of the original
 PNG image, but it can be ignored since the raw image has the color type you requested instead
-if the color type of the LodePNGColorMode and PNG image aren't the same, a conversion
 between the color types is done if the color types are supported. If it is not
 supported, an error is returned. If the types are the same, no conversion is done.
-even though some conversions aren't supported, LodePNG supports loading PNGs from any
 colortype and saving PNGs to any colortype, sometimes it just requires preparing
 the raw image correctly before encoding.
-both encoder and decoder use the same color converter.

Non supported color conversions:
-color to greyscale: no error is thrown, but the result will look ugly because
only the red channel is taken
-anything, to palette when that palette does not have that color in it: in this
case an error is thrown

Supported color conversions:
-anything to 8-bit RGB, 8-bit RGBA, 16-bit RGB, 16-bit RGBA
-any grey or grey+alpha, to grey or grey+alpha
-anything to a palette, as long as the palette has the requested colors in it
-removing alpha channel
-higher to smaller bitdepth, and vice versa

If you want no color conversion to be done:
-In the encoder, you can make it save a PNG with any color type by giving the
raw color mode and LodePNGInfo the same color mode, and setting auto_convert to
LAC_NO.
-In the decoder, you can make it store the pixel data in the same color type
as the PNG has, by setting the color_convert setting to false. Settings in
info_raw are then ignored.

The function lodepng_convert does the color conversion. It is available in the
interface but normally isn't needed since the encoder and decoder already call
it.

6.3. padding bits
-----------------

In the PNG file format, if a less than 8-bit per pixel color type is used and the scanlines
have a bit amount that isn't a multiple of 8, then padding bits are used so that each
scanline starts at a fresh byte. But that is NOT true for the LodePNG raw input and output.
The raw input image you give to the encoder, and the raw output image you get from the decoder
will NOT have these padding bits, e.g. in the case of a 1-bit image with a width
of 7 pixels, the first pixel of the second scanline will the the 8th bit of the first byte,
not the first bit of a new byte.

6.4. A note about 16-bits per channel and endianness
----------------------------------------------------

LodePNG uses unsigned char arrays for 16-bit per channel colors too, just like
for any other color format. The 16-bit values are stored in big endian (most
significant byte first) in these arrays. This is the opposite order of the
little endian used by x86 CPU's.

LodePNG always uses big endian because the PNG file format does so internally.
Conversions to other formats than PNG uses internally are not supported by
LodePNG on purpose, there are myriads of formats, including endianness of 16-bit
colors, the order in which you store R, G, B and A, and so on. Supporting and
converting to/from all that is outside the scope of LodePNG.

This may mean that, depending on your use case, you may want to convert the big
endian output of LodePNG to little endian with a for loop. This is certainly not
always needed, many applications and libraries support big endian 16-bit colors
anyway, but it means you cannot simply cast the unsigned char* buffer to an
unsigned short* buffer on x86 CPUs.


7. error values
---------------

All functions in LodePNG that return an error code, return 0 if everything went
OK, or a non-zero code if there was an error.

The meaning of the LodePNG error values can be retrieved with the function
lodepng_error_text: given the numerical error code, it returns a description
of the error in English as a string.

Check the implementation of lodepng_error_text to see the meaning of each code.


8. chunks and PNG editing
-------------------------

If you want to add extra chunks to a PNG you encode, or use LodePNG for a PNG
editor that should follow the rules about handling of unknown chunks, or if your
program is able to read other types of chunks than the ones handled by LodePNG,
then that's possible with the chunk functions of LodePNG.

A PNG chunk has the following layout:

4 bytes length
4 bytes type name
length bytes data
4 bytes CRC

8.1. iterating through chunks
-----------------------------

If you have a buffer containing the PNG image data, then the first chunk (the
IHDR chunk) starts at byte number 8 of that buffer. The first 8 bytes are the
signature of the PNG and are not part of a chunk. But if you start at byte 8
then you have a chunk, and can check the following things of it.

NOTE: none of these functions check for memory buffer boundaries. To avoid
exploits, always make sure the buffer contains all the data of the chunks.
When using lodepng_chunk_next, make sure the returned value is within the
allocated memory.

unsigned lodepng_chunk_length(const unsigned char* chunk):

Get the length of the chunk's data. The total chunk length is this length + 12.

void lodepng_chunk_type(char type[5], const unsigned char* chunk):
unsigned char lodepng_chunk_type_equals(const unsigned char* chunk, const char* type):

Get the type of the chunk or compare if it's a certain type

unsigned char lodepng_chunk_critical(const unsigned char* chunk):
unsigned char lodepng_chunk_private(const unsigned char* chunk):
unsigned char lodepng_chunk_safetocopy(const unsigned char* chunk):

Check if the chunk is critical in the PNG standard (only IHDR, PLTE, IDAT and IEND are).
Check if the chunk is private (public chunks are part of the standard, private ones not).
Check if the chunk is safe to copy. If it's not, then, when modifying data in a critical
chunk, unsafe to copy chunks of the old image may NOT be saved in the new one if your
program doesn't handle that type of unknown chunk.

unsigned char* lodepng_chunk_data(unsigned char* chunk):
const unsigned char* lodepng_chunk_data_const(const unsigned char* chunk):

Get a pointer to the start of the data of the chunk.

unsigned lodepng_chunk_check_crc(const unsigned char* chunk):
void lodepng_chunk_generate_crc(unsigned char* chunk):

Check if the crc is correct or generate a correct one.

unsigned char* lodepng_chunk_next(unsigned char* chunk):
const unsigned char* lodepng_chunk_next_const(const unsigned char* chunk):

Iterate to the next chunk. This works if you have a buffer with consecutive chunks. Note that these
functions do no boundary checking of the allocated data whatsoever, so make sure there is enough
data available in the buffer to be able to go to the next chunk.

unsigned lodepng_chunk_append(unsigned char** out, size_t* outlength, const unsigned char* chunk):
unsigned lodepng_chunk_create(unsigned char** out, size_t* outlength, unsigned length,
                              const char* type, const unsigned char* data):

These functions are used to create new chunks that are appended to the data in *out that has
length *outlength. The append function appends an existing chunk to the new data. The create
function creates a new chunk with the given parameters and appends it. Type is the 4-letter
name of the chunk.

8.2. chunks in info_png
-----------------------

The LodePNGInfo struct contains fields with the unknown chunk in it. It has 3
buffers (each with size) to contain 3 types of unknown chunks:
the ones that come before the PLTE chunk, the ones that come between the PLTE
and the IDAT chunks, and the ones that come after the IDAT chunks.
It's necessary to make the distionction between these 3 cases because the PNG
standard forces to keep the ordering of unknown chunks compared to the critical
chunks, but does not force any other ordering rules.

info_png.unknown_chunks_data[0] is the chunks before PLTE
info_png.unknown_chunks_data[1] is the chunks after PLTE, before IDAT
info_png.unknown_chunks_data[2] is the chunks after IDAT

The chunks in these 3 buffers can be iterated through and read by using the same
way described in the previous subchapter.

When using the decoder to decode a PNG, you can make it store all unknown chunks
if you set the option settings.remember_unknown_chunks to 1. By default, this
option is off (0).

The encoder will always encode unknown chunks that are stored in the info_png.
If you need it to add a particular chunk that isn't known by LodePNG, you can
use lodepng_chunk_append or lodepng_chunk_create to the chunk data in
info_png.unknown_chunks_data[x].

Chunks that are known by LodePNG should not be added in that way. E.g. to make
LodePNG add a bKGD chunk, set background_defined to true and add the correct
parameters there instead.


9. compiler support
-------------------

No libraries other than the current standard C library are needed to compile
LodePNG. For the C++ version, only the standard C++ library is needed on top.
Add the files lodepng.c(pp) and lodepng.h to your project, include
lodepng.h where needed, and your program can read/write PNG files.

If performance is important, use optimization when compiling! For both the
encoder and decoder, this makes a large difference.

Make sure that LodePNG is compiled with the same compiler of the same version
and with the same settings as the rest of the program, or the interfaces with
std::vectors and std::strings in C++ can be incompatible.

CHAR_BITS must be 8 or higher, because LodePNG uses unsigned chars for octets.

*) gcc and g++

LodePNG is developed in gcc so this compiler is natively supported. It gives no
warnings with compiler options "-Wall -Wextra -pedantic -ansi", with gcc and g++
version 4.7.1 on Linux, 32-bit and 64-bit.

*) Mingw

The Mingw compiler (a port of gcc) for Windows is fully supported by LodePNG.

*) Visual Studio 2005 and up, Visual C++ Express Edition 2005 and up

Visual Studio may give warnings about 'fopen' being deprecated. A multiplatform library
can't support the proposed Visual Studio alternative however, so LodePNG keeps using
fopen. If you don't want to see the deprecated warnings, put this on top of lodepng.h
before the inclusions:
#define _CRT_SECURE_NO_DEPRECATE

With warning level 4 (W4), there may be a lot of warnings about possible loss of data
due to integer conversions. I'm not planning to resolve these warnings. The gcc compiler
doesn't give those even with strict warning flags. With warning level 3 in VS 2008
Express Edition, LodePNG is, other than the fopen warnings, warning-free again since
version 20120923.

Visual Studio may want "stdafx.h" files to be included in each source file. That
is not standard C++ and will not be added to the stock LodePNG. Try to find a
setting to disable it for this source file.

*) Visual Studio 6.0

LodePNG support for Visual Studio 6.0 is not guaranteed because VS6 doesn't
follow the C++ standard correctly.

*) Comeau C/C++

Vesion 20070107 compiles without problems on the Comeau C/C++ Online Test Drive
at http://www.comeaucomputing.com/tryitout in both C90 and C++ mode.

*) Compilers on Macintosh

LodePNG has been reported to work both with the gcc and LLVM for Macintosh, both
for C and C++.

*) Other Compilers

If you encounter problems on other compilers, feel free to let me know and I may
try to fix it if the compiler is modern standards complient.


10. examples
------------

This decoder example shows the most basic usage of LodePNG. More complex
examples can be found on the LodePNG website.

10.1. decoder C++ example
-------------------------

#include "lodepng.h"
#include <iostream>

int main(int argc, char *argv[])
{
  const char* filename = argc > 1 ? argv[1] : "test.png";

  //load and decode
  std::vector<unsigned char> image;
  unsigned width, height;
  unsigned error = lodepng::decode(image, width, height, filename);

  //if there's an error, display it
  if(error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;

  //the pixels are now in the vector "image", 4 bytes per pixel, ordered RGBARGBA..., use it as texture, draw it, ...
}

10.2. decoder C example
-----------------------

#include "lodepng.h"

int main(int argc, char *argv[])
{
  unsigned error;
  unsigned char* image;
  size_t width, height;
  const char* filename = argc > 1 ? argv[1] : "test.png";

  error = lodepng_decode32_file(&image, &width, &height, filename);

  if(error) printf("decoder error %u: %s\n", error, lodepng_error_text(error));

  / * use image here * /

  free(image);
  return 0;
}


11. changes
-----------

The version number of LodePNG is the date of the change given in the format
yyyymmdd.

Some changes aren't backwards compatible. Those are indicated with a (!)
symbol.

*) 28 jan 2013: Bugfix with color key.
*) 27 okt 2012: Tweaks in text chunk keyword length error handling.
*) 8 okt 2012 (!): Added new filter strategy (entropy) and new auto color mode.
    (no palette). Better deflate tree encoding. New compression tweak settings.
    Faster color conversions while decoding. Some internal cleanups.
*) 23 sep 2012: Reduced warnings in Visual Studio a little bit.
*) 1 sep 2012 (!): Removed #define's for giving custom (de)compression functions
    and made it work with function pointers instead.
*) 23 jun 2012: Added more filter strategies. Made it easier to use custom alloc
    and free functions and toggle #defines from compiler flags. Small fixes.
*) 6 may 2012 (!): Made plugging in custom zlib/deflate functions more flexible.
*) 22 apr 2012 (!): Made interface more consistent, renaming a lot. Removed
    redundant C++ codec classes. Reduced amount of structs. Everything changed,
    but it is cleaner now imho and functionality remains the same. Also fixed
    several bugs and shrinked the implementation code. Made new samples.
*) 6 nov 2011 (!): By default, the encoder now automatically chooses the best
    PNG color model and bit depth, based on the amount and type of colors of the
    raw image. For this, autoLeaveOutAlphaChannel replaced by auto_choose_color.
*) 9 okt 2011: simpler hash chain implementation for the encoder.
*) 8 sep 2011: lz77 encoder lazy matching instead of greedy matching.
*) 23 aug 2011: tweaked the zlib compression parameters after benchmarking.
    A bug with the PNG filtertype heuristic was fixed, so that it chooses much
    better ones (it's quite significant). A setting to do an experimental, slow,
    brute force search for PNG filter types is added.
*) 17 aug 2011 (!): changed some C zlib related function names.
*) 16 aug 2011: made the code less wide (max 120 characters per line).
*) 17 apr 2011: code cleanup. Bugfixes. Convert low to 16-bit per sample colors.
*) 21 feb 2011: fixed compiling for C90. Fixed compiling with sections disabled.
*) 11 dec 2010: encoding is made faster, based on suggestion by Peter Eastman
    to optimize long sequences of zeros.
*) 13 nov 2010: added LodePNG_InfoColor_hasPaletteAlpha and
    LodePNG_InfoColor_canHaveAlpha functions for convenience.
*) 7 nov 2010: added LodePNG_error_text function to get error code description.
*) 30 okt 2010: made decoding slightly faster
*) 26 okt 2010: (!) changed some C function and struct names (more consistent).
     Reorganized the documentation and the declaration order in the header.
*) 08 aug 2010: only changed some comments and external samples.
*) 05 jul 2010: fixed bug thanks to warnings in the new gcc version.
*) 14 mar 2010: fixed bug where too much memory was allocated for char buffers.
*) 02 sep 2008: fixed bug where it could create empty tree that linux apps could
    read by ignoring the problem but windows apps couldn't.
*) 06 jun 2008: added more error checks for out of memory cases.
*) 26 apr 2008: added a few more checks here and there to ensure more safety.
*) 06 mar 2008: crash with encoding of strings fixed
*) 02 feb 2008: support for international text chunks added (iTXt)
*) 23 jan 2008: small cleanups, and #defines to divide code in sections
*) 20 jan 2008: support for unknown chunks allowing using LodePNG for an editor.
*) 18 jan 2008: support for tIME and pHYs chunks added to encoder and decoder.
*) 17 jan 2008: ability to encode and decode compressed zTXt chunks added
    Also vareous fixes, such as in the deflate and the padding bits code.
*) 13 jan 2008: Added ability to encode Adam7-interlaced images. Improved
    filtering code of encoder.
*) 07 jan 2008: (!) changed LodePNG to use ISO C90 instead of C++. A
    C++ wrapper around this provides an interface almost identical to before.
    Having LodePNG be pure ISO C90 makes it more portable. The C and C++ code
    are together in these files but it works both for C and C++ compilers.
*) 29 dec 2007: (!) changed most integer types to unsigned int + other tweaks
*) 30 aug 2007: bug fixed which makes this Borland C++ compatible
*) 09 aug 2007: some VS2005 warnings removed again
*) 21 jul 2007: deflate code placed in new namespace separate from zlib code
*) 08 jun 2007: fixed bug with 2- and 4-bit color, and small interlaced images
*) 04 jun 2007: improved support for Visual Studio 2005: crash with accessing
    invalid std::vector element [0] fixed, and level 3 and 4 warnings removed
*) 02 jun 2007: made the encoder add a tag with version by default
*) 27 may 2007: zlib and png code separated (but still in the same file),
    simple encoder/decoder functions added for more simple usage cases
*) 19 may 2007: minor fixes, some code cleaning, new error added (error 69),
    moved some examples from here to lodepng_examples.cpp
*) 12 may 2007: palette decoding bug fixed
*) 24 apr 2007: changed the license from BSD to the zlib license
*) 11 mar 2007: very simple addition: ability to encode bKGD chunks.
*) 04 mar 2007: (!) tEXt chunk related fixes, and support for encoding
    palettized PNG images. Plus little interface change with palette and texts.
*) 03 mar 2007: Made it encode dynamic Huffman shorter with repeat codes.
    Fixed a bug where the end code of a block had length 0 in the Huffman tree.
*) 26 feb 2007: Huffman compression with dynamic trees (BTYPE 2) now implemented
    and supported by the encoder, resulting in smaller PNGs at the output.
*) 27 jan 2007: Made the Adler-32 test faster so that a timewaste is gone.
*) 24 jan 2007: gave encoder an error interface. Added color conversion from any
    greyscale type to 8-bit greyscale with or without alpha.
*) 21 jan 2007: (!) Totally changed the interface. It allows more color types
    to convert to and is more uniform. See the manual for how it works now.
*) 07 jan 2007: Some cleanup & fixes, and a few changes over the last days:
    encode/decode custom tEXt chunks, separate classes for zlib & deflate, and
    at last made the decoder give errors for incorrect Adler32 or Crc.
*) 01 jan 2007: Fixed bug with encoding PNGs with less than 8 bits per channel.
*) 29 dec 2006: Added support for encoding images without alpha channel, and
    cleaned out code as well as making certain parts faster.
*) 28 dec 2006: Added "Settings" to the encoder.
*) 26 dec 2006: The encoder now does LZ77 encoding and produces much smaller files now.
    Removed some code duplication in the decoder. Fixed little bug in an example.
*) 09 dec 2006: (!) Placed output parameters of public functions as first parameter.
    Fixed a bug of the decoder with 16-bit per color.
*) 15 okt 2006: Changed documentation structure
*) 09 okt 2006: Encoder class added. It encodes a valid PNG image from the
    given image buffer, however for now it's not compressed.
*) 08 sep 2006: (!) Changed to interface with a Decoder class
*) 30 jul 2006: (!) LodePNG_InfoPng , width and height are now retrieved in different
    way. Renamed decodePNG to decodePNGGeneric.
*) 29 jul 2006: (!) Changed the interface: image info is now returned as a
    struct of type LodePNG::LodePNG_Info, instead of a vector, which was a bit clumsy.
*) 28 jul 2006: Cleaned the code and added new error checks.
    Corrected terminology "deflate" into "inflate".
*) 23 jun 2006: Added SDL example in the documentation in the header, this
    example allows easy debugging by displaying the PNG and its transparency.
*) 22 jun 2006: (!) Changed way to obtain error value. Added
    loadFile function for convenience. Made decodePNG32 faster.
*) 21 jun 2006: (!) Changed type of info vector to unsigned.
    Changed position of palette in info vector. Fixed an important bug that
    happened on PNGs with an uncompressed block.
*) 16 jun 2006: Internally changed unsigned into unsigned where
    needed, and performed some optimizations.
*) 07 jun 2006: (!) Renamed functions to decodePNG and placed them
    in LodePNG namespace. Changed the order of the parameters. Rewrote the
    documentation in the header. Renamed files to lodepng.cpp and lodepng.h
*) 22 apr 2006: Optimized and improved some code
*) 07 sep 2005: (!) Changed to std::vector interface
*) 12 aug 2005: Initial release (C++, decoder only)


12. contact information
-----------------------

Feel free to contact me with suggestions, problems, comments, ... concerning
LodePNG. If you encounter a PNG image that doesn't work properly with this
decoder, feel free to send it and I'll use it to find and fix the problem.

My email address is (puzzle the account and domain together with an @ symbol):
Domain: gmail dot com.
Account: lode dot vandevenne.


Copyright (c) 2005-2012 Lode Vandevenne
*/


//------------------------------------------------------ TGA FORMAT

class TGA
{
public:
    enum TGAFormat
    {
        RGB = 0x1907,
        RGBA = 0x1908,
        ALPHA = 0x1906,
        UNKNOWN = -1
    };

    enum TGAError
    {
        TGA_NO_ERROR = 1,   // No error
        TGA_FILE_NOT_FOUND, // File was not found 
        TGA_BAD_IMAGE_TYPE, // Color mapped image or image is not uncompressed
        TGA_BAD_DIMENSION,  // Dimension is not a power of 2 
        TGA_BAD_BITS,       // Image bits is not 8, 24 or 32 
        TGA_BAD_DATA        // Image data could not be loaded 
	};

    TGA(void) : 
        m_texFormat(TGA::UNKNOWN),
        m_nImageWidth(0),
        m_nImageHeight(0),
        m_nImageBits(0),
        m_nImageData(NULL) {}

    ~TGA(void);

    TGA::TGAError load( const char *name );
    TGA::TGAError saveFromExternalData( const char *name, int w, int h, TGAFormat fmt, const unsigned char *externalImage );

    TGAFormat       m_texFormat;
    int             m_nImageWidth;
    int             m_nImageHeight;
    int             m_nImageBits;
    unsigned char * m_nImageData;
    
private:

    int returnError(FILE *s, int error);
    unsigned char *getRGBA(FILE *s, int size);
    unsigned char *getRGB(FILE *s, int size);
    unsigned char *getGray(FILE *s, int size);
    void           writeRGBA(FILE *s, const unsigned char *externalImage, int size);
    void           writeRGB(FILE *s, const unsigned char *externalImage, int size);
    void           writeGrayAsRGB(FILE *s, const unsigned char *externalImage, int size);
    void           writeGray(FILE *s, const unsigned char *externalImage, int size);
};

// Vector/Matrix Math
// R. Hoetzlein (c) 2005-2011
// Z-Lib License
// 
// This software is provided 'as-is', without any express or implied
// warranty.  In no event will the authors be held liable for any damages
// arising from the use of this software.
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software
// 2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.
//

#ifndef VECTOR_DEF
	#define VECTOR_DEF

	#include <stdlib.h>
	//#define VECTOR_INITIALIZE				// Initializes vectors	
																
	class Vector2DC;						// Forward Referencing
	class Vector2DI;
	class Vector2DF;
	class Vector3DC;
	class Vector3DI;
	class Vector3DF;
	class Vector4DC;
	class Vector4DF;	
	class MatrixF;
	class Matrix4F;

	// Vector2DC Declaration
	
	#define VNAME		2DC
	#define VTYPE		unsigned char
	
	#define LUNA_CORE

	class Vector2DC {
	public:
		VTYPE x, y;

		// Constructors/Destructors
		Vector2DC();
		Vector2DC (VTYPE xa, VTYPE ya);
		Vector2DC (Vector2DC &op);	
		Vector2DC (Vector2DI &op);	
		Vector2DC (Vector2DF &op);	
		Vector2DC (Vector3DC &op);	
		Vector2DC (Vector3DI &op);	
		Vector2DC (Vector3DF &op);	
		Vector2DC (Vector4DF &op);

		Vector2DC &Set (VTYPE xa, VTYPE ya)	{x=xa; y=ya; return *this;}

		// Member Functions
		Vector2DC &operator= (Vector2DC &op);
		Vector2DC &operator= (Vector2DI &op);
		Vector2DC &operator= (Vector2DF &op);
		Vector2DC &operator= (Vector3DC &op);
		Vector2DC &operator= (Vector3DI &op);
		Vector2DC &operator= (Vector3DF &op);
		Vector2DC &operator= (Vector4DF &op);
		
		Vector2DC &operator+= (Vector2DC &op);
		Vector2DC &operator+= (Vector2DI &op);
		Vector2DC &operator+= (Vector2DF &op);
		Vector2DC &operator+= (Vector3DC &op);
		Vector2DC &operator+= (Vector3DI &op);
		Vector2DC &operator+= (Vector3DF &op);
		Vector2DC &operator+= (Vector4DF &op);

		Vector2DC &operator-= (Vector2DC &op);
		Vector2DC &operator-= (Vector2DI &op);
		Vector2DC &operator-= (Vector2DF &op);
		Vector2DC &operator-= (Vector3DC &op);
		Vector2DC &operator-= (Vector3DI &op);
		Vector2DC &operator-= (Vector3DF &op);
		Vector2DC &operator-= (Vector4DF &op);
	
		Vector2DC &operator*= (Vector2DC &op);
		Vector2DC &operator*= (Vector2DI &op);
		Vector2DC &operator*= (Vector2DF &op);
		Vector2DC &operator*= (Vector3DC &op);
		Vector2DC &operator*= (Vector3DI &op);
		Vector2DC &operator*= (Vector3DF &op);
		Vector2DC &operator*= (Vector4DF &op);

		Vector2DC &operator/= (Vector2DC &op);
		Vector2DC &operator/= (Vector2DI &op);
		Vector2DC &operator/= (Vector2DF &op);
		Vector2DC &operator/= (Vector3DC &op);
		Vector2DC &operator/= (Vector3DI &op);
		Vector2DC &operator/= (Vector3DF &op);
		Vector2DC &operator/= (Vector4DF &op);

		// Note: Cross product does not exist for 2D vectors (only 3D)
		
		double Dot(Vector2DC &v);
		double Dot(Vector2DI &v);
		double Dot(Vector2DF &v);

		double Dist (Vector2DC &v);
		double Dist (Vector2DI &v);
		double Dist (Vector2DF &v);
		double Dist (Vector3DC &v);
		double Dist (Vector3DI &v);
		double Dist (Vector3DF &v);
		double Dist (Vector4DF &v);

		double DistSq (Vector2DC &v);		
		double DistSq (Vector2DI &v);		
		double DistSq (Vector2DF &v);		
		double DistSq (Vector3DC &v);		
		double DistSq (Vector3DI &v);		
		double DistSq (Vector3DF &v);		
		double DistSq (Vector4DF &v);

		Vector2DC &Normalize (void);
		double Length (void);

		VTYPE &X(void)				{return x;}
		VTYPE &Y(void)				{return y;}
		VTYPE Z(void)				{return 0;}
		VTYPE W(void)				{return 0;}
		const VTYPE &X(void) const	{return x;}
		const VTYPE &Y(void) const	{return y;}
		const VTYPE Z(void) const	{return 0;}
		const VTYPE W(void) const	{return 0;}
		VTYPE *Data (void)			{return &x;}

	};
	
	#undef VNAME
	#undef VTYPE

	// Vector2DI Declaration

	#define VNAME		2DI
	#define VTYPE		int

	class Vector2DI {
	public:
		VTYPE x, y;

		// Constructors/Destructors
		Vector2DI();							
		Vector2DI (const VTYPE xa, const VTYPE ya);
		Vector2DI (const Vector2DC &op);				
		Vector2DI (const Vector2DI &op);				// *** THESE SHOULD ALL BE const
		Vector2DI (const Vector2DF &op);				
		Vector2DI (const Vector3DC &op);				
		Vector2DI (const Vector3DI &op);				
		Vector2DI (const Vector3DF &op);				
		Vector2DI (const Vector4DF &op);

		// Member Functions
		Vector2DI &operator= (const Vector2DC &op);
		Vector2DI &operator= (const Vector2DI &op);
		Vector2DI &operator= (const Vector2DF &op);
		Vector2DI &operator= (const Vector3DC &op);
		Vector2DI &operator= (const Vector3DI &op);
		Vector2DI &operator= (const Vector3DF &op);
		Vector2DI &operator= (const Vector4DF &op);

		Vector2DI &operator+= (const Vector2DC &op);
		Vector2DI &operator+= (const Vector2DI &op);
		Vector2DI &operator+= (const Vector2DF &op);
		Vector2DI &operator+= (const Vector3DC &op);
		Vector2DI &operator+= (const Vector3DI &op);
		Vector2DI &operator+= (const Vector3DF &op);
		Vector2DI &operator+= (const Vector4DF &op);

		Vector2DI &operator-= (const Vector2DC &op);
		Vector2DI &operator-= (const Vector2DI &op);
		Vector2DI &operator-= (const Vector2DF &op);
		Vector2DI &operator-= (const Vector3DC &op);
		Vector2DI &operator-= (const Vector3DI &op);
		Vector2DI &operator-= (const Vector3DF &op);
		Vector2DI &operator-= (const Vector4DF &op);
	
		Vector2DI &operator*= (const Vector2DC &op);
		Vector2DI &operator*= (const Vector2DI &op);
		Vector2DI &operator*= (const Vector2DF &op);
		Vector2DI &operator*= (const Vector3DC &op);
		Vector2DI &operator*= (const Vector3DI &op);
		Vector2DI &operator*= (const Vector3DF &op);
		Vector2DI &operator*= (const Vector4DF &op);

		Vector2DI &operator/= (const Vector2DC &op);
		Vector2DI &operator/= (const Vector2DI &op);
		Vector2DI &operator/= (const Vector2DF &op);
		Vector2DI &operator/= (const Vector3DC &op);
		Vector2DI &operator/= (const Vector3DI &op);
		Vector2DI &operator/= (const Vector3DF &op);
		Vector2DI &operator/= (const Vector4DF &op);


		// Note: Cross product does not exist for 2D vectors (only 3D)
		
		double Dot (const Vector2DC &v);
		double Dot (const Vector2DI &v);
		double Dot (const Vector2DF &v);

		double Dist (const Vector2DC &v);
		double Dist (const Vector2DI &v);
		double Dist (const Vector2DF &v);
		double Dist (const Vector3DC &v);
		double Dist (const Vector3DI &v);
		double Dist (const Vector3DF &v);
		double Dist (const Vector4DF &v);

		double DistSq (const Vector2DC &v);
		double DistSq (const Vector2DI &v);
		double DistSq (const Vector2DF &v);
		double DistSq (const Vector3DC &v);
		double DistSq (const Vector3DI &v);
		double DistSq (const Vector3DF &v);
		double DistSq (const Vector4DF &v);
		
		Vector2DI &Normalize (void);
		double Length (void);

		VTYPE &X(void)				{return x;}
		VTYPE &Y(void)				{return y;}
		VTYPE Z(void)				{return 0;}
		VTYPE W(void)				{return 0;}
		const VTYPE &X(void) const	{return x;}
		const VTYPE &Y(void) const	{return y;}
		const VTYPE Z(void) const	{return 0;}
		const VTYPE W(void) const	{return 0;}
		VTYPE *Data (void)			{return &x;}
	};
	
	#undef VNAME
	#undef VTYPE

	// Vector2DF Declarations

	#define VNAME		2DF
	#define VTYPE		float

	class Vector2DF {
	public:
		VTYPE x, y;

		// Constructors/Destructors
		 Vector2DF ();
		 Vector2DF (const VTYPE xa, const VTYPE ya);
		 Vector2DF (const Vector2DC &op);
		 Vector2DF (const Vector2DI &op);
		 Vector2DF (const Vector2DF &op);
		 Vector2DF (const Vector3DC &op);
		 Vector2DF (const Vector3DI &op);
		 Vector2DF (const Vector3DF &op);
		 Vector2DF (const Vector4DF &op);

		 Vector2DF &Set (const float xa, const float ya);
		 
		 // Member Functions
		 Vector2DF &operator= (const Vector2DC &op);
		 Vector2DF &operator= (const Vector2DI &op);
		 Vector2DF &operator= (const Vector2DF &op);
		 Vector2DF &operator= (const Vector3DC &op);
		 Vector2DF &operator= (const Vector3DI &op);
		 Vector2DF &operator= (const Vector3DF &op);
		 Vector2DF &operator= (const Vector4DF &op);
		
		 Vector2DF &operator+= (const Vector2DC &op);
		 Vector2DF &operator+= (const Vector2DI &op);
		 Vector2DF &operator+= (const Vector2DF &op);
		 Vector2DF &operator+= (const Vector3DC &op);
		 Vector2DF &operator+= (const Vector3DI &op);
		 Vector2DF &operator+= (const Vector3DF &op);
		 Vector2DF &operator+= (const Vector4DF &op);

		 Vector2DF &operator-= (const Vector2DC &op);
		 Vector2DF &operator-= (const Vector2DI &op);
		 Vector2DF &operator-= (const Vector2DF &op);
		 Vector2DF &operator-= (const Vector3DC &op);
		 Vector2DF &operator-= (const Vector3DI &op);
		 Vector2DF &operator-= (const Vector3DF &op);
		 Vector2DF &operator-= (const Vector4DF &op);

		 Vector2DF &operator*= (const Vector2DC &op);
		 Vector2DF &operator*= (const Vector2DI &op);
		 Vector2DF &operator*= (const Vector2DF &op);
		 Vector2DF &operator*= (const Vector3DC &op);
		 Vector2DF &operator*= (const Vector3DI &op);
		 Vector2DF &operator*= (const Vector3DF &op);
		 Vector2DF &operator*= (const Vector4DF &op);

		 Vector2DF &operator/= (const Vector2DC &op);
		 Vector2DF &operator/= (const Vector2DI &op);
		 Vector2DF &operator/= (const Vector2DF &op);
		 Vector2DF &operator/= (const Vector3DC &op);
		 Vector2DF &operator/= (const Vector3DI &op);
		 Vector2DF &operator/= (const Vector3DF &op);
		 Vector2DF &operator/= (const Vector4DF &op);

		 Vector2DF &operator/= (const double v)		{x /= (float) v; y /= (float) v; return *this;}

		// Note: Cross product does not exist for 2D vectors (only 3D)
		
		 double Dot(const Vector2DC &v);
		 double Dot(const Vector2DI &v);
		 double Dot(const Vector2DF &v);

		 double Dist (const Vector2DC &v);
		 double Dist (const Vector2DI &v);
		 double Dist (const Vector2DF &v);
		 double Dist (const Vector3DC &v);
		 double Dist (const Vector3DI &v);
		 double Dist (const Vector3DF &v);
		 double Dist (const Vector4DF &v);

		 double DistSq (const Vector2DC &v);
		 double DistSq (const Vector2DI &v);
		 double DistSq (const Vector2DF &v);
		 double DistSq (const Vector3DC &v);
		 double DistSq (const Vector3DI &v);
		 double DistSq (const Vector3DF &v);
		 double DistSq (const Vector4DF &v);

		 Vector2DF &Normalize (void);
		 double Length (void);
		 
		 VTYPE &X(void)				{return x;}
		 VTYPE &Y(void)				{return y;}
		 VTYPE Z(void)					{return 0;}
		 VTYPE W(void)					{return 0;}
		 const VTYPE &X(void) const	{return x;}
		 const VTYPE &Y(void) const	{return y;}
		 const VTYPE Z(void) const		{return 0;}
		 const VTYPE W(void) const		{return 0;}
		 VTYPE *Data (void)			{return &x;}
	};
	
	#undef VNAME
	#undef VTYPE

	// Vector3DC Declaration
	
	#define VNAME		3DC
	#define VTYPE		unsigned char

	class Vector3DC {
	public:	
		VTYPE x, y, z;
	
		// Constructors/Destructors
		Vector3DC();
		Vector3DC (const VTYPE xa, const VTYPE ya, const VTYPE za);
		Vector3DC  ( const Vector2DC &op);
		Vector3DC  ( const Vector2DI &op);
		Vector3DC  ( const Vector2DF &op);
		Vector3DC  ( const Vector3DC &op);
		Vector3DC  ( const Vector3DI &op);
		Vector3DC  ( const Vector3DF &op);
		Vector3DC  ( const Vector4DF &op);

		// Member Functions
		Vector3DC &Set (VTYPE xa, VTYPE ya, VTYPE za);
		
		Vector3DC &operator=  ( const Vector2DC &op);
		Vector3DC &operator=  ( const Vector2DI &op);
		Vector3DC &operator=  ( const Vector2DF &op);
		Vector3DC &operator=  ( const Vector3DC &op);
		Vector3DC &operator=  ( const Vector3DI &op);
		Vector3DC &operator=  ( const Vector3DF &op);
		Vector3DC &operator=  ( const Vector4DF &op);
		
		Vector3DC &operator+=  ( const Vector2DC &op);
		Vector3DC &operator+=  ( const Vector2DI &op);
		Vector3DC &operator+=  ( const Vector2DF &op);
		Vector3DC &operator+=  ( const Vector3DC &op);
		Vector3DC &operator+=  ( const Vector3DI &op);
		Vector3DC &operator+=  ( const Vector3DF &op);
		Vector3DC &operator+=  ( const Vector4DF &op);

		Vector3DC &operator-=  ( const Vector2DC &op);
		Vector3DC &operator-=  ( const Vector2DI &op);
		Vector3DC &operator-=  ( const Vector2DF &op);
		Vector3DC &operator-=  ( const Vector3DC &op);
		Vector3DC &operator-=  ( const Vector3DI &op);
		Vector3DC &operator-=  ( const Vector3DF &op);
		Vector3DC &operator-=  ( const Vector4DF &op);
	
		Vector3DC &operator*=  ( const Vector2DC &op);
		Vector3DC &operator*=  ( const Vector2DI &op);
		Vector3DC &operator*=  ( const Vector2DF &op);
		Vector3DC &operator*=  ( const Vector3DC &op);
		Vector3DC &operator*=  ( const Vector3DI &op);
		Vector3DC &operator*=  ( const Vector3DF &op);
		Vector3DC &operator*=  ( const Vector4DF &op);

		Vector3DC &operator/=  ( const Vector2DC &op);
		Vector3DC &operator/=  ( const Vector2DI &op);
		Vector3DC &operator/=  ( const Vector2DF &op);
		Vector3DC &operator/=  ( const Vector3DC &op);
		Vector3DC &operator/=  ( const Vector3DI &op);
		Vector3DC &operator/=  ( const Vector3DF &op);
		Vector3DC &operator/=  ( const Vector4DF &op);

		Vector3DC &Cross  ( const Vector3DC &v);
		Vector3DC &Cross  ( const Vector3DI &v);
		Vector3DC &Cross  ( const Vector3DF &v);	
		
		double Dot ( const Vector3DC &v);
		double Dot ( const Vector3DI &v);
		double Dot ( const Vector3DF &v);

		double Dist  ( const Vector2DC &v);
		double Dist  ( const Vector2DI &v);
		double Dist  ( const Vector2DF &v);
		double Dist  ( const Vector3DC &v);
		double Dist  ( const Vector3DI &v);
		double Dist  ( const Vector3DF &v);
		double Dist  ( const Vector4DF &v);

		double DistSq  ( const Vector2DC &v);
		double DistSq  ( const Vector2DI &v);
		double DistSq  ( const Vector2DF &v);
		double DistSq  ( const Vector3DC &v);
		double DistSq  ( const Vector3DI &v);
		double DistSq  ( const Vector3DF &v);
		double DistSq  ( const Vector4DF &v);

		Vector3DC &Normalize (void);
		double Length (void);
	
		VTYPE &X(void)				{return x;}
		VTYPE &Y(void)				{return y;}
		VTYPE &Z(void)				{return z;}
		VTYPE W(void)					{return 0;}
		const VTYPE &X(void) const	{return x;}
		const VTYPE &Y(void) const	{return y;}
		const VTYPE &Z(void) const	{return z;}
		const VTYPE W(void) const		{return 0;}
		VTYPE *Data (void)			{return &x;}
	};
	
	#undef VNAME
	#undef VTYPE

	// Vector3DI Declaration

	#define VNAME		3DI
	#define VTYPE		int

	class Vector3DI {
	public:
		VTYPE x, y, z;
	
		// Constructors/Destructors
		Vector3DI();
		Vector3DI (const VTYPE xa, const VTYPE ya, const VTYPE za);
		Vector3DI (const Vector2DC &op);
		Vector3DI (const Vector2DI &op);
		Vector3DI (const Vector2DF &op);
		Vector3DI (const Vector3DC &op);
		Vector3DI (const Vector3DI &op);
		Vector3DI (const Vector3DF &op);
		Vector3DI (const Vector4DF &op);

		// Set Functions
		Vector3DI &Set (const VTYPE xa, const VTYPE ya, const VTYPE za);

		// Member Functions
		Vector3DI &operator= (const Vector2DC &op);
		Vector3DI &operator= (const Vector2DI &op);
		Vector3DI &operator= (const Vector2DF &op);
		Vector3DI &operator= (const Vector3DC &op);
		Vector3DI &operator= (const Vector3DI &op);
		Vector3DI &operator= (const Vector3DF &op);
		Vector3DI &operator= (const Vector4DF &op);
		
		Vector3DI &operator+= (const Vector2DC &op);
		Vector3DI &operator+= (const Vector2DI &op);
		Vector3DI &operator+= (const Vector2DF &op);
		Vector3DI &operator+= (const Vector3DC &op);
		Vector3DI &operator+= (const Vector3DI &op);
		Vector3DI &operator+= (const Vector3DF &op);
		Vector3DI &operator+= (const Vector4DF &op);

		Vector3DI &operator-= (const Vector2DC &op);
		Vector3DI &operator-= (const Vector2DI &op);
		Vector3DI &operator-= (const Vector2DF &op);
		Vector3DI &operator-= (const Vector3DC &op);
		Vector3DI &operator-= (const Vector3DI &op);
		Vector3DI &operator-= (const Vector3DF &op);
		Vector3DI &operator-= (const Vector4DF &op);
	
		Vector3DI &operator*= (const Vector2DC &op);
		Vector3DI &operator*= (const Vector2DI &op);
		Vector3DI &operator*= (const Vector2DF &op);
		Vector3DI &operator*= (const Vector3DC &op);
		Vector3DI &operator*= (const Vector3DI &op);
		Vector3DI &operator*= (const Vector3DF &op);
		Vector3DI &operator*= (const Vector4DF &op);

		Vector3DI &operator/= (const Vector2DC &op);
		Vector3DI &operator/= (const Vector2DI &op);
		Vector3DI &operator/= (const Vector2DF &op);
		Vector3DI &operator/= (const Vector3DC &op);
		Vector3DI &operator/= (const Vector3DI &op);
		Vector3DI &operator/= (const Vector3DF &op);
		Vector3DI &operator/= (const Vector4DF &op);

		Vector3DI operator+ (const int op)			{ return Vector3DI(x+(VTYPE) op, y+(VTYPE) op, z+(VTYPE) op); }
		Vector3DI operator+ (const float op)		{ return Vector3DI(x+(VTYPE) op, y+(VTYPE) op, z+(VTYPE) op); }
		Vector3DI operator+ (const Vector3DI &op)	{ return Vector3DI(x+op.x, y+op.y, z+op.z); }
		Vector3DI operator- (const int op)			{ return Vector3DI(x-(VTYPE) op, y-(VTYPE) op, z-(VTYPE) op); }
		Vector3DI operator- (const float op)		{ return Vector3DI(x-(VTYPE) op, y-(VTYPE) op, z-(VTYPE) op); }
		Vector3DI operator- (const Vector3DI &op)	{ return Vector3DI(x-op.x, y-op.y, z-op.z); }
		Vector3DI operator* (const int op)			{ return Vector3DI(x*(VTYPE) op, y*(VTYPE) op, z*(VTYPE) op); }
		Vector3DI operator* (const float op)		{ return Vector3DI(x*(VTYPE) op, y*(VTYPE) op, z*(VTYPE) op); }
		Vector3DI operator* (const Vector3DI &op)	{ return Vector3DI(x*op.x, y*op.y, z*op.z); }		

		Vector3DI &Cross (const Vector3DC &v);
		Vector3DI &Cross (const Vector3DI &v);
		Vector3DI &Cross (const Vector3DF &v);	
		
		double Dot(const Vector3DC &v);
		double Dot(const Vector3DI &v);
		double Dot(const Vector3DF &v);

		double Dist (const Vector2DC &v);
		double Dist (const Vector2DI &v);
		double Dist (const Vector2DF &v);
		double Dist (const Vector3DC &v);
		double Dist (const Vector3DI &v);
		double Dist (const Vector3DF &v);
		double Dist (const Vector4DF &v);

		double DistSq (const Vector2DC &v);
		double DistSq (const Vector2DI &v);
		double DistSq (const Vector2DF &v);
		double DistSq (const Vector3DC &v);
		double DistSq (const Vector3DI &v);
		double DistSq (const Vector3DF &v);
		double DistSq (const Vector4DF &v);

		Vector3DI &Normalize (void);
		double Length (void);

		VTYPE &X(void)				{return x;}
		VTYPE &Y(void)				{return y;}
		VTYPE &Z(void)				{return z;}
		VTYPE W(void)					{return 0;}
		const VTYPE &X(void) const	{return x;}
		const VTYPE &Y(void) const	{return y;}
		const VTYPE &Z(void) const	{return z;}
		const VTYPE W(void) const		{return 0;}
		VTYPE *Data (void)			{return &x;}
	};
	
	#undef VNAME
	#undef VTYPE

	// Vector3DF Declarations

	#define VNAME		3DF
	#define VTYPE		float

	class Vector3DF {
	public:
		VTYPE x, y, z;
	
		// Constructors/Destructors
		Vector3DF() {x=0; y=0; z=0;}
		Vector3DF (const VTYPE xa, const VTYPE ya, const VTYPE za);
		Vector3DF (const Vector2DC &op);
		Vector3DF (const Vector2DI &op);
		Vector3DF (const Vector2DF &op);
		Vector3DF (const Vector3DC &op);
		Vector3DF (const Vector3DI &op);
		Vector3DF (const Vector3DF &op);
		Vector3DF (const Vector4DF &op);

		// Set Functions
		Vector3DF &Set (const VTYPE xa, const VTYPE ya, const VTYPE za);
		
		// Member Functions
		Vector3DF &operator= (const int op);
		Vector3DF &operator= (const double op);
		Vector3DF &operator= (const Vector2DC &op);
		Vector3DF &operator= (const Vector2DI &op);
		Vector3DF &operator= (const Vector2DF &op);
		Vector3DF &operator= (const Vector3DC &op);
		Vector3DF &operator= (const Vector3DI &op);
		Vector3DF &operator= (const Vector3DF &op);
		Vector3DF &operator= (const Vector4DF &op);

		Vector3DF &operator+= (const int op);
		Vector3DF &operator+= (const double op);
		Vector3DF &operator+= (const Vector2DC &op);
		Vector3DF &operator+= (const Vector2DI &op);
		Vector3DF &operator+= (const Vector2DF &op);
		Vector3DF &operator+= (const Vector3DC &op);
		Vector3DF &operator+= (const Vector3DI &op);
		Vector3DF &operator+= (const Vector3DF &op);
		Vector3DF &operator+= (const Vector4DF &op);

		Vector3DF &operator-= (const int op);
		Vector3DF &operator-= (const double op);
		Vector3DF &operator-= (const Vector2DC &op);
		Vector3DF &operator-= (const Vector2DI &op);
		Vector3DF &operator-= (const Vector2DF &op);
		Vector3DF &operator-= (const Vector3DC &op);
		Vector3DF &operator-= (const Vector3DI &op);
		Vector3DF &operator-= (const Vector3DF &op);
		Vector3DF &operator-= (const Vector4DF &op);
	
		Vector3DF &operator*= (const int op);
		Vector3DF &operator*= (const double op);
		Vector3DF &operator*= (const Vector2DC &op);
		Vector3DF &operator*= (const Vector2DI &op);
		Vector3DF &operator*= (const Vector2DF &op);
		Vector3DF &operator*= (const Vector3DC &op);
		Vector3DF &operator*= (const Vector3DI &op);
		Vector3DF &operator*= (const Vector3DF &op);
		Vector3DF &operator*= (const Vector4DF &op);
		Vector3DF &operator*= (const Matrix4F &op);
		Vector3DF &operator*= (const MatrixF &op);				// see vector.cpp

		Vector3DF &operator/= (const int op);
		Vector3DF &operator/= (const double op);
		Vector3DF &operator/= (const Vector2DC &op);
		Vector3DF &operator/= (const Vector2DI &op);
		Vector3DF &operator/= (const Vector2DF &op);
		Vector3DF &operator/= (const Vector3DC &op);
		Vector3DF &operator/= (const Vector3DI &op);
		Vector3DF &operator/= (const Vector3DF &op);
		Vector3DF &operator/= (const Vector4DF &op);

		// Slow operations - require temporary variables
		Vector3DF operator+ (int op)			{ return Vector3DF(x+float(op), y+float(op), z+float(op)); }
		Vector3DF operator+ (float op)		{ return Vector3DF(x+op, y+op, z+op); }
		Vector3DF operator+ (Vector3DF &op)	{ return Vector3DF(x+op.x, y+op.y, z+op.z); }
		Vector3DF operator- (int op)			{ return Vector3DF(x-float(op), y-float(op), z-float(op)); }
		Vector3DF operator- (float op)		{ return Vector3DF(x-op, y-op, z-op); }
		Vector3DF operator- (Vector3DF &op)	{ return Vector3DF(x-op.x, y-op.y, z-op.z); }
		Vector3DF operator* (int op)			{ return Vector3DF(x*float(op), y*float(op), z*float(op)); }
		Vector3DF operator* (float op)		{ return Vector3DF(x*op, y*op, z*op); }
		Vector3DF operator* (Vector3DF &op)	{ return Vector3DF(x*op.x, y*op.y, z*op.z); }		
		// --


		Vector3DF &Cross (const Vector3DC &v);
		Vector3DF &Cross (const Vector3DI &v);
		Vector3DF &Cross (const Vector3DF &v);	
		
		double Dot(const Vector3DC &v);
		double Dot(const Vector3DI &v);
		double Dot(const Vector3DF &v);

		double Dist (const Vector2DC &v);
		double Dist (const Vector2DI &v);
		double Dist (const Vector2DF &v);
		double Dist (const Vector3DC &v);
		double Dist (const Vector3DI &v);
		double Dist (const Vector3DF &v);
		double Dist (const Vector4DF &v);

		double DistSq (const Vector2DC &v);
		double DistSq (const Vector2DI &v);
		double DistSq (const Vector2DF &v);
		double DistSq (const Vector3DC &v);
		double DistSq (const Vector3DI &v);
		double DistSq (const Vector3DF &v);
		double DistSq (const Vector4DF &v);

		Vector3DF &Random ()		{ x=float(rand())/RAND_MAX; y=float(rand())/RAND_MAX; z=float(rand())/RAND_MAX;  return *this;}
		Vector3DF &Random (Vector3DF a, Vector3DF b)		{ x=a.x+float(rand()*(b.x-a.x))/RAND_MAX; y=a.y+float(rand()*(b.y-a.y))/RAND_MAX; z=a.z+float(rand()*(b.z-a.z))/RAND_MAX;  return *this;}
		Vector3DF &Random (float x1,float x2, float y1, float y2, float z1, float z2)	{ x=x1+float(rand()*(x2-x1))/RAND_MAX; y=y1+float(rand()*(y2-y1))/RAND_MAX; z=z1+float(rand()*(z2-z1))/RAND_MAX;  return *this;}

		Vector3DF RGBtoHSV ();
		Vector3DF HSVtoRGB ();

		Vector3DF &Normalize (void);
		double Length (void);
		
		VTYPE &X()				{return x;}
		VTYPE &Y()				{return y;}
		VTYPE &Z()				{return z;}
		VTYPE W()					{return 0;}
		const VTYPE &X() const	{return x;}
		const VTYPE &Y() const	{return y;}
		const VTYPE &Z() const	{return z;}
		const VTYPE W() const		{return 0;}
		VTYPE *Data ()			{return &x;}
	};
	
	#undef VNAME
	#undef VTYPE

	// Vector4DC Declarations

	#define VNAME		4DC
	#define VTYPE		unsigned char

	class Vector4DC {
	public:
		VTYPE x, y, z, w;
	
		Vector4DC &Set (const float xa, const float ya, const float za)	{ x = (VTYPE) xa; y= (VTYPE) ya; z=(VTYPE) za; w=1; return *this;}
		Vector4DC &Set (const float xa, const float ya, const float za, const float wa )	{ x =(VTYPE) xa; y= (VTYPE) ya; z=(VTYPE) za; w=(VTYPE) wa; return *this;}
		Vector4DC &Set (const VTYPE xa, const VTYPE ya, const VTYPE za)	{ x = (VTYPE) xa; y= (VTYPE) ya; z=(VTYPE) za; w=1; return *this;}
		Vector4DC &Set (const VTYPE xa, const VTYPE ya, const VTYPE za, const VTYPE wa )	{ x =(VTYPE) xa; y= (VTYPE) ya; z=(VTYPE) za; w=(VTYPE) wa; return *this;}

		// Constructors/Destructors
		Vector4DC();
		Vector4DC (const VTYPE xa, const VTYPE ya, const VTYPE za, const VTYPE wa);
		Vector4DC (const Vector2DC &op);
		Vector4DC (const Vector2DI &op);
		Vector4DC (const Vector2DF &op);
		Vector4DC (const Vector3DC &op);
		Vector4DC (const Vector3DI &op);
		Vector4DC (const Vector3DF &op);
		Vector4DC (const Vector4DC &op);
		Vector4DC (const Vector4DF &op);

		// Member Functions
		Vector4DC &operator= ( const int op);
		Vector4DC &operator= ( const double op);
		Vector4DC &operator= ( const Vector2DC &op);
		Vector4DC &operator= ( const Vector2DI &op);
		Vector4DC &operator= ( const Vector2DF &op);
		Vector4DC &operator= ( const Vector3DC &op);
		Vector4DC &operator= ( const Vector3DI &op);
		Vector4DC &operator= ( const Vector3DF &op);
		Vector4DC &operator= ( const Vector4DC &op);
		Vector4DC &operator= ( const Vector4DF &op);

		Vector4DC &operator+= ( const int op);
		Vector4DC &operator+= ( const double op);
		Vector4DC &operator+= ( const Vector2DC &op);
		Vector4DC &operator+= ( const Vector2DI &op);
		Vector4DC &operator+= ( const Vector2DF &op);
		Vector4DC &operator+= ( const Vector3DC &op);
		Vector4DC &operator+= ( const Vector3DI &op);
		Vector4DC &operator+= ( const Vector3DF &op);
		Vector4DC &operator+= ( const Vector4DC &op);
		Vector4DC &operator+= ( const Vector4DF &op);

		Vector4DC &operator-= ( const int op);
		Vector4DC &operator-= ( const double op);
		Vector4DC &operator-= ( const Vector2DC &op);
		Vector4DC &operator-= ( const Vector2DI &op);
		Vector4DC &operator-= ( const Vector2DF &op);
		Vector4DC &operator-= ( const Vector3DC &op);
		Vector4DC &operator-= ( const Vector3DI &op);
		Vector4DC &operator-= ( const Vector3DF &op);
		Vector4DC &operator-= ( const Vector4DC &op);
		Vector4DC &operator-= ( const Vector4DF &op);

		Vector4DC &operator*= ( const int op);
		Vector4DC &operator*= ( const double op);
		Vector4DC &operator*= ( const Vector2DC &op);
		Vector4DC &operator*= ( const Vector2DI &op);
		Vector4DC &operator*= ( const Vector2DF &op);
		Vector4DC &operator*= ( const Vector3DC &op);
		Vector4DC &operator*= ( const Vector3DI &op);
		Vector4DC &operator*= ( const Vector3DF &op);
		Vector4DC &operator*= ( const Vector4DC &op);
		Vector4DC &operator*= ( const Vector4DF &op);
		
		Vector4DC &operator/= ( const int op);
		Vector4DC &operator/= ( const double op);
		Vector4DC &operator/= ( const Vector2DC &op);
		Vector4DC &operator/= ( const Vector2DI &op);
		Vector4DC &operator/= ( const Vector2DF &op);
		Vector4DC &operator/= ( const Vector3DC &op);
		Vector4DC &operator/= ( const Vector3DI &op);
		Vector4DC &operator/= ( const Vector3DF &op);
		Vector4DC &operator/= ( const Vector4DC &op);
		Vector4DC &operator/= ( const Vector4DF &op);

		// Slow operations - require temporary variables
		Vector4DC operator+ ( const int op);
		Vector4DC operator+ ( const float op);
		Vector4DC operator+ ( const Vector4DC &op);
		Vector4DC operator- ( const int op);
		Vector4DC operator- ( const float op);
		Vector4DC operator- ( const Vector4DC &op);
		Vector4DC operator* ( const int op);
		Vector4DC operator* ( const float op);
		Vector4DC operator* ( const Vector4DC &op);
		// --

		double Dot( const Vector4DF &v);
		double Dist ( const Vector4DF &v);
		double DistSq ( const Vector4DF &v);
		Vector4DC &Normalize (void);
		double Length (void);

		Vector4DC &Random ()		{ x=(VTYPE) float(rand()*255)/RAND_MAX; y=(VTYPE) float(rand()*255)/RAND_MAX; z=(VTYPE) float(rand()*255)/RAND_MAX; w = 1;  return *this;}
		VTYPE *Data (void);
	};
	#undef VNAME
	#undef VTYPE


	// Vector4DF Declarations

	#define VNAME		4DF
	#define VTYPE		float

	class Vector4DF {
	public:
		VTYPE x, y, z, w;
	
		Vector4DF &Set (const float xa, const float ya, const float za)	{ x =xa; y= ya; z=za; w=1; return *this;}
		Vector4DF &Set (const float xa, const float ya, const float za, const float wa )	{ x =xa; y= ya; z=za; w=wa; return *this;}

		// Constructors/Destructors
		Vector4DF() {x=0; y=0; z=0; w=0;}
		Vector4DF (const VTYPE xa, const VTYPE ya, const VTYPE za, const VTYPE wa);
		Vector4DF (const Vector2DC &op);
		Vector4DF (const Vector2DI &op);
		Vector4DF (const Vector2DF &op);
		Vector4DF (const Vector3DC &op);
		Vector4DF (const Vector3DI &op);
		Vector4DF (const Vector3DF &op);
		Vector4DF (const Vector4DF &op);

		// Member Functions
		Vector4DF &operator= (const int op);
		Vector4DF &operator= (const double op);
		Vector4DF &operator= (const Vector2DC &op);
		Vector4DF &operator= (const Vector2DI &op);
		Vector4DF &operator= (const Vector2DF &op);
		Vector4DF &operator= (const Vector3DC &op);
		Vector4DF &operator= (const Vector3DI &op);
		Vector4DF &operator= (const Vector3DF &op);
		Vector4DF &operator= (const Vector4DF &op);

		Vector4DF &operator+= (const int op);
		Vector4DF &operator+= (const float op);
		Vector4DF &operator+= (const double op);
		Vector4DF &operator+= (const Vector2DC &op);
		Vector4DF &operator+= (const Vector2DI &op);
		Vector4DF &operator+= (const Vector2DF &op);
		Vector4DF &operator+= (const Vector3DC &op);
		Vector4DF &operator+= (const Vector3DI &op);
		Vector4DF &operator+= (const Vector3DF &op);
		Vector4DF &operator+= (const Vector4DF &op);

		Vector4DF &operator-= (const int op);
		Vector4DF &operator-= (const double op);
		Vector4DF &operator-= (const Vector2DC &op);
		Vector4DF &operator-= (const Vector2DI &op);
		Vector4DF &operator-= (const Vector2DF &op);
		Vector4DF &operator-= (const Vector3DC &op);
		Vector4DF &operator-= (const Vector3DI &op);
		Vector4DF &operator-= (const Vector3DF &op);
		Vector4DF &operator-= (const Vector4DF &op);

		Vector4DF &operator*= (const int op);
		Vector4DF &operator*= (const double op);
		Vector4DF &operator*= (const Vector2DC &op);
		Vector4DF &operator*= (const Vector2DI &op);
		Vector4DF &operator*= (const Vector2DF &op);
		Vector4DF &operator*= (const Vector3DC &op);
		Vector4DF &operator*= (const Vector3DI &op);
		Vector4DF &operator*= (const Vector3DF &op);
		Vector4DF &operator*= (const Vector4DF &op);
		Vector4DF &operator*= (const float* op );
		Vector4DF &operator*= (const Matrix4F &op);
		Vector4DF &operator*= (const MatrixF &op);				// see vector.cpp

		Vector4DF &operator/= (const int op);
		Vector4DF &operator/= (const double op);
		Vector4DF &operator/= (const Vector2DC &op);
		Vector4DF &operator/= (const Vector2DI &op);
		Vector4DF &operator/= (const Vector2DF &op);
		Vector4DF &operator/= (const Vector3DC &op);
		Vector4DF &operator/= (const Vector3DI &op);
		Vector4DF &operator/= (const Vector3DF &op);
		Vector4DF &operator/= (const Vector4DF &op);

		// Slow operations - require temporary variables
		Vector4DF operator+ (const int op)			{ return Vector4DF(x+float(op), y+float(op), z+float(op), w+float(op)); }
		Vector4DF operator+ (const float op)		{ return Vector4DF(x+op, y+op, z+op, w*op); }
		Vector4DF operator+ (const Vector4DF &op)	{ return Vector4DF(x+op.x, y+op.y, z+op.z, w+op.w); }
		Vector4DF operator- (const int op)			{ return Vector4DF(x-float(op), y-float(op), z-float(op), w-float(op)); }
		Vector4DF operator- (const float op)		{ return Vector4DF(x-op, y-op, z-op, w*op); }
		Vector4DF operator- (const Vector4DF &op)	{ return Vector4DF(x-op.x, y-op.y, z-op.z, w-op.w); }
		Vector4DF operator* (const int op)			{ return Vector4DF(x*float(op), y*float(op), z*float(op), w*float(op)); }
		Vector4DF operator* (const float op)		{ return Vector4DF(x*op, y*op, z*op, w*op); }
		Vector4DF operator* (const Vector4DF &op)	{ return Vector4DF(x*op.x, y*op.y, z*op.z, w*op.w); }		
		// --

		Vector4DF &Set ( CLRVAL clr )	{
			x = (float) RED(clr);		// (float( c      & 0xFF)/255.0)	
			y = (float) GRN(clr);		// (float((c>>8)  & 0xFF)/255.0)
			z = (float) BLUE(clr);		// (float((c>>16) & 0xFF)/255.0)
			w = (float) ALPH(clr);		// (float((c>>24) & 0xFF)/255.0)
			return *this;
		}
		Vector4DF& fromClr ( CLRVAL clr ) { return Set (clr); }
		CLRVAL toClr () { return (CLRVAL) COLORA( x, y, z, w ); }

		Vector4DF& Clamp ( float xc, float yc, float zc, float wc )
		{
			x = (x > xc) ? xc : x;
			y = (y > yc) ? yc : y;
			z = (z > zc) ? zc : z;
			w = (w > wc) ? wc : w;
			return *this;
		}

		Vector4DF &Cross (const Vector4DF &v);	
		
		double Dot (const Vector4DF &v);

		double Dist (const Vector4DF &v);

		double DistSq (const Vector4DF &v);

		Vector4DF &Normalize (void);
		double Length (void);

		Vector4DF &Random ()		{ x=float(rand())/RAND_MAX; y=float(rand())/RAND_MAX; z=float(rand())/RAND_MAX; w = 1;  return *this;}

		VTYPE &X(void)				{return x;}
		VTYPE &Y(void)				{return y;}
		VTYPE &Z(void)				{return z;}
		VTYPE &W(void)				{return w;}
		const VTYPE &X(void) const	{return x;}
		const VTYPE &Y(void) const	{return y;}
		const VTYPE &Z(void) const	{return z;}
		const VTYPE &W(void) const	{return w;}
		VTYPE *Data (void)			{return &x;}
	};
	
	#undef VNAME
	#undef VTYPE

#endif

#ifndef MATRIX_DEF
	#define MATRIX_DEF
		
	#include <stdio.h>
	#include <iostream>
	#include <memory.h>
	#include <math.h>
	#include <string>


	//#define MATRIX_INITIALIZE				// Initializes vectors	

	class MatrixC;							// Forward Referencing
	class MatrixI;
	class MatrixF;

	class Matrix {
	public:
		// Member Virtual Functions		
		virtual Matrix &operator= (unsigned char c)=0;
		virtual Matrix &operator= (int c)=0;
		virtual Matrix &operator= (double c)=0;		
		virtual Matrix &operator= (MatrixC &op)=0;
		virtual Matrix &operator= (MatrixI &op)=0;
		virtual Matrix &operator= (MatrixF &op)=0;
		
		virtual Matrix &operator+= (unsigned char c)=0;
		virtual Matrix &operator+= (int c)=0;
		virtual Matrix &operator+= (double c)=0;		
		virtual Matrix &operator+= (MatrixC &op)=0;
		virtual Matrix &operator+= (MatrixI &op)=0;
		virtual Matrix &operator+= (MatrixF &op)=0;

		virtual Matrix &operator-= (unsigned char c)=0;
		virtual Matrix &operator-= (int c)=0;
		virtual Matrix &operator-= (double c)=0;		
		virtual Matrix &operator-= (MatrixC &op)=0;
		virtual Matrix &operator-= (MatrixI &op)=0;
		virtual Matrix &operator-= (MatrixF &op)=0;

		virtual Matrix &operator*= (unsigned char c)=0;
		virtual Matrix &operator*= (int c)=0;
		virtual Matrix &operator*= (double c)=0;		
		virtual Matrix &operator*= (MatrixC &op)=0;
		virtual Matrix &operator*= (MatrixI &op)=0;
		virtual Matrix &operator*= (MatrixF &op)=0;

		virtual Matrix &operator/= (unsigned char c)=0;
		virtual Matrix &operator/= (int c)=0;
		virtual Matrix &operator/= (double c)=0;		
		virtual Matrix &operator/= (MatrixC &op)=0;
		virtual Matrix &operator/= (MatrixI &op)=0;
		virtual Matrix &operator/= (MatrixF &op)=0;

		virtual Matrix &Multiply (MatrixF &op)=0;
		virtual Matrix &Resize (int x, int y)=0;
		virtual Matrix &ResizeSafe (int x, int y)=0;
		virtual Matrix &InsertRow (int r)=0;
		virtual Matrix &InsertCol (int c)=0;
		virtual Matrix &Transpose (void)=0;
		virtual Matrix &Identity (int order)=0;
		/*Matrix &RotateX (double ang);
		Matrix &RotateY (double ang);
		Matrix &RotateZ (double ang); */
		virtual Matrix &Basis (Vector3DF &c1, Vector3DF &c2, Vector3DF &c3)=0;
		virtual Matrix &GaussJordan (MatrixF &b)		{ return *this; }
		virtual Matrix &ConjugateGradient (MatrixF &b)	{ return *this; }

		virtual int GetRows(void)=0;
		virtual int GetCols(void)=0;
		virtual int GetLength(void)=0;		

		virtual unsigned char *GetDataC (void)=0;
		virtual int	*GetDataI (void)=0;
		virtual double *GetDataF (void)=0;
	};
	
	// MatrixC Declaration	
	#define VNAME		C
	#define VTYPE		unsigned char

	class MatrixC {
	public:
		VTYPE *data;
		int rows, cols, len;		

		// Constructors/Destructors
		MatrixC ();
		~MatrixC ();
		MatrixC (int r, int c);

		// Member Functions
		VTYPE &operator () (int c, int r);
		MatrixC &operator= (unsigned char c);
		MatrixC &operator= (int c);
		MatrixC &operator= (double c);		
		MatrixC &operator= (MatrixC &op);
		MatrixC &operator= (MatrixI &op);
		MatrixC &operator= (MatrixF &op);
		
		MatrixC &operator+= (unsigned char c);
		MatrixC &operator+= (int c);
		MatrixC &operator+= (double c);		
		MatrixC &operator+= (MatrixC &op);
		MatrixC &operator+= (MatrixI &op);
		MatrixC &operator+= (MatrixF &op);

		MatrixC &operator-= (unsigned char c);
		MatrixC &operator-= (int c);
		MatrixC &operator-= (double c);		
		MatrixC &operator-= (MatrixC &op);
		MatrixC &operator-= (MatrixI &op);
		MatrixC &operator-= (MatrixF &op);

		MatrixC &operator*= (unsigned char c);
		MatrixC &operator*= (int c);
		MatrixC &operator*= (double c);		
		MatrixC &operator*= (MatrixC &op);
		MatrixC &operator*= (MatrixI &op);
		MatrixC &operator*= (MatrixF &op);

		MatrixC &operator/= (unsigned char c);
		MatrixC &operator/= (int c);
		MatrixC &operator/= (double c);		
		MatrixC &operator/= (MatrixC &op);
		MatrixC &operator/= (MatrixI &op);
		MatrixC &operator/= (MatrixF &op);

		MatrixC &Multiply (MatrixF &op);
		MatrixC &Resize (int x, int y);
		MatrixC &ResizeSafe (int x, int y);
		MatrixC &InsertRow (int r);
		MatrixC &InsertCol (int c);
		MatrixC &Transpose (void);
		MatrixC &Identity (int order);		
		MatrixC &Basis (Vector3DF &c1, Vector3DF &c2, Vector3DF &c3);
		MatrixC &GaussJordan (MatrixF &b);

		int GetX();
		int GetY();	
		int GetRows(void);
		int GetCols(void);
		int GetLength(void);
		VTYPE *GetData(void);

		unsigned char *GetDataC (void)	{return data;}
		int *GetDataI (void)				{return NULL;}
		double *GetDataF (void)			{return NULL;}		

		double GetF (int r, int c);
	};
	#undef VNAME
	#undef VTYPE

	// MatrixI Declaration	
	#define VNAME		I
	#define VTYPE		int

	class MatrixI {
	public:
		VTYPE *data;
		int rows, cols, len;		
	
		// Constructors/Destructors
		MatrixI ();
		~MatrixI ();
		MatrixI (int r, int c);

		// Member Functions
		VTYPE &operator () (int c, int r);
		MatrixI &operator= (unsigned char c);
		MatrixI &operator= (int c);
		MatrixI &operator= (double c);		
		MatrixI &operator= (MatrixC &op);
		MatrixI &operator= (MatrixI &op);
		MatrixI &operator= (MatrixF &op);
		
		MatrixI &operator+= (unsigned char c);
		MatrixI &operator+= (int c);
		MatrixI &operator+= (double c);		
		MatrixI &operator+= (MatrixC &op);
		MatrixI &operator+= (MatrixI &op);
		MatrixI &operator+= (MatrixF &op);

		MatrixI &operator-= (unsigned char c);
		MatrixI &operator-= (int c);
		MatrixI &operator-= (double c);		
		MatrixI &operator-= (MatrixC &op);
		MatrixI &operator-= (MatrixI &op);
		MatrixI &operator-= (MatrixF &op);

		MatrixI &operator*= (unsigned char c);
		MatrixI &operator*= (int c);
		MatrixI &operator*= (double c);		
		MatrixI &operator*= (MatrixC &op);
		MatrixI &operator*= (MatrixI &op);
		MatrixI &operator*= (MatrixF &op);

		MatrixI &operator/= (unsigned char c);
		MatrixI &operator/= (int c);
		MatrixI &operator/= (double c);		
		MatrixI &operator/= (MatrixC &op);
		MatrixI &operator/= (MatrixI &op);
		MatrixI &operator/= (MatrixF &op);

		MatrixI &Multiply (MatrixF &op);
		MatrixI &Resize (int x, int y);
		MatrixI &ResizeSafe (int x, int y);
		MatrixI &InsertRow (int r);
		MatrixI &InsertCol (int c);
		MatrixI &Transpose (void);
		MatrixI &Identity (int order);		
		MatrixI &Basis (Vector3DF &c1, Vector3DF &c2, Vector3DF &c3);
		MatrixI &GaussJordan (MatrixF &b);

		int GetX();
		int GetY();	
		int GetRows(void);
		int GetCols(void);
		int GetLength(void);
		VTYPE *GetData(void);

		unsigned char *GetDataC (void)	{return NULL;}
		int *GetDataI (void)				{return data;}
		double *GetDataF (void)			{return NULL;}
		
		double GetF (int r, int c);
	};
	#undef VNAME
	#undef VTYPE

	// MatrixF Declaration	
	#define VNAME		F
	#define VTYPE		double

	class MatrixF {
	public:	
		VTYPE *data;
		int rows, cols, len;		

		// Constructors/Destructors		
		MatrixF ();
		~MatrixF ();
		MatrixF (const int r, const int c);

		// Member Functions
		VTYPE GetVal ( int c, int r );
		VTYPE &operator () (const int c, const int r);
		MatrixF &operator= (const unsigned char c);
		MatrixF &operator= (const int c);
		MatrixF &operator= (const double c);		
		MatrixF &operator= (const MatrixC &op);
		MatrixF &operator= (const MatrixI &op);
		MatrixF &operator= (const MatrixF &op);
		
		MatrixF &operator+= (const unsigned char c);
		MatrixF &operator+= (const int c);
		MatrixF &operator+= (const double c);		
		MatrixF &operator+= (const MatrixC &op);
		MatrixF &operator+= (const MatrixI &op);
		MatrixF &operator+= (const MatrixF &op);

		MatrixF &operator-= (const unsigned char c);
		MatrixF &operator-= (const int c);
		MatrixF &operator-= (const double c);		
		MatrixF &operator-= (const MatrixC &op);
		MatrixF &operator-= (const MatrixI &op);
		MatrixF &operator-= (const MatrixF &op);

		MatrixF &operator*= (const unsigned char c);
		MatrixF &operator*= (const int c);
		MatrixF &operator*= (const double c);		
		MatrixF &operator*= (const MatrixC &op);
		MatrixF &operator*= (const MatrixI &op);
		MatrixF &operator*= (const MatrixF &op);		

		MatrixF &operator/= (const unsigned char c);
		MatrixF &operator/= (const int c);
		MatrixF &operator/= (const double c);		
		MatrixF &operator/= (const MatrixC &op);
		MatrixF &operator/= (const MatrixI &op);
		MatrixF &operator/= (const MatrixF &op);

		MatrixF &Multiply4x4 (const MatrixF &op);
		MatrixF &Multiply (const MatrixF &op);
		MatrixF &Resize (const int x, const int y);
		MatrixF &ResizeSafe (const int x, const int y);
		MatrixF &InsertRow (const int r);
		MatrixF &InsertCol (const int c);
		MatrixF &Transpose (void);
		MatrixF &Identity (const int order);
		MatrixF &RotateX (const double ang);
		MatrixF &RotateY (const double ang);
		MatrixF &RotateZ (const double ang);
		MatrixF &Ortho (double sx, double sy, double n, double f);		
		MatrixF &Translate (double tx, double ty, double tz);
		MatrixF &Basis (const Vector3DF &c1, const Vector3DF &c2, const Vector3DF &c3);
		MatrixF &GaussJordan (MatrixF &b);
		MatrixF &ConjugateGradient (MatrixF &b);
		MatrixF &Submatrix ( MatrixF& b, int mx, int my);
		MatrixF &MatrixVector5 (MatrixF& x, int mrows, MatrixF& b );
		MatrixF &ConjugateGradient5 (MatrixF &b, int mrows );
		double Dot ( MatrixF& b );

		void Print ( char* fname );

		int GetX();
		int GetY();	
		int GetRows(void);
		int GetCols(void);
		int GetLength(void);
		VTYPE *GetData(void);
		void GetRowVec (int r, Vector3DF &v);

		unsigned char *GetDataC (void) const	{return NULL;}
		int *GetDataI (void)	const			{return NULL;}
		double *GetDataF (void) const		{return data;}

		double GetF (const int r, const int c);
	};
	#undef VNAME
	#undef VTYPE

	// MatrixF Declaration	
	#define VNAME		F
	#define VTYPE		float

	class LUNA_CORE Matrix4F {
	public:	
		VTYPE	data[16];		

		// Constructors/Destructors
		Matrix4F ( float* dat );
		Matrix4F () { for (int n=0; n < 16; n++) data[n] = 0.0; }
		Matrix4F ( float f0, float f1, float f2, float f3, float f4, float f5, float f6, float f7, float f8, float f9, float f10, float f11,	float f12, float f13, float f14, float f15 );

		// Member Functions
		VTYPE &operator () (const int n)					{ return data[n]; }
		VTYPE &operator () (const int c, const int r)	{ return data[ (r<<2)+c ]; }		
		Matrix4F &operator= (const unsigned char c);
		Matrix4F &operator= (const int c);
		Matrix4F &operator= (const double c);				
		Matrix4F &operator+= (const unsigned char c);
		Matrix4F &operator+= (const int c);
		Matrix4F &operator+= (const double c);				
		Matrix4F &operator-= (const unsigned char c);
		Matrix4F &operator-= (const int c);
		Matrix4F &operator-= (const double c);
		Matrix4F &operator*= (const unsigned char c);
		Matrix4F &operator*= (const int c);
		Matrix4F &operator*= (const double c);
		Matrix4F &operator/= (const unsigned char c);
		Matrix4F &operator/= (const int c);
		Matrix4F &operator/= (const double c);		

		Matrix4F &operator=  (const float* op);
		Matrix4F &operator*= (const Matrix4F& op);
		Matrix4F &operator*= (const float* op);	

		Matrix4F &PreTranslate (const Vector3DF& t);
		Matrix4F &operator+= (const Vector3DF& t);		// quick translate
		Matrix4F &operator*= (const Vector3DF& t);		// quick scale
		
		Matrix4F &Transpose (void);
		Matrix4F &RotateZYX ( const Vector3DF& angs );
		Matrix4F &RotateZYXT (const Vector3DF& angs, const Vector3DF& t);
		Matrix4F &RotateTZYX (const Vector3DF& angs, const Vector3DF& t);
		Matrix4F &RotateX (const double ang);
		Matrix4F &RotateY (const double ang);
		Matrix4F &RotateZ (const double ang);
		Matrix4F &Ortho (double sx, double sy, double n, double f);		
		Matrix4F &Translate (double tx, double ty, double tz);
		Matrix4F &Scale (double sx, double sy, double sz);
		Matrix4F &Basis (const Vector3DF &yaxis);
		Matrix4F &Basis (const Vector3DF &c1, const Vector3DF &c2, const Vector3DF &c3);		
		Matrix4F &InvertTRS ();
		Matrix4F &Identity ();
		Matrix4F &Identity (const int order);
		Matrix4F &Multiply (const Matrix4F &op);

		void Print ();
		std::string WriteToStr ();

		Matrix4F operator* (const float &op);	
		Matrix4F operator* (const Vector3DF &op);	

		// Scale-Rotate-Translate (compound matrix)
		Matrix4F &TransSRT (const Vector3DF &c1, const Vector3DF &c2, const Vector3DF &c3, const Vector3DF& t, const Vector3DF& s);
		Matrix4F &SRT (const Vector3DF &c1, const Vector3DF &c2, const Vector3DF &c3, const Vector3DF& t, const Vector3DF& s);
		Matrix4F &SRT (const Vector3DF &c1, const Vector3DF &c2, const Vector3DF &c3, const Vector3DF& t, const float s);

		// invTranslate-invRotate-invScale (compound matrix)
		Matrix4F &InvTRS (const Vector3DF &c1, const Vector3DF &c2, const Vector3DF &c3, const Vector3DF& t, const Vector3DF& s);
		Matrix4F &InvTRS (const Vector3DF &c1, const Vector3DF &c2, const Vector3DF &c3, const Vector3DF& t, const float s);

		Matrix4F &operator= ( float* mat);
		Matrix4F &InverseProj ( float* mat );
		Matrix4F &InverseView ( float* mat, Vector3DF& pos );
		Vector4DF GetT ( float* mat );

		int GetX()			{ return 4; }
		int GetY()			{ return 4; }
		int GetRows(void)	{ return 4; }
		int GetCols(void)	{ return 4; }	
		int GetLength(void)	{ return 16; }
		VTYPE *GetData(void)	{ return data; }
		void GetRowVec (int r, Vector3DF &v);

		unsigned char *GetDataC (void) const	{return NULL;}
		int *GetDataI (void)	const			{return NULL;}
		float *GetDataF (void) const		{return (float*) data;}

		float GetF (const int r, const int c);
	};
	#undef VNAME
	#undef VTYPE

#endif


#ifndef DEF_PIVOTX
	#define DEF_PIVOTX
	
	class PivotX {
	public:
		PivotX()	{ from_pos.Set(0,0,0); to_pos.Set(0,0,0); ang_euler.Set(0,0,0); scale.Set(1,1,1); trans.Identity(); }
		PivotX( Vector3DF& f, Vector3DF& t, Vector3DF& s, Vector3DF& a) { from_pos=f; to_pos=t; scale=s; ang_euler=a; }

		void setPivot ( float x, float y, float z, float rx, float ry, float rz );
		void setPivot ( Vector3DF& pos, Vector3DF& ang ) { from_pos = pos; ang_euler = ang; }
		void setPivot ( PivotX  piv )	{ from_pos = piv.from_pos; to_pos = piv.to_pos; ang_euler = piv.ang_euler; updateTform(); }		
		void setPivot ( PivotX& piv )	{ from_pos = piv.from_pos; to_pos = piv.to_pos; ang_euler = piv.ang_euler; updateTform(); }

		void setIdentity ()		{ from_pos.Set(0,0,0); to_pos.Set(0,0,0); ang_euler.Set(0,0,0); scale.Set(1,1,1); trans.Identity(); }

		void setAng ( float rx, float ry, float rz )	{ ang_euler.Set(rx,ry,rz);	updateTform(); }
		void setAng ( Vector3DF& a )					{ ang_euler = a;			updateTform(); }

		void setPos ( float x, float y, float z )		{ from_pos.Set(x,y,z);		updateTform(); }
		void setPos ( Vector3DF& p )					{ from_pos = p;				updateTform(); }

		void setToPos ( float x, float y, float z )		{ to_pos.Set(x,y,z);		updateTform(); }
		
		void updateTform ();
		void setTform ( Matrix4F& t )		{ trans = t; }
		inline Matrix4F& getTform ()		{ return trans; }
		inline float* getTformData ()		{ return trans.GetDataF(); }

		// Pivot		
		PivotX getPivot ()	{ return PivotX(from_pos, to_pos, scale, ang_euler); }
		Vector3DF& getPos ()			{ return from_pos; }
		Vector3DF& getToPos ()			{ return to_pos; }
		Vector3DF& getAng ()			{ return ang_euler; }
		Vector3DF getDir ()			{ 
			return to_pos - from_pos; 
		}

		Vector3DF	from_pos;
		Vector3DF	to_pos;
		Vector3DF	scale;
		Vector3DF	ang_euler;
		Matrix4F	trans;
		
		//Quatern	ang_quat;
		//Quatern	dang_quat;
	};

#endif




// Camera Class
// R. Hoetzlein (c) 2005-2011
// Z-Lib License
// 
// This software is provided 'as-is', without any express or implied
// warranty.  In no event will the authors be held liable for any damages
// arising from the use of this software.
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software
// 2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.
//


#ifndef DEF_CAMERA_3D
	#define	DEF_CAMERA_3D
	
	#define DEG_TO_RAD			(3.141592/180.0)

	class Camera3D : public PivotX {
	public:
		enum eProjection {
			Perspective = 0,
			Parallel = 1
		};
		Camera3D ();

		void draw_gl();

		// Camera settings
		void setAspect ( float asp )					{ mAspect = asp;			updateMatricies(); }
		void setPos ( float x, float y, float z )		{ from_pos.Set(x,y,z);		updateMatricies(); }
		void setToPos ( float x, float y, float z )		{ to_pos.Set(x,y,z);		updateMatricies(); }
		void setFov (float fov)							{ mFov = fov;				updateMatricies(); }
		void setNearFar (float n, float f )				{ mNear = n; mFar = f;		updateMatricies(); }
		void setTile ( float x1, float y1, float x2, float y2 )		{ mTile.Set ( x1, y1, x2, y2 );		updateMatricies(); }
		void setProjection (eProjection proj_type);
		//void setModelMatrix ();
		//void setModelMatrix ( Matrix4F& model );
		
		// Camera motion
		void setOrbit  ( float ax, float ay, float az, Vector3DF tp, float dist, float dolly );
		void setOrbit  ( Vector3DF angs, Vector3DF tp, float dist, float dolly );
		void setAngles ( float ax, float ay, float az );
		void moveOrbit ( float ax, float ay, float az, float dist );		
		void moveToPos ( float tx, float ty, float tz );		
		void moveRelative ( float dx, float dy, float dz );

		// Frustum testing
		bool pointInFrustum ( float x, float y, float z );
		bool boxInFrustum ( Vector3DF bmin, Vector3DF bmax);
		float calculateLOD ( Vector3DF pnt, float minlod, float maxlod, float maxdist );

		// Utility functions
		void updateMatricies ();					// Updates camera axes and projection matricies
		void updateFrustum ();						// Updates frustum planes
		Vector3DF inverseRay ( float x, float y, float z );
		Vector4DF project ( Vector3DF& p );
		Vector4DF project ( Vector3DF& p, Matrix4F& vm );		// Project point - override view matrix

		void getVectors ( Vector3DF& dir, Vector3DF& up, Vector3DF& side )	{ dir = dir_vec; up = up_vec; side = side_vec; }
		void getBounds ( float dst, Vector3DF& min, Vector3DF& max );
		float getNear ()				{ return mNear; }
		float getFar ()					{ return mFar; }
		float getFov ()					{ return mFov; }
		float getDolly()				{ return mDolly; }	
		float getOrbitDist()			{ return mOrbitDist; }
		Vector3DF& getUpDir ()			{ return up_dir; }
		Vector4DF& getTile ()			{ return mTile; }
		Matrix4F& getViewMatrix ()		{ return view_matrix; }
		Matrix4F& getInvView ()			{ return invrot_matrix; }
		Matrix4F& getProjMatrix ()		{ return tileproj_matrix; }	
		Matrix4F& getFullProjMatrix ()	{ return proj_matrix; }
		Matrix4F& getModelMatrix()		{ return model_matrix; }
		Matrix4F& getMVMatrix()			{ return mv_matrix; }
		float getAspect ()				{ return mAspect; }

	public:
		eProjection		mProjType;								// Projection type

		// Camera Parameters									// NOTE: Pivot maintains camera from and orientation
		float			mDolly;									// Camera to distance
		float			mOrbitDist;
		float			mFov, mAspect;							// Camera field-of-view
		float			mNear, mFar;							// Camera frustum planes
		Vector3DF		dir_vec, side_vec, up_vec;				// Camera aux vectors (N, V, and U)
		Vector3DF		up_dir;
		Vector4DF		mTile;
		
		// Transform Matricies
		Matrix4F		rotate_matrix;							// Vr matrix (rotation only)
		Matrix4F		view_matrix;							// V matrix	(rotation + translation)
		Matrix4F		proj_matrix;							// P matrix
		Matrix4F		invrot_matrix;							// Vr^-1 matrix
		Matrix4F		invproj_matrix;
		Matrix4F		tileproj_matrix;						// tiled projection matrix
		Matrix4F		model_matrix;
		Matrix4F		mv_matrix;
		float			frustum[6][4];							// frustum plane equations

		bool			mOps[8];
		int				mWire;
				
	};

#endif


#endif
