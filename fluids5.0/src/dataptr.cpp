

#include "common_defs.h"
#include "dataptr.h"

#include <GL/glew.h>
#include <cuda.h>
#include <cudaGL.h>

int DataPtr::mFBO = -1;

bool cudaCheck ( CUresult status, char* msg )
{
	if ( status != CUDA_SUCCESS ) {
		const char* stat = "";
		cuGetErrorString ( status, &stat );
		printf ( "CUDA ERROR: %s (in %s)\n", stat, msg  );		
		std::string lin; std::getline ( std::cin, lin );
		exit(-7);
		return false;
	} 
	return true;
}

DataPtr::~DataPtr()
{	
	// must be explicitly freed 
}

void DataPtr::Clear ()
{
  // delete CPU mem
	if ( mCpu != 0x0 ) {  
    free (mCpu);	
    mCpu = 0;
  }
	
  if (mGLID != -1) {
    // delete GL_TEX (and CUDA interop)  
    if (mUseFlags & DT_GLTEX) {
      if (mUseFlags & DT_CUARRAY) { 
        if (mGrsc != 0x0) cuGraphicsUnregisterResource(mGrsc);
        mGrsc = 0; mGpu = 0;
      }
      glDeleteTextures(1, (GLuint*)&mGLID);
    }
    // delete GL_VBO (and CUDA interop)
    if (mUseFlags & DT_GLVBO) {
      if (mUseFlags & DT_CUMEM) {
        if (mGrsc != 0x0) cuGraphicsUnregisterResource(mGrsc);
        mGrsc = 0; mGpu = 0;
      }
    } 
    mGLID = -1;
  }
  // delete CUDA linear memory
  if (mUseFlags & DT_CUMEM && mGpu != 0x0) {
    cuCheck(cuMemFree(mGpu), "DataPtr:Clear", "cuMemFree", "", false);
    mGpu = 0;
  }

	mNum = 0; mMax = 0; mSize = 0;	
}

int DataPtr::getStride ( uchar dtype )
{
	int bpp = dtype;
	switch (dtype) {
	case DT_UCHAR:	bpp = 1; break;
	case DT_UCHAR3:	bpp = 3; break;
	case DT_UCHAR4:	bpp = 4; break;
	case DT_USHORT: bpp = 2; break;
	case DT_UINT:	bpp = 4; break;
	case DT_FLOAT:	bpp = 4; break;
	case DT_FLOAT4:	bpp = 16; break;
	};
	return bpp;
}

void DataPtr::SetUsage ( uchar dt, uchar flags, Vector3DI res )
{
	mUseType = dt;
	if ( flags != DT_MISC ) mUseFlags = flags;
	if ( res.x != -1) mUseRes = res;	
}

void DataPtr::ReallocateCPU ( uint64_t sz )
{
	if ( mSize == sz ) return;
	char* newdata = (char*) malloc ( sz );
	if ( mCpu != 0x0 ) {
		memcpy ( newdata, mCpu, mSize );	
		free ( mCpu );		
	}
	mSize = sz;
	mCpu = newdata;	
}

void DataPtr::Resize ( int stride, uint64_t cnt, char* dat, uchar dest_flags )
{
	Clear();
	Append ( stride, cnt, dat, dest_flags );
}

int DataPtr::Append ( int stride, uint64_t cnt, char* dat, uchar dest_flags )
{ 
	bool mbDebug = false;
	
	if (cnt==0) return 0;

	DataPtr newdat;
	mStride = stride;
	uint64_t data_size = getDataSz ( cnt, stride );
	uint64_t old_size = mSize;
	uint64_t new_size = mSize + data_size;
	newdat.mCpu = 0x0;
	newdat.mGLID = -1;
	newdat.mGpu = 0;
		
	// CPU allocation
	if ( dest_flags & DT_CPU ) {		
		ReallocateCPU ( new_size );		
		if ( dat != 0x0 ) {
			memcpy ( mCpu + old_size, dat, data_size );
		}		
	}
	// Update number
	mMax += cnt;
	mSize = new_size;

	char* src = (dat!=0) ? dat : mCpu;

	// GPU allocation
	if ( dest_flags & DT_GLTEX ) {		
			// OpenGL Texture
			if ( mGLID==-1 ) glGenTextures( 1, (GLuint*) &mGLID );
			checkGL ( "glGenTextures (DataPtr::Append)" );
			glBindTexture ( GL_TEXTURE_2D, mGLID );			
			glPixelStorei ( GL_PACK_ALIGNMENT, 1 );	
			glPixelStorei ( GL_UNPACK_ALIGNMENT, 1 );
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);		
			checkGL ( "glBindTexture (DataPtr::Append)" );
			switch (mUseType) {
			case DT_UCHAR:	glTexImage2D ( GL_TEXTURE_2D, 0, GL_R8,		mUseRes.x, mUseRes.y, 0, GL_RED,	GL_UNSIGNED_BYTE, src );	break;
			case DT_UCHAR3:	glTexImage2D ( GL_TEXTURE_2D, 0, GL_RGB8,	mUseRes.x, mUseRes.y, 0, GL_RGB,	GL_UNSIGNED_BYTE, src );	break;
			case DT_UCHAR4:	glTexImage2D ( GL_TEXTURE_2D, 0, GL_RGBA8,	mUseRes.x, mUseRes.y, 0, GL_RGBA,	GL_UNSIGNED_BYTE, src );	break;
			case DT_FLOAT:	glTexImage2D ( GL_TEXTURE_2D, 0, GL_R32F,	mUseRes.x, mUseRes.y, 0, GL_RED,	GL_FLOAT, src);				break;
			case DT_FLOAT4:	glTexImage2D ( GL_TEXTURE_2D, 0, GL_RGBA32F, mUseRes.x, mUseRes.y, 0, GL_RGBA,	GL_FLOAT, src);				break;
			};
			checkGL ( "glTexImage2D (DataPtr::Append)" );

			// CUDA-GL interop
			if ( dest_flags & DT_CUARRAY ) {
				if ( mGrsc != 0 ) cudaCheck ( cuGraphicsUnregisterResource ( mGrsc ), "cuGraphicsUnregisterResource (DataPtr::Append)" );
				cudaCheck ( cuGraphicsGLRegisterImage ( &mGrsc, mGLID, GL_TEXTURE_3D, CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST ), "cuGraphicsRegisterImage (DataPtr::Append)");
				checkGL ( "cuGraphicsGLRegisterImage (DataPtr::Allocate)" );
				cudaCheck ( cuGraphicsMapResources(1, &mGrsc, 0), "cuGraphicsMapResources (DataPtr::Append)");
				cudaCheck ( cuGraphicsSubResourceGetMappedArray ( &mGarray, mGrsc, 0, 0 ), "cuGraphicsSubResourceGetMappedArray (DataPtr::Append)");
				cudaCheck ( cuGraphicsUnmapResources(1, &mGrsc, 0), "cuGraphicsUnmapResources (DataPtr::Append)");
			}
	} else if ( dest_flags & DT_GLVBO ) {
			// OpenGL VBO
			if ( mGLID == -1 ) glGenBuffers ( 1, (GLuint*) &mGLID );               // GLID = VBO
			glBindBuffer ( GL_ARRAY_BUFFER, mGLID );						
			glBufferData ( GL_ARRAY_BUFFER, new_size, src, GL_STATIC_DRAW );

			// CUDA-GL interop
			if ( dest_flags & DT_CUMEM ) {
				if ( mGrsc != 0 ) cuCheck ( cuGraphicsUnregisterResource ( mGrsc ), "", "cuGraphicsUnregisterResource (DataPtr::Append)", "", false );
				cuCheck ( cuGraphicsGLRegisterBuffer ( &mGrsc, mGLID, CU_GRAPHICS_REGISTER_FLAGS_NONE ), "", "cuGraphicsGLReg", "", false );
				cuCheck ( cuGraphicsMapResources(1, &mGrsc, 0), "", "cuGraphicsMapResrc", "", false );
				size_t sz = 0;
				cuCheck ( cuGraphicsResourceGetMappedPointer ( &mGpu, &sz, mGrsc ),  "", "cuGraphicsResrcGetMappedPtr", "", false );	
			}
	
	} else if ( dest_flags & DT_CUARRAY ) {
			// CUarray using CUDA		

	} else if ( dest_flags & DT_CUMEM ) {	
			// CUDA Linear Memory			
			cudaCheck ( cuMemAlloc ( &newdat.mGpu, new_size ), "cuMemAlloc (DataPtr::Append)" );
			if ( dat != 0x0 ) {
				cudaCheck ( cuMemcpyHtoD ( newdat.mGpu + old_size, src, data_size), "cuMemcpyHtoD (DataPtr::Append)" );
			}
			if ( mGpu != 0x0 ) 
				cudaCheck ( cuMemFree ( mGpu ), "cuMemFree (DataPtr::Append)");			
			mGpu = newdat.mGpu;
	}
	return mSize / mStride;
}

void DataPtr::Commit ()
{
	int sz = mNum * mStride;		// only copy in-use elements

	if ( mUseFlags & DT_GLTEX ) {
			//-- need to save x for this to work
			glBindTexture ( GL_TEXTURE_2D, mGLID );				
			switch (mUseType) {
			case DT_UCHAR:	glTexImage2D ( GL_TEXTURE_2D, 0, GL_R8,		mUseRes.x, mUseRes.y, 0, GL_RED,	GL_UNSIGNED_BYTE, mCpu );	break;
			case DT_UCHAR3:	glTexImage2D ( GL_TEXTURE_2D, 0, GL_RGB8,	mUseRes.x, mUseRes.y, 0, GL_RGB,	GL_UNSIGNED_BYTE, mCpu );	break;
			case DT_UCHAR4:	glTexImage2D ( GL_TEXTURE_2D, 0, GL_RGBA8,	mUseRes.x, mUseRes.y, 0, GL_RGBA,	GL_UNSIGNED_BYTE, mCpu );	break;
			case DT_FLOAT:	glTexImage2D ( GL_TEXTURE_2D, 0, GL_R32F,	mUseRes.x, mUseRes.y, 0, GL_RED,	GL_FLOAT, mCpu);			break;
			case DT_FLOAT4:	glTexImage2D ( GL_TEXTURE_2D, 0, GL_RGBA32F, mUseRes.x, mUseRes.y, 0, GL_RGBA,	GL_FLOAT, mCpu);		break;
			}; 
	} else if ( mUseFlags & DT_GLVBO ) {				
		if ( mUseFlags & DT_CUMEM ) {
			// CUDA-GL Interop
			cudaCheck ( cuMemcpyHtoD ( mGpu, mCpu, sz), "dataptr commit" );
		} else {
			// OpenGL VBO 
			if ( mGLID == -1 ) glGenBuffers ( 1, (GLuint*) &mGLID );  
			glBindBuffer ( GL_ARRAY_BUFFER, mGLID );						
			glBufferData ( GL_ARRAY_BUFFER, sz, mCpu, GL_STATIC_DRAW );		
		}

	} else if ( mUseFlags & DT_CUMEM ) {
		// CUDA Linear memory
		if ( mGpu != 0x0 ) cudaCheck ( cuMemcpyHtoD ( mGpu, mCpu, sz), "dataptr commit" );
	} 
}


void DataPtr::Retrieve ()
{
	if ( mCpu == 0 ) {
		uint64_t new_size = mSize; mSize = 0;		// trick reallocate 
		ReallocateCPU ( new_size );					// ensure space exists on cpu
	}

	if ( mUseFlags & DT_GLTEX ) {

		int w, h;
		glBindTexture(GL_TEXTURE_2D, mGLID );		
		glGetTexLevelParameteriv( GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &w );
		glGetTexLevelParameteriv( GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &h ); 

		if (mFBO==-1) glGenFramebuffers (1, (GLuint*) &mFBO);
		glBindFramebuffer ( GL_FRAMEBUFFER, mFBO);
		glFramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, mGLID, 0);
		GLenum status = glCheckFramebufferStatus (GL_FRAMEBUFFER);
		if (status != GL_FRAMEBUFFER_COMPLETE) 	{
			dbgprintf ( "ERROR: Binding frame buffer for texture read pixels.\n" );
			return;
		}	

		switch (mUseType) {
		case DT_UCHAR:	glReadPixels(0, 0, w, h, GL_RED, GL_UNSIGNED_BYTE, mCpu );		break;
		case DT_UCHAR3:	glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, mCpu );		break;
		case DT_UCHAR4:	glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, mCpu );		break;
		case DT_USHORT: glReadPixels(0, 0, w, h, GL_RED, GL_UNSIGNED_SHORT, mCpu );		break;
		case DT_FLOAT:	glReadPixels(0, 0, w, h, GL_RED,  GL_FLOAT, mCpu );				break;
		case DT_FLOAT4:	glReadPixels(0, 0, w, h, GL_RGBA, GL_FLOAT, mCpu );				break;
		}; 

		// Flip Y
		/*int pitch = w * 3;
		unsigned char* buf = (unsigned char*)malloc(pitch);
		for (int y = 0; y < h / 2; y++) {
			memcpy(buf, pixbuf + (y * pitch), pitch);
			memcpy(pixbuf + (y * pitch), pixbuf + ((h - y - 1) * pitch), pitch);
			memcpy(pixbuf + ((h - y - 1) * pitch), buf, pitch);
		}*/

	} else if ( mUseFlags & DT_GLVBO ) {
		// assuming interop is enables
		cuCheck ( cuMemcpyDtoH ( mCpu, mGpu, mSize ), "DataPtr::Retrieve", "cuMemcpyDtoH", "DT_GLVBO", false );
	
	} else if ( mUseFlags & DT_CUMEM ) {
		cuCheck ( cuMemcpyDtoH ( mCpu, mGpu, mSize ), "DataPtr::Retrieve", "cuMemcpyDtoH", "DT_CUMEM", false );		
	}
}

void DataPtr::CopyTo ( DataPtr* dest, uchar dest_flags )
{
	if ( mSize != dest->mSize ) {
		dbgprintf ( "ERROR: CopyTo sizes don't match.\n" );
		exit(-11);
	}	
	if ( (mUseFlags & DT_CPU) && (dest_flags & DT_CPU) )
		memcpy ( dest->mCpu, mCpu, mSize );

	if ( (mUseFlags & DT_CUMEM) && (dest_flags & DT_CUMEM) )	
		cuCheck ( cuMemcpyDtoD ( dest->mGpu, mGpu, mSize), "DataPtr::CopyTo", "cuMemcpyDtoD", "", false );		// also covers cuda-interop cases

	//if ( (mUseFlags & DT_GLVBO) && (dest_flags & DT_GLVBO) )

}


