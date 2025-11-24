
#ifndef DEF_DATAPTR_H
	#define DEF_DATAPTR_H

	#include "common_defs.h"
	#include <assert.h>
	#include <vector>
	#include <string>		
	#include "vec.h"	
	#include "cuda.h"
	
	#define PUSH_CTX		cuCtxPushCurrent(cuCtx);
	#define POP_CTX			CUcontext pctx; cuCtxPopCurrent(&pctx);

	extern bool cudaCheck ( CUresult status, char* msg );

	#define DT_MISC			0
	#define DT_UCHAR		1
	#define DT_UCHAR3		3
	#define DT_UCHAR4		4
	#define DT_USHORT		5
	#define DT_UINT			6
	#define DT_INT			7
	#define DT_FLOAT		8
	#define DT_FLOAT3		12
	#define DT_FLOAT4		16	
	
	#define DT_CPU			1		// use flags
	#define DT_CUMEM		2
	#define DT_CUARRAY		4
	#define DT_GLTEX		8	
	#define DT_GLVBO		16

	class DataPtr {
	public:
		DataPtr() { mNum=0; mMax=0; mStride=0; mUseRes.Set(0,0,0); mUseType=DT_MISC; mUseFlags=DT_MISC; mSize=0; mCpu=0; mGpu=0; mGLID=-1; mGrsc=0; mGarray=0; }
		~DataPtr();

		void			Resize ( int stride, uint64_t cnt, char* dat=0x0, uchar dest_flags=DT_CPU );
		int				Append ( int stride, uint64_t cnt, char* dat=0x0, uchar dest_flags=DT_CPU );
		void			SetUsage ( uchar dt, uchar flags=DT_MISC, Vector3DI res = Vector3DI(-1,-1,-1) );		// special usage (2D,3D,GLtex,GLvbo,etc.)
		void			ReallocateCPU ( uint64_t sz );		
		void			CopyTo ( DataPtr* dest, uchar dest_flags );
		void			Commit ();		
		void			Retrieve ();		
		void			Clear ();

		int				getStride ( uchar dtype );
		uint64_t		getDataSz ( int cnt, int stride )	{ return (uint64_t) cnt * stride; }

		int				getNum()	{ return mNum; }
		int				getMax()	{ return mMax; }
		char*			getData()	{ return mCpu; }
		CUdeviceptr		getGPU()	{ return mGpu; }
		char*			getPtr(int n)	{ return mCpu + n*mStride; }

	public:
		uint64_t		mNum, mMax, mSize;
		int				mStride;
		uchar			mRefID, mUseType, mUseFlags;	// usage
		Vector3DI		mUseRes;		
		bool			bCpu, bGpu;
		char*			mCpu;				
		CUdeviceptr		mGpu;

		int				mGLID;			// CUDA-GL interop		
		CUgraphicsResource	mGrsc;
		CUarray			mGarray;		

		static int		mFBO;
	};

#endif




