

#ifndef DEF_COMMON
	#define DEF_COMMON

	#pragma warning ( disable: 4005)

	#ifdef _WIN32

		#define WIN32_LEAN_AND_MEAN
		#include <windows.h>
		#undef WIN32_LEAN_AND_MEAN

		#pragma warning ( disable : 4800 )			// cast to bool performance warning
		#pragma warning ( disable : 4996 )			// fopen_s, strcpy_s (not linux compatible)
		#pragma warning ( disable : 4244 )			// conversion from double to float
		#pragma warning ( disable : 4305 )			// truncation from double to float (constants)

		#include "inttypes.h"

        #define ALIGN(x)			__declspec(align(x))
        #define CACHE_ALIGNED       __declspec(align(64))

		typedef signed char			sint8_t;
		typedef signed short		sint16_t;
		typedef signed int			sint32_t;
		typedef signed long			sint64_t;

        typedef	unsigned char		        uchar;
		typedef uint64_t			xlong;
		typedef uint8_t				XCHAR;
		typedef uint8_t				XBYTE;
		typedef uint16_t			XBYTE2;
		typedef uint32_t			XBYTE4;
		typedef uint64_t			XBYTE8;
		typedef sint8_t				schar;
		typedef sint16_t			sshort;
		typedef sint32_t			sint;
		typedef sint64_t			slong;
		typedef	uint8_t				uchar;
		typedef uint16_t			ushort;
		typedef uint32_t			uint;
		typedef uint64_t			uxlong;     // note: keyword 'ulong' cannot be used with NV_ARM. 'slong' is signed, dont use here

		#define FALSE	0
		#define TRUE	1

		// DWORD included from windows.h (32-bit unsigned int)

      #else   // ANDOID and linux

            #define ALIGN(x)		__attribute__ ((aligned(x)))
            #define CACHE_ALIGNED   __attribute__ ((aligned(64)))

            #include "inttypes.h"

            // typedef __s64				xlong;
            typedef unsigned long long	xlong;
            typedef unsigned char		XCHAR;
            typedef unsigned char		XBYTE;	  // 8-bit
            typedef unsigned short		XBYTE2;	  // 16-bit
            typedef unsigned long		XBYTE4;   // 32-bit
            typedef long long	      	XBYTE8;	  // 64-bit
            typedef XBYTE4		      	DWORD;

            #define FALSE	0
            #define TRUE	1

            typedef uint8_t			    uchar;
            typedef uint16_t		    ushort;
            typedef uint32_t		    uint;
            typedef uint64_t		    uxlong;

            typedef int8_t			    schar;
            typedef int16_t			    sshort;
            typedef int32_t			    sint;
            typedef int64_t			    slong;

            // avoids Clang warnings
            #define __cdecl
            #define __stdcall

            #include <stdarg.h>  // for va_start, va_args

	#endif

    // universal functions
    #include <vector>
    #include <string>
    void checkGL( const char* msg );
    void checkMem( xlong& total, xlong& used, xlong& app);    
    char getPathDelim();
    void addSearchPath ( const char* path );
    bool getFileLocation ( const char* filename, char* outpath );
    bool getFileLocation ( const char* filename, char* outpath, std::vector<std::string> paths );
    bool getFileLocation ( const std::string filename, std::string &outpath );
    unsigned long getFileSize ( const std::string filename );
    unsigned long getFilePos ( FILE* fp );
    void dbgprintf(const char * fmt, ...);

    #ifdef USE_CUDA
        #include <cuda.h>    
        #define DEV_FIRST		-1
        #define DEV_CURRENT		-2
        #define DEV_EXISTING	-3

        bool cuCheck (CUresult launch_stat, char* method, char* apicall, char* arg, bool bDebug);
        void cuStart ( int devsel, CUcontext ctxsel, CUdevice& dev, CUcontext& ctx, CUstream* strm, bool verbose);
        void cuGetMemUsage ( int& total, int& used, int& free );
    #endif

    void strncpy_sc ( char *dst, const char *src, size_t len);                      // cross-platform
    void strncpy_sc (char *dst, size_t dstsz, const char *src, size_t count );      // cross-platform

    // macros
	#ifndef imax
    	#define imax(a,b) (((a) > (b)) ? (a) : (b))
	#endif
	#ifndef imin
	    #define imin(a,b) (((a) < (b)) ? (a) : (b))
	#endif

	typedef uint32_t			CLRVAL;
    #ifndef COLOR
	    #define COLOR(r,g,b)	( (uint(r*255.0f)<<24) | (uint(g*255.0f)<<16) | (uint(b*255.0f)<<8) )
	#endif
    #ifndef COLORA
	    #define COLORA(r,g,b,a)	( (uint(a*255.0f)<<24) | (uint(b*255.0f)<<16) | (uint(g*255.0f)<<8) | uint(r*255.0f) )
    #endif
	#define ALPH(c)			(float((c>>24) & 0xFF)/255.0)
	#define BLUE(c)			(float((c>>16) & 0xFF)/255.0)
	#define GRN(c)			(float((c>>8)  & 0xFF)/255.0)
	#define RED(c)			(float( c      & 0xFF)/255.0)

    #ifndef CLRVEC
	    #define CLRVEC(c)			( Vector4DF( RED(c), GRN(c), BLUE(c), ALPH(c) ) )
    #endif
    #ifndef VECCLR
	    #define VECCLR(v)			( COLORA( v.x, v.y, v.z, v.w ) )
    #endif

	// Math defs
    #ifndef PI
        #define PI					(3.14159265358979f)			// sometimes useful :)
    #endif
    #ifndef DEGtoRAD
	    #define DEGtoRAD			(3.14159265358979f/180.0f)
    #endif
    #ifndef RADtoDEG
	    #define RADtoDEG			(180.0f/3.14159265358979f)
    #endif

#endif
