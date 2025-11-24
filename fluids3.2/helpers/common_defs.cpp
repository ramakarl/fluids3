
#include "common_defs.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <vector>
#include <string.h>

#if defined(__ANDROID__)
    #include <android/log.h>               // for Android printf logs  
#elif defined(_WIN32)
    #include <windows.h>
    #include <processthreadsapi.h>      // Process memory usage on Win32
    #include <psapi.h>  
#endif

static std::vector<std::string> gPaths = { "./", ASSET_PATH };

static xlong gMemStart = 0;       // amount of used memory when application starts

void dbgprintf(const char * fmt, ...)
{
    va_list  vlist;
    va_start(vlist, fmt);    
    #if defined(__ANDROID__)
        __android_log_vprint(ANDROID_LOG_DEBUG, "NAPP", fmt, vlist );
    #elif defined(__linux__)
        vprintf(fmt, vlist);
    #elif defined(_WIN32)
        vprintf(fmt, vlist);
    #endif
}

char getPathDelim()
{
    #ifdef _WIN32
        return '\\';
    #else
        return '/';
    #endif
}

void addSearchPath ( const char* path )
{
    std::string pathstr = path;
    // every search path must be terminated with a delimiter. add one if needed
    if ( pathstr.at( pathstr.length()-1) != getPathDelim() ) {
        pathstr = pathstr + getPathDelim();
    }
    // add the path
    dbgprintf ( "Added search path: %s\n", pathstr.c_str() );
    gPaths.push_back ( pathstr );
}
bool getFileLocation ( const char* filename, char* outpath )
{
    bool result = getFileLocation ( filename, outpath, gPaths );
    return result;
}
bool getFileLocation ( const char* filename, char* outpath, std::vector<std::string> searchPaths )
{
    bool found = false;
    FILE* fp = fopen( filename, "rb" );
    if (fp) {
        found = true;
        strcpy ( outpath, filename );
    } else {
        for (int i=0; i < searchPaths.size(); i++) {
            if (searchPaths[i].empty() ) continue;
            sprintf ( outpath, "%s%s", searchPaths[i].c_str(), filename );
            fp = fopen( outpath, "rb" );
            if (fp)	{ found = true;	break; }
        }
    }
    if ( found ) fclose ( fp );
    return found;
}

bool getFileLocation ( const std::string filename, std::string& outpath )
{
    char instr[2048];
    char outstr[2048];
    strncpy_sc (instr, filename.c_str(), 2048 );
    bool result = getFileLocation ( instr, outstr, gPaths );
    outpath = outstr;
    return result;
}

unsigned long getFileSize ( const std::string filename )
{
    char instr[2048];
    strncpy_sc (instr, filename.c_str(), 2048) ;
    FILE* fp;
    fp = fopen ( instr, "rb");
    if ( fp==0x0 ) return 0;
    fseek ( fp, 0, SEEK_END );
    unsigned long fsize = ftell ( fp );
    fclose ( fp );

    return fsize;
}
unsigned long getFilePos ( FILE* fp )
{
    return ftell ( fp );
}


void strncpy_sc ( char *dst, const char *src, size_t len)
{
#if defined(__ANDROID__)
    strlcpy (dst, src, len );
#elif defined(__linux__)
    strncpy ( dst, src, len );
#elif defined(_WIN32)
    strncpy ( dst, src, len );
#endif
}

void strncpy_sc (char *dst, size_t dstsz, const char *src, size_t len )
{
 #if defined(__ANDROID__)
    strlcpy (dst, src, len );
#elif defined(__linux__)
    strncpy( dst, src, len );
#elif defined(_WIN32)
    strncpy_s (dst, dstsz, src, len );
#endif
    
    /*//C11 standard
    //src or dest is a null pointer
    //dstsz or count is zero or greater than RSIZE_MAX
    //dstsz is less or equal strnlen_s(src, count), in other words, truncation would occur
    //overlap would occur between the source and the destination strings

    #define RSIZE_MAX INT64_MAX

        bool nullFailed = ( !src || !dst );
        bool sizeFailed = ( dstsz == 0 || dstsz > RSIZE_MAX || count == 0 || count > RSIZE_MAX );
        bool truncFailed = ( dstsz <= strnlen(src, count) );
        bool overlapFailed = false; // TODO - unsure of the test here

        if ( nullFailed || sizeFailed || truncFailed || overlapFailed ) {
            return;
            // TODO - return nice error
        } else {
            strlcpy ( dst, src, count );
        }
    #endif*/
}


#if defined(__ANDROID__)

    #include <EGL/egl.h>
    #include <GLES3/gl3.h>
    void checkGL(const char* msg)
    {
        GLenum errCode = 0;
        errCode = glGetError();
        if ( errCode != GL_NO_ERROR )
            dbgprintf("GL ERROR: %s, code: %x\n", msg, errCode );
    }

#elif defined(__linux__)

    void checkGL(const char* msg) {}
    void checkMem(xlong& total, xlong& used, xlong& app) {}

#elif defined(_WIN32)

    #ifdef USE_OPENGL
        #include <GL/glew.h>
        void checkGL(const char* msg)
        {
            GLenum errCode = 0;
            errCode = glGetError();
            if (errCode != 0 ) {
                std::string err;
                switch ( errCode ) {
                case 0x500:   err = "GL_INVALID_ENUM";    break;
                case 0x501:   err = "GL_INVALID_VALUE";    break;
                case 0x502:   err = "GL_INVALID_OPERATION";    break;
                case 0x503:   err = "GL_STACK_OVERFLOW";    break;
                case 0x504:   err = "GL_STACK_UNDERFLOW";    break;
                case 0x505:   err = "GL_OUT_OF_MEMORY";    break;
                case 0x506:   err = "GL_INVALID_FRAMEBUFFER_OPERATION";    break;
                case 0x507:   err = "GL_CONTEXT_LOST";    break;
                default:  err = "UNKNOWN"; break;
                };
                dbgprintf("GL ERROR: %s, code: %x %s\n", msg, errCode, err.c_str() );
            }
        }
    #else
        void checkGL(const char* msg) {
            dbgprintf( "WARNING: OpenGL is used without checkGL. May need to enable REQUIRED_OPENGL\n" );
        }
    #endif

    void checkMem(xlong& total, xlong& used, xlong& app)
    {
        struct _MEMORYSTATUSEX memx;
        memset(&memx, 0, sizeof(memx));
        memx.dwLength = sizeof(memx);
        
        GlobalMemoryStatusEx(&memx);        
        total = memx.ullTotalPhys;
        used = memx.ullTotalPhys - memx.ullAvailPhys;
        
        PROCESS_MEMORY_COUNTERS pmc;

        BOOL result = GetProcessMemoryInfo( GetCurrentProcess(), &pmc, sizeof(pmc));
        app = 0;
        if ( result )
            app = pmc.WorkingSetSize;
    }      


#endif





