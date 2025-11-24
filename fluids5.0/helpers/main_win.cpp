

//---------- MAIN WIN -- We are on Windows platform here

#ifdef USE_NETWORK
    #include "network_system.h"
#endif

#include <windows.h>
#include <windowsx.h>
#include <shellscalingapi.h>		// Windows DPI awareness (4K displays)
#include <d2d1.h>					// Windows DPI awareness (4K displays)

#include <shellapi.h>		        // Open browser in shell

#include <GL/glew.h>
#include <GL/wglew.h>

#include <stdio.h>
#include <fcntl.h>
#include <io.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>

#include "nv_gui.h"

#include "main.h"

// Global singletons
Application* pApp = 0x0;
TexInterface gTex;  

static double sys_frequency;

extern "C" { _declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001; }

// OSWindow
// This structure contains Hardware/Platform specific variables
// that would normally go into main.h, but we want to be OS specific.
// Therefore to keep main.h cross-platform, we have a struct for OS-specific variables

struct OSWindow
{
    OSWindow(Application* app) : m_app(app), _hDC(NULL), _hRC(NULL), _hWnd(NULL), _hWndDummy(NULL),
        _hInstance(NULL), m_visible(true)
    {
    }
    Application*    m_app;
    int             m_screen;
    bool            m_visible; 
    
    HDC             _hDC;
    HGLRC           _hRC;
    HWND            _hWnd;
    HWND            _hWndDummy;
    HINSTANCE       _hInstance;
    LPSTR           _lpCmdLine;
    int             _nCmdShow;
    bool		    _bARBVerbose;
};

//----------------------------------------------------- System functions
// These are global, static, platform-specific functions

static const WORD MAX_CONSOLE_LINES = 500;

void sysVisibleConsole()
{
    int hConHandle;
    long lStdHandle;

    CONSOLE_SCREEN_BUFFER_INFO coninfo;

    FILE* fp;

    // allocate a console for this app
    AllocConsole();

    // set the screen buffer to be big enough to let us scroll text
    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE),
        &coninfo);

    coninfo.dwSize.Y = MAX_CONSOLE_LINES;
    SetConsoleScreenBufferSize(GetStdHandle(STD_OUTPUT_HANDLE),
        coninfo.dwSize);

    // redirect unbuffered STDOUT to the console
    lStdHandle = (long)GetStdHandle(STD_OUTPUT_HANDLE);
    hConHandle = _open_osfhandle(lStdHandle, _O_TEXT);
    fp = _fdopen(hConHandle, "w");

    *stdout = *fp;

    setvbuf(stdout, NULL, _IONBF, 0);
    // redirect unbuffered STDIN to the console
    lStdHandle = (long)GetStdHandle(STD_INPUT_HANDLE);
    hConHandle = _open_osfhandle(lStdHandle, _O_TEXT);
    fp = _fdopen(hConHandle, "r");

    *stdin = *fp;

    setvbuf(stdin, NULL, _IONBF, 0);
    // redirect unbuffered STDERR to the console
    lStdHandle = (long)GetStdHandle(STD_ERROR_HANDLE);
    hConHandle = _open_osfhandle(lStdHandle, _O_TEXT);
    fp = _fdopen(hConHandle, "w");

    *stderr = *fp;

    setvbuf(stderr, NULL, _IONBF, 0);

    // make cout, wcout, cin, wcin, wcerr, cerr, wclog and clog

    // point to console as well

    //ios::sync_with_stdio();
}

double sysGetTime()
{
    LARGE_INTEGER time;
    if (QueryPerformanceCounter(&time)) {
        return (double(time.QuadPart) / sys_frequency);
    }
    return 0;
}

void sysSleep(double seconds)
{
    Sleep(DWORD(seconds * 1000.0));
}

int sysGetKeyMods()
{
    int mods = 0;

    if (GetKeyState(VK_SHIFT) & (1 << 31))
        mods |= KMOD_SHIFT;
    if (GetKeyState(VK_CONTROL) & (1 << 31))
        mods |= KMOD_CONTROL;
    if (GetKeyState(VK_MENU) & (1 << 31))
        mods |= KMOD_ALT;
    if ((GetKeyState(VK_LWIN) | GetKeyState(VK_RWIN)) & (1 << 31))
        mods |= KMOD_SUPER;
    return mods;
}

#define INTERNAL_KEY_INVALID -2

// Translates a Windows key to the corresponding GLFW key
//
int sysTranslateKey(WPARAM wParam, LPARAM lParam)
{
    // Check for numeric keypad keys
    // NOTE: This way we always force "NumLock = ON", which is intentional since
    //       the returned key code should correspond to a physical location.
    if ((HIWORD(lParam) & 0x100) == 0) {
        switch (MapVirtualKey(HIWORD(lParam) & 0xFF, 1)) {
        case VK_INSERT:   return KEY_KP_0;
        case VK_END:      return KEY_KP_1;
        case VK_DOWN:     return KEY_KP_2;
        case VK_NEXT:     return KEY_KP_3;
        case VK_LEFT:     return KEY_KP_4;
        case VK_CLEAR:    return KEY_KP_5;
        case VK_RIGHT:    return KEY_KP_6;
        case VK_HOME:     return KEY_KP_7;
        case VK_UP:       return KEY_KP_8;
        case VK_PRIOR:    return KEY_KP_9;
        case VK_DIVIDE:   return KEY_KP_DIVIDE;
        case VK_MULTIPLY: return KEY_KP_MULTIPLY;
        case VK_SUBTRACT: return KEY_KP_SUBTRACT;
        case VK_ADD:      return KEY_KP_ADD;
        case VK_DELETE:   return KEY_KP_DECIMAL;
        default:          break;
        }
    }

    // Check which key was pressed or released
    switch (wParam)
    {
        // The SHIFT keys require special handling
    case VK_SHIFT:
    {
        // Compare scan code for this key with that of VK_RSHIFT in
        // order to determine which shift key was pressed (left or
        // right)
        const DWORD scancode = MapVirtualKey(VK_RSHIFT, 0);
        if ((DWORD)((lParam & 0x01ff0000) >> 16) == scancode)
            return KEY_RIGHT_SHIFT;

        return KEY_LEFT_SHIFT;
    }

    // The CTRL keys require special handling
    case VK_CONTROL:
    {
        MSG next;
        DWORD time;

        // Is this an extended key (i.e. right key)?
        if (lParam & 0x01000000)
            return KEY_RIGHT_CONTROL;

        // Here is a trick: "Alt Gr" sends LCTRL, then RALT. We only
        // want the RALT message, so we try to see if the next message
        // is a RALT message. In that case, this is a false LCTRL!
        time = GetMessageTime();

        if (PeekMessage(&next, NULL, 0, 0, PM_NOREMOVE))
        {
            if (next.message == WM_KEYDOWN ||
                next.message == WM_SYSKEYDOWN ||
                next.message == WM_KEYUP ||
                next.message == WM_SYSKEYUP)
            {
                if (next.wParam == VK_MENU &&
                    (next.lParam & 0x01000000) &&
                    next.time == time)
                {
                    // Next message is a RALT down message, which
                    // means that this is not a proper LCTRL message
                    return INTERNAL_KEY_INVALID;
                }
            }
        }

        return KEY_LEFT_CONTROL;
    }

    // The ALT keys require special handling
    case VK_MENU:
    {
        // Is this an extended key (i.e. right key)?
        if (lParam & 0x01000000)
            return KEY_RIGHT_ALT;

        return KEY_LEFT_ALT;
    }

    // The ENTER keys require special handling
    case VK_RETURN:
    {
        // Is this an extended key (i.e. right key)?
        if (lParam & 0x01000000)
            return KEY_KP_ENTER;

        return KEY_ENTER;
    }

    // Funcion keys (non-printable keys)
    case VK_ESCAPE:        return KEY_ESCAPE;
    case VK_TAB:           return KEY_TAB;
    case VK_BACK:          return KEY_BACKSPACE;
    case VK_HOME:          return KEY_HOME;
    case VK_END:           return KEY_END;
    case VK_PRIOR:         return KEY_PAGE_UP;
    case VK_NEXT:          return KEY_PAGE_DOWN;
    case VK_INSERT:        return KEY_INSERT;
    case VK_DELETE:        return KEY_DELETE;
    case VK_LEFT:          return KEY_LEFT;
    case VK_UP:            return KEY_UP;
    case VK_RIGHT:         return KEY_RIGHT;
    case VK_DOWN:          return KEY_DOWN;
    case VK_F1:            return KEY_F1;
    case VK_F2:            return KEY_F2;
    case VK_F3:            return KEY_F3;
    case VK_F4:            return KEY_F4;
    case VK_F5:            return KEY_F5;
    case VK_F6:            return KEY_F6;
    case VK_F7:            return KEY_F7;
    case VK_F8:            return KEY_F8;
    case VK_F9:            return KEY_F9;
    case VK_F10:           return KEY_F10;
    case VK_F11:           return KEY_F11;
    case VK_F12:           return KEY_F12;
    case VK_F13:           return KEY_F13;
    case VK_F14:           return KEY_F14;
    case VK_F15:           return KEY_F15;
    case VK_F16:           return KEY_F16;
    case VK_F17:           return KEY_F17;
    case VK_F18:           return KEY_F18;
    case VK_F19:           return KEY_F19;
    case VK_F20:           return KEY_F20;
    case VK_NUMLOCK:       return KEY_NUM_LOCK;
    case VK_CAPITAL:       return KEY_CAPS_LOCK;
    case VK_SNAPSHOT:      return KEY_PRINT_SCREEN;
    case VK_SCROLL:        return KEY_SCROLL_LOCK;
    case VK_PAUSE:         return KEY_PAUSE;
    case VK_LWIN:          return KEY_LEFT_SUPER;
    case VK_RWIN:          return KEY_RIGHT_SUPER;
    case VK_APPS:          return KEY_MENU;

        // Numeric keypad
    case VK_NUMPAD0:       return KEY_KP_0;
    case VK_NUMPAD1:       return KEY_KP_1;
    case VK_NUMPAD2:       return KEY_KP_2;
    case VK_NUMPAD3:       return KEY_KP_3;
    case VK_NUMPAD4:       return KEY_KP_4;
    case VK_NUMPAD5:       return KEY_KP_5;
    case VK_NUMPAD6:       return KEY_KP_6;
    case VK_NUMPAD7:       return KEY_KP_7;
    case VK_NUMPAD8:       return KEY_KP_8;
    case VK_NUMPAD9:       return KEY_KP_9;
    case VK_DIVIDE:        return KEY_KP_DIVIDE;
    case VK_MULTIPLY:      return KEY_KP_MULTIPLY;
    case VK_SUBTRACT:      return KEY_KP_SUBTRACT;
    case VK_ADD:           return KEY_KP_ADD;
    case VK_DECIMAL:       return KEY_KP_DECIMAL;

        // Printable keys are mapped according to US layout
    case VK_SPACE:         return KEY_SPACE;
    case 0x30:             return KEY_0;
    case 0x31:             return KEY_1;
    case 0x32:             return KEY_2;
    case 0x33:             return KEY_3;
    case 0x34:             return KEY_4;
    case 0x35:             return KEY_5;
    case 0x36:             return KEY_6;
    case 0x37:             return KEY_7;
    case 0x38:             return KEY_8;
    case 0x39:             return KEY_9;
    case 0x41:             return KEY_A;
    case 0x42:             return KEY_B;
    case 0x43:             return KEY_C;
    case 0x44:             return KEY_D;
    case 0x45:             return KEY_E;
    case 0x46:             return KEY_F;
    case 0x47:             return KEY_G;
    case 0x48:             return KEY_H;
    case 0x49:             return KEY_I;
    case 0x4A:             return KEY_J;
    case 0x4B:             return KEY_K;
    case 0x4C:             return KEY_L;
    case 0x4D:             return KEY_M;
    case 0x4E:             return KEY_N;
    case 0x4F:             return KEY_O;
    case 0x50:             return KEY_P;
    case 0x51:             return KEY_Q;
    case 0x52:             return KEY_R;
    case 0x53:             return KEY_S;
    case 0x54:             return KEY_T;
    case 0x55:             return KEY_U;
    case 0x56:             return KEY_V;
    case 0x57:             return KEY_W;
    case 0x58:             return KEY_X;
    case 0x59:             return KEY_Y;
    case 0x5A:             return KEY_Z;
    case 0xBD:             return KEY_MINUS;
    case 0xBB:             return KEY_EQUAL;
    case 0xDB:             return KEY_LEFT_BRACKET;
    case 0xDD:             return KEY_RIGHT_BRACKET;
    case 0xDC:             return KEY_BACKSLASH;
    case 0xBA:             return KEY_SEMICOLON;
    case 0xDE:             return KEY_APOSTROPHE;
    case 0xC0:             return KEY_GRAVE_ACCENT;
    case 0xBC:             return KEY_COMMA;
    case 0xBE:             return KEY_PERIOD;
    case 0xBF:             return KEY_SLASH;
    case 0xDF:             return KEY_WORLD_1;
    case 0xE2:             return KEY_WORLD_2;
    default:               break;
    }

    // No matching translation was found
    return KEY_UNKNOWN;
}

//------------------------------------------------ WINDOWS ENTRY POINTS

int main(int argc, char** argv)
{
    HINSTANCE hinstance = GetModuleHandle(NULL);
    
    WinMain(hinstance, NULL, NULL, 1);

    ExitProcess(0);		// also terminates worker threads
    return 0;
}

void ClearMessageQueue ()
{
    MSG msg;
    while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE) > 0) //Clear message queue!
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
}

int WINAPI WinMain(HINSTANCE hInstance,
    HINSTANCE hPrevInstance,
    LPSTR     lpCmdLine,
    int       nCmdShow)
{
    //-- Get timer frequency
    LARGE_INTEGER sysfrequency;
    if (QueryPerformanceFrequency(&sysfrequency))   sys_frequency = (double) sysfrequency.QuadPart;
    else                                            sys_frequency = 1;

    //-- Debugging modes
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG | _CRTDBG_MODE_WNDW);

    //-- Call user startup()
    // This sets the desired window configuration (cflags, opengl version, title, width, height)
    pApp->startup();

    //    pApp->appHandleArgs();

    //-- Start native window
    // This creates a Window, OpenGL context and calls the user app init()
    pApp->appStartWindow(hInstance, lpCmdLine, &nCmdShow, 0x0);

    double timeStart = sysGetTime();
    double timeBegin = sysGetTime();

    // m_display_frame = 0;				// current frame
    // m_golden_frame = GoldenFrame;		// golden frame, frame to capture
    //  char	outpng[1024];				// golden frame, png output filename
    //  sprintf(outpng, "out_%s.png", shortname.c_str());

    //-- Main Windows loop
    MSG msg;
    pApp->m_running = true;  

    while (pApp->m_running) {

        //-- Windows message pump
        while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        //-- Render display
        pApp->appRun();

        //-- screenshots
        //if ( m_display_frame==m_golden_frame ) save_frame ( outpng );
        //m_display_frame++;

        //-- Exit key
        if (pApp->getKeyPress(KEY_ESCAPE)) {
            pApp->setKeyPress ( KEY_ESCAPE, false );    // release it

            if (pApp->m_fullscreen) {
                // Return from fullscreen
                pApp->appRestore();
                
            }
            else {
                // Post close message for proper shutdown
                dbgprintf("ESC pressed.\n");
                PostMessage(pApp->m_win->_hWnd, WM_CLOSE, 0, 0);
            }
        }
    }

    disable_nvgui();

    pApp->appShutdown();

    FreeConsole();        

    //---- Memory leak detection on exit
    #ifdef MEMORY_LEAKS_CHECK
        _CrtDumpMemoryLeaks();
    #endif
    #ifdef DEBUG_HEAP
        _CrtDumpMemoryLeaks();
    #endif

    return EXIT_SUCCESS;
}


LRESULT CALLBACK WinProc (HWND m_hWnd,
    UINT   msg,
    WPARAM wParam,
    LPARAM lParam)
{
    AppEnum btn;

    // if no valid app or win yet, return message control to Windows
    if (pApp == NULL)           return DefWindowProc(m_hWnd, msg, wParam, lParam);            
    if (pApp->m_win == NULL)    return DefWindowProc(m_hWnd, msg, wParam, lParam); 

    // Message handling    
    switch (msg) {
    case WM_ACTIVATE:        
        pApp->m_active = true;
        break;
    case WM_SHOWWINDOW:        
        pApp->m_active = (wParam ? true : false);
        break;
    case WM_PAINT:
        pApp->appPostRedisplay();
        break;
    case WM_KEYDOWN:
    case WM_SYSKEYDOWN: {        
        const int scancode = (lParam >> 16) & 0xff;
        const int key = sysTranslateKey(wParam, lParam);        
        pApp->setMods( sysGetKeyMods() );
        pApp->appSetKeyPress(key, true);
        pApp->keyboard( key, AppEnum::BUTTON_PRESS, pApp->getMods(), pApp->getX(), pApp->getY());
        break;
    }
    case WM_KEYUP:
    case WM_SYSKEYUP: {
        const int scancode = (lParam >> 16) & 0xff;
        const int key = sysTranslateKey(wParam, lParam);                
        pApp->setMods(sysGetKeyMods());
        pApp->appSetKeyPress(key, false);
        pApp->keyboard(key, AppEnum::BUTTON_RELEASE, pApp->getMods(), pApp->getX(), pApp->getY());        
        } break;

    case WM_CHAR:
    case WM_SYSCHAR: {
        unsigned int key = (unsigned int)wParam;
        //if (key < 32 || (key > 126 && key < 160))   break;				
        pApp->appSetKeyPress(key, true);
        pApp->keyboardchar(key, pApp->getMods(), pApp->getX(), pApp->getY());
        } break;

    case WM_MOUSEWHEEL:
        pApp->mousewheel((short)HIWORD(wParam));
        break;
    case WM_LBUTTONDBLCLK: case WM_RBUTTONDBLCLK: case WM_MBUTTONDBLCLK:
        btn = (msg == WM_LBUTTONDBLCLK) ? AppEnum::BUTTON_LEFT : ((msg == WM_RBUTTONDBLCLK) ? AppEnum::BUTTON_RIGHT : AppEnum::BUTTON_MIDDLE);
        pApp->appUpdateMouse(float(GET_X_LPARAM(lParam)), float(GET_Y_LPARAM(lParam)), btn, AppEnum::BUTTON_REPEAT);         
        pApp->mouse ( btn, AppEnum::BUTTON_REPEAT, pApp->getMods(), pApp->getX(), pApp->getY());
        break;
    case WM_LBUTTONDOWN: case WM_RBUTTONDOWN: case WM_MBUTTONDOWN:        
        btn = (msg == WM_LBUTTONDOWN) ? AppEnum::BUTTON_LEFT : ((msg == WM_RBUTTONDOWN) ? AppEnum::BUTTON_RIGHT : AppEnum::BUTTON_MIDDLE);
        pApp->appUpdateMouse(float(GET_X_LPARAM(lParam)), float(GET_Y_LPARAM(lParam)), btn, AppEnum::BUTTON_PRESS);
        pApp->mouse( btn, AppEnum::BUTTON_PRESS, pApp->getMods(), pApp->getX(), pApp->getY());
        break;
    case WM_LBUTTONUP: case WM_RBUTTONUP: case WM_MBUTTONUP:        
        btn = (msg == WM_LBUTTONUP) ? AppEnum::BUTTON_LEFT : ((msg == WM_RBUTTONUP) ? AppEnum::BUTTON_RIGHT : AppEnum::BUTTON_MIDDLE);
        pApp->appUpdateMouse(float(GET_X_LPARAM(lParam)), float(GET_Y_LPARAM(lParam)), btn, AppEnum::BUTTON_RELEASE);
        pApp->mouse( btn, AppEnum::BUTTON_RELEASE, pApp->getMods(), pApp->getX(), pApp->getY());
        pApp->m_mouseButton = AppEnum::BUTTON_NONE;
        break;            
    case WM_MOUSEMOVE: {
        pApp->appUpdateMouse(float(GET_X_LPARAM(lParam)), float(GET_Y_LPARAM(lParam)));                
        pApp->motion( pApp->m_mouseButton, pApp->getX(), pApp->getY(), pApp->getDX(), pApp->getDY());
    } break;
    case WM_SIZE: {
        int sw = LOWORD(lParam), sh = HIWORD(lParam);
        if (sw > 0 && sh > 0) {
            pApp->setWinSz (sw, sh);
            pApp->reshape (sw, sh);
        }
    } break;
    case WM_SYSCOMMAND:
        switch (wParam) {
        case SC_MAXIMIZE: pApp->appSetFullscreen(true);	break;
        }
        break;
    case WM_QUIT: case WM_CLOSE:
        // Request to terminate main loop.
        pApp->m_running = false;      
        break;
    case WM_DESTROY:
        // Proper shutdown. Only WM_DESTROY calls PostQuitMessage
        PostQuitMessage(0);
        break;
    default:
        break;
    }
    return DefWindowProc(m_hWnd, msg, wParam, lParam);
}



static int stringInExtensionString(const char* string, const char* exts)
{
    const GLubyte* extensions = (const GLubyte*)exts;
    const GLubyte* start;
    GLubyte* where;
    GLubyte* terminator;

    // It takes a bit of care to be fool-proof about parsing the
    // OpenGL extensions string. Don't be fooled by sub-strings,
    // etc.
    start = extensions;
    for (;;) {
        where = (GLubyte*)strstr((const char*)start, string);
        if (!where) return GL_FALSE;
        terminator = where + strlen(string);
        if (where == start || *(where - 1) == ' ') {
            if (*terminator == ' ' || *terminator == '\0')
                break;
        }
        start = terminator;
    }

    return GL_TRUE;
}

int sysExtensionSupported(const char* name)
{
    int i;
    GLint count;

    // Check if extension is in the modern OpenGL extensions string list
    // This should be safe to use since GL 3.0 is around for a long time :)
    glGetIntegerv(GL_NUM_EXTENSIONS, &count);

    for (i = 0; i < count; i++) {
        const char* en = (const char*)glGetStringi(GL_EXTENSIONS, i);
        if (!en) return GL_FALSE;
        if (strcmp(en, name) == 0) return GL_TRUE;
    }

    // Check platform specifc gets
    const char* exts = NULL;

    if (WGLEW_ARB_extensions_string) {
        exts = wglGetExtensionsStringARB(pApp->m_win->_hDC);
    }
    if (!exts && WGLEW_EXT_extensions_string) {
        exts = wglGetExtensionsStringEXT();
    }
    if (!exts) return FALSE;
    return stringInExtensionString(name, exts);
}

static void APIENTRY sysOpenGLCallback(GLenum source,
    GLenum type,
    GLuint id,
    GLenum severity,
    GLsizei length,
    const GLchar* message,
    const GLvoid* userParam)
{
    if (pApp == 0x0) return;

    GLenum filter = pApp->m_debugFilter;
    GLenum severitycmp = severity;
    // minor fixup for filtering so notification becomes lowest priority
    if (GL_DEBUG_SEVERITY_NOTIFICATION == filter) {
        filter = GL_DEBUG_SEVERITY_LOW_ARB + 1;
    }
    if (GL_DEBUG_SEVERITY_NOTIFICATION == severitycmp) {
        severitycmp = GL_DEBUG_SEVERITY_LOW_ARB + 1;
    }
    if (!filter || severitycmp <= filter) {
        //static std::map<GLuint, bool> ignoreMap;
        //if(ignoreMap[id] == true)
        //    return;
        char* strSource = "0";
        char* strType = strSource;
        switch (source) {
        case GL_DEBUG_SOURCE_API_ARB:              strSource = "API";         break;
        case GL_DEBUG_SOURCE_WINDOW_SYSTEM_ARB:    strSource = "WINDOWS";     break;
        case GL_DEBUG_SOURCE_SHADER_COMPILER_ARB:  strSource = "SHADER COMP.";       break;
        case GL_DEBUG_SOURCE_THIRD_PARTY_ARB:      strSource = "3RD PARTY";          break;
        case GL_DEBUG_SOURCE_APPLICATION_ARB:      strSource = "APP";         break;
        case GL_DEBUG_SOURCE_OTHER_ARB:            strSource = "OTHER";       break;
        }
        switch (type) {
        case GL_DEBUG_TYPE_ERROR_ARB:               strType = "ERROR";          break;
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR_ARB: strType = "Deprecated";     break;
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR_ARB:  strType = "Undefined";      break;
        case GL_DEBUG_TYPE_PORTABILITY_ARB:         strType = "Portability";    break;
        case GL_DEBUG_TYPE_PERFORMANCE_ARB:         strType = "Performance";    break;
        case GL_DEBUG_TYPE_OTHER_ARB:               strType = "Other";          break;
        }
        dbgprintf("ARB_DEBUG: %s - %s : %s\n", strSource, strType, message);
    }
}


//------------------------------------------------------------ Application

Application::Application() : m_renderCnt(1), m_win(0), m_debugFilter(0)
{
    dbgprintf("Application (constructor)\n");
    pApp = this;        // global handle
    m_mouseX = -1;
    m_mouseY = -1;
    m_mods = 0;
    m_fullscreen = false;
    m_startup = true;
    m_active = false;
    memset(m_keyPressed, 0, sizeof(m_keyPressed));
    memset(m_keyToggled, 0, sizeof(m_keyToggled));
}

// appStart - called from the user function startup() to indicate desired application config
bool Application::appStart(const std::string& title, const std::string& shortname, int width, int height, int Major, int Minor, int MSAA, bool GLDebug )
{
    dbgprintf("appStart");

    bool vsyncstate = true;

    m_winSz[0] = width;             // desired width & height, may not be actual/final
    m_winSz[1] = height;

    m_cflags.major = Major;         // desired OpenGL settings
    m_cflags.minor = Minor;
    m_cflags.MSAA = MSAA;
    m_cflags.debug = GLDebug;

    m_active = false;                // not yet active
    m_title = title;

    return true;
}

bool Application::appStartWindow (void* arg1, void* arg2, void* arg3, void* arg4)
{
    dbgprintf("appStartWindow.\n");

    if (m_win == 0x0) {
        m_win = new OSWindow(this);    // create os-specific variables first time
    }

    //-- Create Window class
    WNDCLASSEX winClass;
    winClass.lpszClassName = "MY_WINDOWS_CLASS";
    winClass.cbSize = sizeof(WNDCLASSEX);
    winClass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC | CS_DBLCLKS;
    winClass.lpfnWndProc = WinProc;
    winClass.hInstance = m_win->_hInstance;
    winClass.hIcon = LoadIcon( m_win->_hInstance, IDI_APPLICATION);
    winClass.hIconSm = LoadIcon( m_win->_hInstance, IDI_APPLICATION);
    winClass.hCursor = LoadCursor(NULL, IDC_ARROW);
    winClass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
    winClass.lpszMenuName = NULL;
    winClass.cbClsExtra = 0;
    winClass.cbWndExtra = 0;

    if (!RegisterClassEx(&winClass))
        return false;

    DWORD style = WS_CLIPSIBLINGS | WS_CLIPCHILDREN | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_SIZEBOX;
    DWORD styleEx = WS_EX_APPWINDOW | WS_EX_WINDOWEDGE;

    RECT rect = { 0, 0, m_winSz[0], m_winSz[1] };       // desired window size
    AdjustWindowRectEx(&rect, style, FALSE, styleEx);

    int sx, sy;

    //-- DPI AWARENESS (4K displays)
    #ifdef USE_DPI_AWARE    
        SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE);

        FLOAT dpiX, dpiY;
        HDC screen = GetDC(0);
        dpiX = static_cast<FLOAT>(GetDeviceCaps(screen, LOGPIXELSX));
        dpiY = static_cast<FLOAT>(GetDeviceCaps(screen, LOGPIXELSY));
        ReleaseDC(0, screen);

        sx = static_cast<UINT>(ceil(rect.right - rect.left * dpiX / 96.f));
        sy = static_cast<UINT>(ceil(rect.bottom - rect.top * dpiY / 96.f));
    #else
        sx = rect.right - rect.left;
        sy = rect.bottom - rect.top;
    #endif 
    //--------------------------------- 
    char title[1024];
    strcpy(title, m_title.c_str());

    //-- Create OS Window
    m_win->_hWnd = CreateWindowEx( styleEx, "MY_WINDOWS_CLASS",
        title, style, 0, 0, sx, sy, NULL, NULL, m_win->_hInstance, (LPVOID)NULL);
    winClass.lpszClassName = "DUMMY";
    winClass.lpfnWndProc = DefWindowProc;
    
    if (!RegisterClassEx(&winClass))
        return false;
    
    m_win->_hWndDummy = CreateWindowEx(NULL, "DUMMY", "Dummy", WS_OVERLAPPEDWINDOW, 0, 0, 10, 10, NULL, NULL,
        m_win->_hInstance, NULL);

    if ( m_win->_hWnd == NULL)
        return false;

    //-- Create OpenGL context
    appCreateGL(&m_cflags, m_winSz[0], m_winSz[1]);
    

    //-- Additional OpenGL initialization
    appInitGL();

    //-- GUI framework
    enable_nvgui();

    //-- User init
    if (m_startup) {                // Call user init() only ONCE per application
        dbgprintf("init()\n");
        if (!init()) { dbgprintf("ERROR: Unable to init() app.\n"); return false; }
    }
    // dbgprintf("activate()\n");        // Call user activate() each time window/surface is recreated
    if (!activate()) { dbgprintf("ERROR: Activate failed.\n"); return false; }
    
    // Show the OS Window
    ShowWindow(m_win->_hWnd, SW_SHOW);
    
//    sysVisibleConsole();

    m_startup = false;
    m_active = true;                // yes, now active.

    // Vsync off by default
    appSwapInterval( 0 );    

    return true;
}

bool Application::appCreateGL(const Application::ContextFlags* cflags, int& width, int& height)
{
    GLuint PixelFormat;

    Application::ContextFlags  settings;
    settings = m_cflags;

    PIXELFORMATDESCRIPTOR pfd;
    memset(&pfd, 0, sizeof(PIXELFORMATDESCRIPTOR));

    pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.cDepthBits = settings.depth;
    pfd.cStencilBits = settings.stencil;

    if (settings.stereo) pfd.dwFlags |= PFD_STEREO;

    // Multisample Anti-Aliasing
    if (settings.MSAA > 1)
    {
        dbgprintf(" Enable Multisample Anti-Aliasing.\n");
        m_win->_hDC = GetDC(m_win->_hWndDummy);
        PixelFormat = ChoosePixelFormat(m_win->_hDC, &pfd);
        SetPixelFormat(m_win->_hDC, PixelFormat, &pfd);
        m_win->_hRC = wglCreateContext(m_win->_hDC);
        wglMakeCurrent(m_win->_hDC, m_win->_hRC);
        glewInit();
        ReleaseDC(m_win->_hWndDummy, m_win->_hDC);
        m_win->_hDC = GetDC(m_win->_hWnd);

        int attri[] = {
            WGL_DRAW_TO_WINDOW_ARB, true,
            WGL_PIXEL_TYPE_ARB, WGL_TYPE_RGBA_ARB,
            WGL_SUPPORT_OPENGL_ARB, true,
            WGL_ACCELERATION_ARB, WGL_FULL_ACCELERATION_ARB,
            WGL_DOUBLE_BUFFER_ARB, true,
            WGL_DEPTH_BITS_ARB, settings.depth,
            WGL_STENCIL_BITS_ARB, settings.stencil,
            WGL_SAMPLE_BUFFERS_ARB, 1,
            WGL_SAMPLES_ARB, settings.MSAA,
            0,0
        };
        GLuint nfmts;
        int fmt;
        if (!wglChoosePixelFormatARB(m_win->_hDC, attri, NULL, 1, &fmt, &nfmts)) {
            wglDeleteContext(m_win->_hRC);
            return false;
        }
        wglDeleteContext(m_win->_hRC);
        DestroyWindow(m_win->_hWndDummy);
        m_win->_hWndDummy = NULL;
        if (!SetPixelFormat(m_win->_hDC, fmt, &pfd))
            return false;

        glEnable(GL_MULTISAMPLE);

    }
    else {
        m_win->_hDC = GetDC(m_win->_hWnd);
        PixelFormat = ChoosePixelFormat(m_win->_hDC, &pfd);
        SetPixelFormat(m_win->_hDC, PixelFormat, &pfd);
    }
    m_win->_hRC = wglCreateContext(m_win->_hDC);
    wglMakeCurrent(m_win->_hDC, m_win->_hRC);

    // calling glewinit NOW because the inside glew, there is mistake to fix...
    // This is the joy of using Core. The query glGetString(GL_EXTENSIONS) is deprecated from the Core profile.
    // You need to use glGetStringi(GL_EXTENSIONS, <index>) instead. Sounds like a "bug" in GLEW.

    glewInit();

#define GLCOMPAT

    if (!wglCreateContextAttribsARB)
        wglCreateContextAttribsARB = (PFNWGLCREATECONTEXTATTRIBSARBPROC)wglGetProcAddress("wglCreateContextAttribsARB");

    if (wglCreateContextAttribsARB) {
        HGLRC hRC = NULL;
        std::vector<int> attribList;
#define ADDATTRIB(a,b) { attribList.push_back(a); attribList.push_back(b); }
        int maj = settings.major;
        int min = settings.minor;
        ADDATTRIB(WGL_CONTEXT_MAJOR_VERSION_ARB, maj)
            ADDATTRIB(WGL_CONTEXT_MINOR_VERSION_ARB, min)
            if (settings.core)
                ADDATTRIB(WGL_CONTEXT_PROFILE_MASK_ARB, WGL_CONTEXT_CORE_PROFILE_BIT_ARB)
            else
                ADDATTRIB(WGL_CONTEXT_PROFILE_MASK_ARB, WGL_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB)
                int ctxtflags = 0;
        if (settings.debug)         ctxtflags |= WGL_CONTEXT_DEBUG_BIT_ARB;
        if (settings.robust)        ctxtflags |= WGL_CONTEXT_ROBUST_ACCESS_BIT_ARB;
        if (settings.forward)       ctxtflags |= WGL_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB;  // use it if you want errors when compat options still used
        ADDATTRIB(WGL_CONTEXT_FLAGS_ARB, ctxtflags);
        ADDATTRIB(0, 0)
            int* p = &(attribList[0]);
        if (!(hRC = wglCreateContextAttribsARB(m_win->_hDC, 0, p)))
        {
            //LOGE("wglCreateContextAttribsARB() failed for OpenGL context.\n");
            return false;
        }
        if (!wglMakeCurrent(m_win->_hDC, hRC)) {
            //LOGE("wglMakeCurrent() failed for OpenGL context.\n"); 
        }
        else {
            wglDeleteContext(m_win->_hRC);
            m_win->_hRC = hRC;
            
            if ( settings.debug ) {
                if (__glewDebugMessageCallbackARB) {
                    __glewDebugMessageCallbackARB = (PFNGLDEBUGMESSAGECALLBACKARBPROC) wglGetProcAddress("glDebugMessageCallbackARB");
                    __glewDebugMessageControlARB = (PFNGLDEBUGMESSAGECONTROLARBPROC) wglGetProcAddress("glDebugMessageControlARB");
                }
                glEnable(GL_DEBUG_OUTPUT);
                glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS_ARB);
                glDebugMessageControlARB(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, NULL, GL_TRUE);
                glDebugMessageCallbackARB( sysOpenGLCallback, 0x0);
            }
        }
    }
    dbgprintf(" Initialize Glew.\n");
    glewInit();
    dbgprintf(" OpenGL Started. Version %s\n", glGetString(GL_VERSION) );

    return true;
}

void Application::appRun()
{
    if (pApp != 0x0) {
        if (pApp->m_active && pApp->m_renderCnt > 0) {
            pApp->m_renderCnt--;
            pApp->display();            // Render user display() function
            appSwapBuffers();           // Swap buffers
        }
    }
}

// update mouse - position only
void Application::appUpdateMouse(float mx, float my){
    m_lastX = m_mouseX;
    m_lastY = m_mouseY;
    m_dX = (m_dX == -1) ? 0 : m_mouseX - mx;
    m_dY = (m_dY == -1) ? 0 : m_mouseY - my;
    m_mouseX = mx;
    m_mouseY = my;
}
// update mouse - with button or state change
void Application::appUpdateMouse(float mx, float my, AppEnum button, AppEnum state)
{
    appUpdateMouse(mx, my);
    if (button != AppEnum::UNDEF) {             // button changed
        m_mouseButton = button;
        m_startX = mx; m_startY = my;
        m_dX = -1; m_dY = -1;        // indicator to ignore the first motion update
    }
    if (state != AppEnum::UNDEF) {              // state changed
        m_mouseState = state;
        m_startX = mx; m_startY = my;
        m_dX = -1; m_dY = -1;
    }
}

bool Application::appStopWindow()
{
    // Cleanup OpenGL
    wglDeleteContext( m_win->_hRC);
    ReleaseDC( m_win->_hWnd, m_win->_hDC);

    // Destroy Window
    DestroyWindow( m_win->_hWnd );

    //UnregisterClass( m_win->_hClass, m_win->_hInstance );

    return true;
}

void Application::appShutdown()
{
    // perform user shutdown() first
    shutdown();

    // destroy Windows & OpenGL surfaces
    appStopWindow();
}


void Application::appForegroundWindow()
{
    Sleep (1000);

    HWND hwnd = m_win->_hWnd;
    HWND hCurWnd = ::GetForegroundWindow();
    DWORD dwMyID = ::GetCurrentThreadId();
    DWORD dwCurID = ::GetWindowThreadProcessId(hCurWnd, NULL);
    ::AttachThreadInput(dwCurID, dwMyID, TRUE);
    ::SetWindowPos( hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE);
    ::SetWindowPos( hwnd, HWND_NOTOPMOST, 0, 0, 0, 0, SWP_SHOWWINDOW | SWP_NOSIZE | SWP_NOMOVE);
    ::SetForegroundWindow( hwnd );
    ::SetFocus( hwnd );
    ::SetActiveWindow( hwnd );
    ::AttachThreadInput(dwCurID, dwMyID, FALSE);
}

void Application::appHandleArgs(int argc, const char** argv)
{
    for (int i = 0; i < argc; i++) {
        if (argv[i][0] == '-') {
            on_arg(argv[i], argv[i + 1]);
            i++;
        }
        if (strcmp(argv[i], "-vsync") == 0 && i + 1 < argc) {
            bool vsync = atoi(argv[i + 1]) ? true : false;
            appSetVSync(vsync);
            i++;
        }
    }
}


/*void NVPWindow::makeContextCurrent()
{
    wglMakeCurrent(m_internal->m_hDC,m_internal->m_hRC);
}
void NVPWindow::makeContextNonCurrent()
{
    wglMakeCurrent(0,0);
}*/

void Application::appOpenKeyboard ()
{
}
void Application::appCloseKeyboard ()
{
}

void Application::appOpenBrowser ( std::string app, std::string query )
{
    // Kill previous browser (process)
	/*if (m_shellprocess != 0) {				
		TerminateProcess(m_shellprocess, 1);	
		CloseHandle(m_shellprocess);
	}*/			

	char appcmd[512];
	strcpy ( appcmd, app.c_str());

	SHELLEXECUTEINFO ShExecInfo = { 0 };
	ShExecInfo.cbSize = sizeof(SHELLEXECUTEINFO);
	ShExecInfo.fMask = SEE_MASK_NOCLOSEPROCESS;
	ShExecInfo.hwnd = NULL;
	ShExecInfo.lpVerb = NULL;
	ShExecInfo.lpFile = appcmd;			// <--- program to run (youtube link)
	ShExecInfo.lpParameters = "";
	ShExecInfo.lpDirectory = NULL;
	ShExecInfo.nShow = SW_SHOW;			
	ShExecInfo.hInstApp = NULL;
	ShellExecuteEx (&ShExecInfo);
	//ShellExecute(0, 0, playlink.c_str(), 0, 0, SW_SHOW);  //-- simple version			

    // m_shellprocess = ShExecInfo.hProcess;		// record the process so we can kill it in future
}

#ifdef USE_NETWORK

    void Application::appSendEventToApp ( Event* e )
    {
        pApp->on_event ( e );
    }
#endif

void Application::appResizeWindow(int w, int h)
{
    RECT rect;
    rect.left = rect.top = 0;
    rect.right = w; rect.bottom = h;
    AdjustWindowRect(&rect, WS_CAPTION, false);
    SetWindowPos(m_win->_hWnd, 0, 0, 0, rect.right - rect.left, rect.bottom - rect.top, SWP_NOMOVE | SWP_NOOWNERZORDER | SWP_NOZORDER);
}

void Application::appSetVSync(bool state)
{
    appSwapInterval(state ? 1 : 0);
    m_vsync = state;
}

void Application::appSwapBuffers()
{
    SwapBuffers(m_win->_hDC);
}

void Application::appSetFullscreen(bool fullscreen)
{
    m_fullscreen = fullscreen;

    LONG style = GetWindowLong(m_win->_hWnd, GWL_STYLE);
    LONG style_ex = GetWindowLong(m_win->_hWnd, GWL_EXSTYLE);

    if (fullscreen) {

        // Save current size for restore
        m_winSz[2] = m_winSz[0];
        m_winSz[3] = m_winSz[1];

        // Fullscreen	  
        SetWindowLong(m_win->_hWnd, GWL_STYLE, style & ~(WS_CAPTION | WS_THICKFRAME));
        SetWindowLong(m_win->_hWnd, GWL_EXSTYLE, style_ex & ~(WS_EX_DLGMODALFRAME | WS_EX_WINDOWEDGE | WS_EX_CLIENTEDGE | WS_EX_STATICEDGE));

        MONITORINFO monitor_info;
        monitor_info.cbSize = sizeof(monitor_info);
        GetMonitorInfo(MonitorFromWindow(m_win->_hWnd, MONITOR_DEFAULTTONEAREST), &monitor_info);
        RECT window_rect(monitor_info.rcMonitor);

        SetWindowPos(m_win->_hWnd, NULL, window_rect.left, window_rect.top,
            window_rect.right - window_rect.left, window_rect.bottom - window_rect.top,
            SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED);
    }
    else {
        // Restore 
        m_winSz[0] = m_winSz[2];
        m_winSz[1] = m_winSz[3];

        SetWindowLong(m_win->_hWnd, GWL_STYLE, style | WS_CAPTION | WS_THICKFRAME);
        SetWindowLong(m_win->_hWnd, GWL_EXSTYLE, style_ex | WS_EX_DLGMODALFRAME | WS_EX_WINDOWEDGE | WS_EX_CLIENTEDGE | WS_EX_STATICEDGE);

        SetWindowPos(m_win->_hWnd, NULL, 20, 20, m_winSz[0], m_winSz[1], SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED);
    }
}

void Application::appMaximize()
{
    ShowWindow(m_win->_hWnd, SW_MAXIMIZE);
}

void Application::appRestore()
{
    ShowWindow(m_win->_hWnd, SW_RESTORE);
    appSetFullscreen(false);
}

void Application::appMinimize()
{
    ShowWindow(m_win->_hWnd, SW_MINIMIZE);
}

bool Application::isActive()
{
    return m_active;
}

void Application::appSwapInterval(int i)
{
    wglSwapIntervalEXT(i);
}


void Application::appSetKeyPress(int key, bool state)
{
    m_keyToggled[key] = (m_keyPressed[key] != state);
    m_keyPressed[key] = state;
}



struct nvVertex {
    nvVertex(float x1, float y1, float z1, float tx1, float ty1, float tz1) { x = x1; y = y1; z = z1; tx = tx1; ty = ty1; tz = tz1; }
    float	x, y, z;
    float	nx, ny, nz;
    float	tx, ty, tz;
};
struct nvFace {
    nvFace(unsigned int x1, unsigned int y1, unsigned int z1) { a = x1; b = y1; c = z1; }
    unsigned int  a, b, c;
};


bool Application::appInitGL()
{
    // additional opengl initialization
    //  (primary init of opengl occurs in WINinteral::initBase)
    initTexGL();     

    glFinish();

    return true;
}



// from file_png.cpp
extern void save_png(char* fname, unsigned char* img, int w, int h, int ch);

void Application::appSaveFrame(char* fname)
{
    int w = getWidth();
    int h = getHeight();

    // Read back pixels
    unsigned char* pixbuf = (unsigned char*)malloc(w * h * 3);

    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, pixbuf);

    // Flip Y
    int pitch = w * 3;
    unsigned char* buf = (unsigned char*)malloc(pitch);
    for (int y = 0; y < h / 2; y++) {
        memcpy(buf, pixbuf + (y * pitch), pitch);
        memcpy(pixbuf + (y * pitch), pixbuf + ((h - y - 1) * pitch), pitch);
        memcpy(pixbuf + ((h - y - 1) * pitch), buf, pitch);
    }

    // Save png
    #ifdef BUILD_PNG
        save_png(fname, pixbuf, w, h, 3);
    #endif

    free(pixbuf);
    free(buf);
}

//-------------------------------------------- Texture interface

//-- screen shader 
static const char* g_tex_vertshader =
"#version 300 es\n"
"#extension GL_ARB_explicit_attrib_location : enable\n"
"layout(location = 0) in vec3 vertex;\n"
"layout(location = 1) in vec3 normal;\n"
"layout(location = 2) in vec3 texcoord;\n"
"uniform vec4 uCoords;\n"
"uniform vec2 uScreen;\n"
"out vec3 vtc;\n"
"void main() {\n"
"   vtc = texcoord * 0.5f + 0.5f;\n"
"   gl_Position = vec4( -1.0f + (uCoords.x/uScreen.x) + (vertex.x+1.0f)*(uCoords.z-uCoords.x)/uScreen.x,\n"
"                       -1.0f + (uCoords.y/uScreen.y) + (vertex.y+1.0f)*(uCoords.w-uCoords.y)/uScreen.y,\n"
"                       0.0f, 1.0f );\n"
"}\n";
static const char* g_tex_fragshader =
"#version 300 es\n"
"  precision mediump float;\n"
"  precision mediump int;\n"			
"uniform sampler2D uTex1;\n"
"uniform sampler2D uTex2;\n"
"uniform int uTexFlags;\n"
"in vec3 vtc;\n"
"layout(location = 0) out vec4 outColor;\n"
"void main() {\n"
"   vec4 op1 = ((uTexFlags & 0x01)==0) ? texture ( uTex1, vtc.xy) : texture ( uTex1, vec2(vtc.x, 1.0f - vtc.y));\n"
"   if ( (uTexFlags & 0x04) != 0 ) {\n"
"		vec4 op2 = ((uTexFlags & 0x02)==0) ? texture ( uTex2, vtc.xy) : texture ( uTex2, vec2(vtc.x, 1.0f - vtc.y));\n"
"		outColor = vec4( op1.xyz * (1.0f - op2.w) + op2.xyz * op2.w, 1.0f );\n"
"   } else { \n"
"		outColor = vec4( op1.xyz, 1.0f );\n"
"   }\n"
"}\n";

#define SHADRS   1

void initTexGL()
{
    int status;
    int maxLog = 65536, lenLog;
    char log[65536];

    // list of internal shaders
    const char** vertcode[SHADRS] = {&g_tex_vertshader};
    const char** fragcode[SHADRS] = {&g_tex_fragshader};

    // load each shader
    for (int n=0; n < SHADRS; n++) {

        // Create a screen-space shader
        gTex.prog[n] = (int) glCreateProgram();
        GLuint vShader = (int) glCreateShader(GL_VERTEX_SHADER);        
        glShaderSource(vShader, 1, (const GLchar**) vertcode[n], NULL);
        glCompileShader(vShader);
        glGetShaderiv(vShader, GL_COMPILE_STATUS, &status);
        if (!status) {
            glGetShaderInfoLog(vShader, maxLog, &lenLog, log);
            dbgprintf("*** Compile Error in init_screenquad vShader\n");
            dbgprintf("  %s\n", log);
        }
        GLuint fShader = (int) glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fShader, 1, (const GLchar**) fragcode[n], NULL);
        glCompileShader(fShader);
        glGetShaderiv(fShader, GL_COMPILE_STATUS, &status);
        if (!status) {
            glGetShaderInfoLog(fShader, maxLog, &lenLog, log);
            dbgprintf("*** Compile Error in init_screenquad fShader\n");
            dbgprintf("  %s\n", log);
        }
        glAttachShader(gTex.prog[n], vShader);
        glAttachShader(gTex.prog[n], fShader);
        glLinkProgram(gTex.prog[n]);    
        glGetProgramiv(gTex.prog[n], GL_LINK_STATUS, &status);
        if (!status) {
            dbgprintf("*** Error! Failed to link in init_screenquad\n");
        }
        checkGL("glLinkProgram (init_screenquad)");

        // Get texture params
        gTex.utex1[n] =     glGetUniformLocation(gTex.prog[n], "uTex1");
        gTex.utex2[n] =     glGetUniformLocation(gTex.prog[n], "uTex2");
        gTex.utexflags[n] = glGetUniformLocation(gTex.prog[n], "uTexFlags");
        gTex.ucoords[n] =   glGetUniformLocation(gTex.prog[n], "uCoords");
        gTex.uscreen[n] =   glGetUniformLocation(gTex.prog[n], "uScreen");        
    }
    glBindVertexArray(0);

    // Create a screen-space quad VBO
    std::vector<nvVertex> verts;
    std::vector<nvFace> faces;
    verts.push_back(nvVertex(-1,-1, 0, -1, 1, 0));
    verts.push_back(nvVertex( 1,-1, 0,  1, 1, 0));
    verts.push_back(nvVertex( 1, 1, 0,  1,-1, 0));
    verts.push_back(nvVertex(-1, 1, 0, -1,-1, 0));
    faces.push_back(nvFace(0, 1, 2));
    faces.push_back(nvFace(2, 3, 0));

    glGenBuffers(1, (GLuint*) &gTex.vbo[0]);
    glGenBuffers(1, (GLuint*) &gTex.vbo[1]);
    checkGL("glGenBuffers (init_screenquad)");
    glGenVertexArrays(1, (GLuint*) &gTex.vbo[2]);
    glBindVertexArray (gTex.vbo[2]);
    checkGL("glGenVertexArrays (init_screenquad)");
    glBindBuffer(GL_ARRAY_BUFFER, gTex.vbo[0]);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(nvVertex), &verts[0].x, GL_STATIC_DRAW_ARB);
    checkGL("glBufferData[V] (init_screenquad)");
    glVertexAttribPointer(0, 3, GL_FLOAT, false, sizeof(nvVertex), 0);			// pos
    glVertexAttribPointer(1, 3, GL_FLOAT, false, sizeof(nvVertex), (void*)12);	// norm
    glVertexAttribPointer(2, 3, GL_FLOAT, false, sizeof(nvVertex), (void*)24);	// texcoord
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gTex.vbo[1]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.size() * 3 * sizeof(int), &faces[0].a, GL_STATIC_DRAW_ARB);
    checkGL("glBufferData[F] (init_screenquad)");
}

void createTexGL(int& glid, int w, int h, int clamp, int fmt, int typ, int filter)
{
    if ( glid != -1) glDeleteTextures(1, (GLuint*) &glid);
    glGenTextures(1, (GLuint*) &glid );
    
    int texDims[2];
    glBindTexture(GL_TEXTURE_2D, glid );
    checkGL("glBindTexture (createTexGL)");
    glGetTexLevelParameteriv( GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &texDims[0] );
    glGetTexLevelParameteriv( GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &texDims[1] );    
    if ( texDims[0]==w && texDims[1]==h ) return;    
    checkGL("getTexLevelParam (createTexGL)");

    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, clamp);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, clamp);
    glTexImage2D(GL_TEXTURE_2D, 0, fmt, w, h, 0, GL_RGBA, typ, 0);
    checkGL("glTexImage2D (createTexGL)");

    glBindTexture(GL_TEXTURE_2D, 0);    
}

void clearGL()
{
    glClearDepth(1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void renderTexGL ( int w, int h, int glid, char inv1 )
{
    renderTexGL( (float)0, (float)0, (float) w, (float) h, glid, inv1);
}
void renderTexGL ( float x1, float y1, float x2, float y2, int glid1, char inv1)
{
    // Prepare pipeline   
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glDepthMask(GL_FALSE);

    // Get viewport dimensions (actual pixels)
    float screen[4];
    glGetFloatv ( GL_VIEWPORT, screen );

    glBindVertexArray( gTex.vbo[2]);                                // Select shader
    checkGL( "glBindVertexArray");    

    int s = 0;      // shader #
    glUseProgram( gTex.prog[s] );
    checkGL( "glUseProgram");    
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);    

    glUniform4f( gTex.ucoords[s], x1, y1, x2, y2);                     // Select texture    
    glUniform2f( gTex.uscreen[s], (float) screen[2], (float) screen[3] );    
    glActiveTexture ( GL_TEXTURE0 );
    glBindTexture ( GL_TEXTURE_2D, glid1 );
    glUniform1i ( gTex.utex1[s], 0 );
    checkGL( "glBindTexture");   
    
    glBindBuffer(GL_ARRAY_BUFFER, gTex.vbo[0]);                     // Select VBO	
    glVertexAttribPointer(0, 3, GL_FLOAT, false, sizeof(nvVertex), 0);
    glVertexAttribPointer(1, 3, GL_FLOAT, false, sizeof(nvVertex), (void*)12);
    glVertexAttribPointer(2, 3, GL_FLOAT, false, sizeof(nvVertex), (void*)24);    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gTex.vbo[1]);
    checkGL( "glBindBuffer");   
 
    int flags = inv1;    
    glUniform1i (gTex.utexflags[s], flags);    // inversion flag

    glDrawElementsInstanced ( GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0, 1);
    
    checkGL("renderTexGL");
    glUseProgram(0);
    glDepthMask(GL_TRUE);
}

void compositeTexGL( float blend, int w, int h, int glid1, int glid2, char inv1, char inv2 )
{
    // Prepare pipeline   
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glDepthMask(GL_FALSE);

    glBindVertexArray( gTex.vbo[2]);                                // Select shader
    checkGL( "glBindVertexArray");    

    int s = 0;  // shader #
    glUseProgram( gTex.prog[s] );
    checkGL( "glUseProgram");    
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);    

    glUniform4f( gTex.ucoords[s], 0, 0, float(w), float(h) );                     // Select texture    
    glUniform2f( gTex.uscreen[s], float(w), float(h) );    

    glActiveTexture ( GL_TEXTURE0 );    
    glBindTexture ( GL_TEXTURE_2D, glid1 );                         // Bind two textures
    glUniform1i ( gTex.utex1[s], 0 );
    glActiveTexture (GL_TEXTURE1);
    glBindTexture (GL_TEXTURE_2D, glid2);
    glUniform1i (gTex.utex2[s], 1);    
    checkGL( "glBindTexture");   
    
    glBindBuffer(GL_ARRAY_BUFFER, gTex.vbo[0]);                     // Select VBO	
    glVertexAttribPointer(0, 3, GL_FLOAT, false, sizeof(nvVertex), 0);
    glVertexAttribPointer(1, 3, GL_FLOAT, false, sizeof(nvVertex), (void*)12);
    glVertexAttribPointer(2, 3, GL_FLOAT, false, sizeof(nvVertex), (void*)24);    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gTex.vbo[1]);
    checkGL( "glBindBuffer");   
 
    int flags = inv1 | (inv2<<1) | 4 ;    
    glUniform1i (gTex.utexflags[s], flags);    // inversion flag

    glDrawElementsInstanced ( GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0, 1);
    
    checkGL( "compositeTexGL");
    glUseProgram(0);
    glDepthMask(GL_TRUE);
   
}
