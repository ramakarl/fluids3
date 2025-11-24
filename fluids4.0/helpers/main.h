

#ifndef __MAIN_H__
	#define __MAIN_H__

    // Application lifecycle:
    //     def. Calls made to the base Application will be defined as 'app functions'.
    //     def. Calls made to the derived user class will be defined as 'user functions'.
    //     def. Calls made from the Java/JNI into a native hook will be defined as 'native functions'.
    //
    // Lifecycle:
    //  1. User instantiates a derived class MyApp inheriting from Application. e.g. class MyApp : public Application {..}
    //  2. Application constructor is called. This sets the global pApp to access the Application abstractly.
    //  3. The OS-platform invokes hooks which start the process. e.g. WinMain or Android onCreate
    //  4. The user function startup() is called, which invoked appStartup with desired configuration variables
    //  5. The OS-platform calls appStartWindow
    //  6. appStartWindow is handed context variables from the OS/Native system. e.g HWND or ANativeWindow, JavaVM, etc.
    //  7. appStartWindow creates a new OSWindow to store these platform-specific globals
    //  8. appStartWindow calls appCreateGL to create an OpenGL context & surface
    //  9. The OpenGL context, surface and display are also stored in the OSWindow variables
    // 10. appStartWindow calls user init() if this is the first invocation
    // 11. appStartWindow calls user activate() to handle repeated changes to the display surface
    // 12. appStartWindow calls enable_nvgui to start the 2D drawing framework
    // 13. The OS-platform OR the Application prepare a main loop
    // 14. appRun is called repeatedly, which makes calls to the user display() to perform draw updates
    // 15. The OS-platform OR the Application handle use & keyboard events, which are passed to appHandleEvent
    // 16. appHandleEvent processes mouse deltas and calls the user mouse() and motion() functions
    // 17. The OS-platform calls addStopWindow if the app is backgrounded, loses focus, or is re-oriented
    // 18. Once the application resumes, the OS-platform will call appStartWindow again.
    // 19. Upon termination appShutdown is called

	#pragma warning(disable:4996) // preventing snprintf >> _snprintf_s

	#include "common_defs.h"	
	#include <stdio.h>
	#include <vector>
	#include <string>
	#include <map>

	// trick for pragma message so we can write:
	// #pragma message(__FILE__"("S__LINE__"): blah")
	#define S__(x) #x
	#define S_(x) S__(x)
	#define S__LINE__ S_(__LINE__)

	#ifdef DEBUG_HEAP
		#define _CRTDBG_MAP_ALLOC  
		#include <stdlib.h>  
		#include <crtdbg.h> 
	#else
		#include <stdlib.h>  
	#endif

	#ifdef WIN32
		#ifdef MEMORY_LEAKS_CHECK
			#   pragma message("build will Check for Memory Leaks!")
			#   define _CRTDBG_MAP_ALLOC
			#   include <stdlib.h>
			#   include <crtdbg.h>
			inline void* operator new(size_t size, const char *file, int line)
			{
			   return ::operator new(size, 1, file, line);
			}

			inline void __cdecl operator delete(void *ptr, const char *file, int line) 
			{
			   ::operator delete(ptr, _NORMAL_BLOCK, file, line);
			}

			#define DEBUG_NEW new( __FILE__, __LINE__)
			#define MALLOC_DBG(x) _malloc_dbg(x, 1, __FILE__, __LINE__);
			#define malloc(x) MALLOC_DBG(x)
			#define new DEBUG_NEW
		#endif
	#endif

	enum AppEnum {
		BUTTON_NONE = 0,					// mouse states
		BUTTON_PRESS = 1,
		BUTTON_RELEASE = 2,
		BUTTON_REPEAT = 3,

		BUTTON_LEFT = 4,					// mouse buttons
		BUTTON_RIGHT = 5,
		BUTTON_MIDDLE = 6,

		ACTION_DOWN = 0,					// mobile actions
		ACTION_MOVE = 1,
		ACTION_UP = 2,
		ACTION_CANCEL = 3,
		GESTURE_SINGLE_TAP = 4,
		GESTURE_DOUBLE_TAP = 5,
		GESTURE_SCALE_BEGIN = 6,
		GESTURE_SCALE = 7,
		GESTURE_SCALE_END = 8,
		ACTION_GLIDE = 9,
		SOFT_KEY_PRESS = 10,

        EVT_XTARGET = 0,
        EVT_YTARGET = 1,
        EVT_XFOCUS = 2,
        EVT_YFOCUS = 3,
        EVT_XSPAN = 4,
        EVT_YSPAN = 5,

		UNDEF = 255
	};
	struct guiEvent {
		int typeOrdinal;
		float xtarget;
		float ytarget;
		float xfocus; // the center of a pinch gesture
		float yfocus;
		float xspan; // the width of a pinch gesture
		float yspan;
	};

	typedef void (*OSProc)(void);
	class OSWindow;							// Forward reference. Described specifically by each platform.
	struct Event;

	//----------------- to be declared in the code of the sample: so the sample can decide how to display messages
	class Application {
	public:
		Application();								// this initializes the global pApp

		// Application-level functions
		// - abstraction virtual, available for app to override
		virtual void startup()  {}
		virtual bool init() { return true; }
        virtual bool activate() { return true; }
		virtual void shutdown() {}
		virtual void reshape(int w, int h) { }
		virtual void on_arg( std::string arg, std::string val ) {}
		virtual void mouse( AppEnum button, AppEnum action, int mods, int x, int y) {}
		virtual void motion( AppEnum button, int x, int y, int dx, int dy) {}
		virtual void mousewheel(int delta) {}
		virtual void keyboard(int keycode, AppEnum action, int mods, int x, int y) {}
		virtual void keyboardchar(unsigned char key, int mods, int x, int y) {}
		virtual void display() {}
		virtual bool begin() { return true; }
		virtual void end() {}
		virtual void checkpoint() {}
	
		#ifdef USE_NETWORK
			virtual void on_event( Event* e )  {}			// Events
			void appSendEventToApp ( Event* e );
		#endif	

		// App Context
		struct ContextFlags {
			int         major, minor, MSAA, depth, stencil;
			bool        debug, robust, core, forward, stereo;
			ContextFlags(int _major=3, int _minor=0, bool _core=true, int _MSAA=1, int _depth=24, int _stencil=8,bool _debug=false, bool _robust=false, bool _forward=false, bool _stereo=false)
			{
				major = _major; minor = _minor; MSAA = _MSAA; depth = _depth; stencil = _stencil; core = _core; debug = _debug;	robust = _robust;
				forward = _forward;	stereo = _stereo;
			}
		};

	  	// Hardware/Platform specific
	  	// - these functions are implemented differently in main_win, main_x11, main_android.cpp
	  	// - functions are listed here generally in order they are called

		bool appStart( const std::string &name, const std::string& shortname, int width, int height, int Major, int Minor, int MSAA=16, bool GLDebug=false );
		void appHandleArgs (int argc, const char** argv);
		bool appStartWindow ( void* arg1=0, void* arg2=0, void* arg3=0, void* arg4=0 );
		bool appCreateGL (const Application::ContextFlags *cflags, int& width, int& height);
		bool appInitGL ();
		void appRun();
		void appSwapBuffers();
		void appPostRedisplay(int n=1) { m_renderCnt=n; }
		void appResizeWindow ( int w, int h );
		void appForegroundWindow();
		void appMaximize();
		void appMinimize();
		void appRestore();
		void appPostQuit();
		bool appStopWindow ();
		void appShutdown ();

		// Input functions
		void appUpdateMouse ( float mx, float my, AppEnum button, AppEnum state);
		void appUpdateMouse ( float mx, float my );
		void appHandleEvent ( guiEvent g );
		void appSetKeyPress ( int key, bool state );
		void appOpenKeyboard ();
		void appCloseKeyboard ();
        void appOpenBrowser ( std::string app, std::string query );

		// Set functions		
		void appSetTitle(const char* title);
		void appSetFullscreen ( bool fullscreen );
		void appSetVSync(bool state);
		void appSwapInterval(int i);
		void appSaveFrame ( char* fname );

		// Accessors
		bool 				isActive();		
		bool 				onPress(int key) 	{ return m_keyPressed[key] && m_keyToggled[key]; }
		inline void         setWinSz(int w, int h) { m_winSz[0]=w; m_winSz[1]=h; }
		inline const int*   getWinSz() const { return m_winSz; }
		inline int          getWidth() const { return m_winSz[0]; }
		inline int          getHeight() const { return m_winSz[1]; }
		inline const int    getWheel() const { return m_wheel; }
		inline int          getMods() const { return m_mods; }
		bool 				getKeyPress(int key) { return m_keyPressed[key]; }
		void				setKeyPress(int key, bool v) { m_keyPressed[key] = v; }
		inline int			getKeyMods();
		inline void         setMods(int m) { m_mods = m; }
		inline float        getX() { return m_mouseX; }
		inline float        getY() { return m_mouseY; }
		inline float        getDX() { return m_dX; }
		inline float        getDY() { return m_dY; }
		inline bool			isFirstFrame()	{ return m_display_frame==0; }
		inline int 			getDisplayFrame() { return m_display_frame; }

	public:

		std::string   	m_title;

		OSWindow*	   	m_win;
		int				m_renderCnt;

		float			m_mouseX, m_mouseY;				// mouse motion
		float 			m_lastX, m_lastY;
		float 			m_spanX, m_spanY;
		float 			m_dX, m_dY;
		float 			m_startX, m_startY;
		AppEnum 		m_mouseButton;
		AppEnum			m_mouseState;
		int				m_wheel;

		int				m_winSz[4];						// window info
		int				m_mods;
		ContextFlags	m_cflags;
		bool			m_doSwap;
		bool            m_startup;
		bool			m_running;
		bool			m_active;
		bool			m_vsync;
		bool			m_fullscreen;

		bool			m_keyPressed[ 400 ];			// keyboard
		bool			m_keyToggled[ 400 ];

		int				m_display_frame;				// frame capture
		int				m_golden_frame;

		unsigned int  	m_debugFilter;
	};

	// External define (for inclusion in other headers)
	//
	extern Application* pApp;

	// Basic OpenGL interface		
	void initTexGL();
	void clearGL ();

	void createTexGL ( int& glid, int w, int h, int clamp=0x812D, int fmt=0x8058, int typ=0x1401, int filter=0x2601 );	// defaults: GL_CLAMP_TO_BORDER, GL_RGBA8, GL_UNSIGNED_BYTE, GL_LINEAR
	void renderTexGL ( int w, int h, int glid, char inv1 = 0);
	void renderTexGL ( float x1, float y1, float x2, float y2, int glid1, char inv1 = 0);
	void compositeTexGL ( float blend, int w, int h, int glid1, int glid2, char inv1 = 0, char inv2 = 0);		// composite two textures	

	struct TexInterface {
		int	prog[3];
		int	vshader[3];
		int	fshader[3];
		int	vbo[3];
		int	utex1[3];
		int	utex2[3];
		int	utexflags[3];
		int	ucoords[3];
		int	uscreen[3];
	};
	extern TexInterface gTex;

	#define	KEY_UNKNOWN     	-1
	#define	KEY_SPACE       	32
	#define KEY_APOSTROPHE      39
	#define KEY_LEFT_PARENTHESIS    40
	#define KEY_RIGHT_PARENTHESIS   41
	#define KEY_ASTERISK            42
	#define KEY_PLUS                43
	#define KEY_COMMA               44
	#define KEY_MINUS               45
	#define KEY_PERIOD              46
	#define KEY_SLASH               47
	#define KEY_0                   48
	#define KEY_1                   49
	#define KEY_2                      50
	#define KEY_3                      51
	#define KEY_4                      52
	#define KEY_5                      53
	#define KEY_6                      54
	#define KEY_7                      55
	#define KEY_8                      56
	#define KEY_9                      57
	#define KEY_COLON                  58
	#define KEY_SEMICOLON              59
	#define KEY_LESS                   60
	#define KEY_EQUAL                  61
	#define KEY_GREATER                62
	#define KEY_A                      65
	#define KEY_B                      66
	#define KEY_C                      67
	#define KEY_D                      68
	#define KEY_E                      69
	#define KEY_F                      70
	#define KEY_G                      71
	#define KEY_H                      72
	#define KEY_I                      73
	#define KEY_J                      74
	#define KEY_K                      75
	#define KEY_L                      76
	#define KEY_M                      77
	#define KEY_N                      78
	#define KEY_O                      79
	#define KEY_P                      80
	#define KEY_Q                      81
	#define KEY_R                      82
	#define KEY_S                      83
	#define KEY_T                      84
	#define KEY_U                      85
	#define KEY_V                      86
	#define KEY_W                      87
	#define KEY_X                      88
	#define KEY_Y                      89
	#define KEY_Z                      90
	#define KEY_LEFT_BRACKET           91
	#define KEY_BACKSLASH              92
	#define KEY_RIGHT_BRACKET          93
	#define KEY_GRAVE_ACCENT       96
	#define KEY_WORLD_1            161
	#define KEY_WORLD_2            162
	#define KEY_ESCAPE             256
	#define KEY_ENTER              257
	#define KEY_TAB                258
	#define KEY_BACKSPACE          259
	#define KEY_INSERT             260
	#define KEY_DELETE             261
	#define KEY_RIGHT              262
	#define KEY_LEFT               263
	#define KEY_DOWN               264
	#define KEY_UP                 265
	#define KEY_PAGE_UP            266
	#define KEY_PAGE_DOWN          267
	#define KEY_HOME               268
	#define KEY_END                269
	#define KEY_CAPS_LOCK          280
	#define KEY_SCROLL_LOCK        281
	#define KEY_NUM_LOCK           282
	#define KEY_PRINT_SCREEN       283
	#define KEY_PAUSE              284
	#define KEY_F1                 290
	#define KEY_F2                 291
	#define KEY_F3                 292
	#define KEY_F4                 293
	#define KEY_F5                 294
	#define KEY_F6                 295
	#define KEY_F7                 296
	#define KEY_F8                 297
	#define KEY_F9                 298
	#define KEY_F10                299
	#define KEY_F11                300
	#define KEY_F12                301
	#define KEY_F13                302
	#define KEY_F14                303
	#define KEY_F15                304
	#define KEY_F16                305
	#define KEY_F17                306
	#define KEY_F18                307
	#define KEY_F19                308
	#define KEY_F20                309
	#define KEY_KP_0               320
	#define KEY_KP_1               321
	#define KEY_KP_2               322
	#define KEY_KP_3               323
	#define KEY_KP_4               324
	#define KEY_KP_5               325
	#define KEY_KP_6               326
	#define KEY_KP_7               327
	#define KEY_KP_8               328
	#define KEY_KP_9               329
	#define KEY_KP_DECIMAL         330
	#define KEY_KP_DIVIDE          331
	#define KEY_KP_MULTIPLY        332
	#define KEY_KP_SUBTRACT        333
	#define KEY_KP_ADD             334
	#define KEY_KP_ENTER          335
	#define KEY_KP_EQUAL          336
	#define KEY_LEFT_SHIFT        340
	#define KEY_LEFT_CONTROL      341
	#define KEY_LEFT_ALT          342
	#define KEY_LEFT_SUPER        343
	#define KEY_RIGHT_SHIFT       344
	#define KEY_RIGHT_CONTROL     345
	#define KEY_RIGHT_ALT         346
	#define KEY_RIGHT_SUPER       347
	#define KEY_MENU              348
	#define KMOD_SHIFT            0x0001
	#define KMOD_CONTROL          0x0002
	#define KMOD_ALT              0x0004
	#define KMOD_SUPER            0x0008

#endif
