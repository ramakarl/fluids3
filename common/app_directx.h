
//----------------------------------------------------------------------------------
// File:   app_directx.h
// Author: Rama Hoetzlein
// Email:  rhoetzlein@nvidia.com
// 
// Copyright (c) 2013 NVIDIA Corporation. All rights reserved.
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
//
//----------------------------------------------------------------------------------

/*!
 *
 * This file provides a DirectX 11 context for rapid prototyping of DX demos.
 * It is intended to give a solution using a single header file and makes calls 
 * to the user application using a GLUT-style interface. 
 * Provided here: DX device, DX context, DX depth stencil/raster/blend states,
 * windows event handling, main loop
 *
 */

#include <windows.h>
#include <windowsx.h>
#include <io.h>
#include <fcntl.h>	
#include <conio.h>
#include <d3d11.h>
#include <dxerr.h>
#include <d3dcompiler.h>

#include <dxgi.h>		// For PerfHUD adapter enumeration
#pragma comment(lib, "dxgi.lib")

#include <string>
#include <vector>

extern void display ();
extern void reshape (int, int);
extern void keyboard_func (unsigned char, int, int);
extern void mouse_click_func (int,int,int,int);
extern void mouse_drag_func (int, int);
extern void mouse_move_func (int, int);
extern void idle_func ();
extern void initialize ();

#define GLUT_UP				0
#define GLUT_DOWN			1
#define GLUT_LEFT_BUTTON	1
#define GLUT_RIGHT_BUTTON	2

float	window_width  = 1024;
float	window_height = 768;

int		mState;	

struct MatrixBuffer 
{
    float m[16];    
}; 

//------------------------------------------------------------------------------------
// Global Variables
//--------------------------------------------------------------------------------------
HINSTANCE               g_hInst = NULL;
HWND                    g_hWnd = NULL;
D3D_DRIVER_TYPE         g_driverType = D3D_DRIVER_TYPE_NULL;
D3D_FEATURE_LEVEL       g_featureLevel = D3D_FEATURE_LEVEL_11_0;
ID3D11Device*           g_pDevice = NULL;
ID3D11DeviceContext*    g_pContext = NULL;
IDXGISwapChain*         g_pSwapChain = NULL;
ID3D11RenderTargetView* g_pRenderTargetView = NULL;
ID3D11Texture2D*		g_pDepthStencil = NULL;
ID3D11DepthStencilState*  g_pDepthStencilState = NULL;
ID3D11DepthStencilState*  g_pDepthOffState = NULL;
ID3D11DepthStencilView*	g_pDepthStencilView = NULL;
ID3D11BlendState*		g_pBlendState = NULL;
ID3D11RasterizerState*	g_pRasterizerState;
ID3D11VertexShader*     g_pVSCurrShader = NULL;		// single draw shader
ID3D11InputLayout*      g_pVSCurrLayout = NULL;
ID3D11PixelShader*      g_pPSCurrShader = NULL;
ID3D11Buffer*           g_pMatrixBuffer[3];
ID3D11SamplerState*		g_pSamplerState = NULL;

FILE*					m_OutCons = 0x0;

//--------------------------------------------------------------------------------------
// Forward declarations
//--------------------------------------------------------------------------------------
HRESULT InitWindow( HINSTANCE hInstance, int nCmdShow );
HRESULT InitDevice();
void CleanupDevice();
LRESULT CALLBACK    WndProc( HWND, UINT, WPARAM, LPARAM );

static std::string utf16ToUTF8( const std::wstring &s )
{
    const int size = ::WideCharToMultiByte( CP_UTF8, 0, s.c_str(), -1, NULL, 0, 0, NULL );
	std::vector<char> buf( size );
    ::WideCharToMultiByte( CP_UTF8, 0, s.c_str(), -1, &buf[0], size, 0, NULL );
	return std::string( &buf[0] );
}

void app_printf ( char* format, ... )
{
	// Note: This is the >only< way to do this. There is no general way to
	// pass on all the arguments from one ellipsis function to another.
	// The function vfprintf was specially designed to allow this.
	va_list argptr;
	va_start (argptr, format);				
	vfprintf ( m_OutCons, format, argptr);			
	va_end (argptr);			
	fflush ( m_OutCons );
}

bool checkHR (HRESULT hr)
{
	if (SUCCEEDED(hr))
		return true;

	// An error occured so find out what it was and output it to the output pane in Vis
	app_printf ( "DX Error: %s, %s\n", DXGetErrorString(hr), DXGetErrorDescription(hr) );		

	return false;
}

int checkSHADER ( HRESULT hr, ID3D10Blob* blob )
{
	if( FAILED( hr ) ) {
		if( blob != NULL ) {
			app_printf ( "%s", (CHAR*) blob->GetBufferPointer() );
			blob->Release();
		}
		// An error occured so find out what it was and output it to the output pane in Vis
		app_printf ( "DX Error: %s, %s\n", DXGetErrorString(hr), DXGetErrorDescription(hr) );		
	}
	return hr;
}


//--------------------------------------------------------------------------------------
// Entry point to the program. Initializes everything and goes into a message processing 
// loop. Idle time is used to render the scene.
//--------------------------------------------------------------------------------------
int WINAPI wWinMain( HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR lpCmdLine, int nCmdShow )
{
    UNREFERENCED_PARAMETER( hPrevInstance );
    UNREFERENCED_PARAMETER( lpCmdLine );

	// Console window for printf
	AllocConsole ();
	long lStdHandle = (long) GetStdHandle( STD_OUTPUT_HANDLE );
	int hConHandle = _open_osfhandle(lStdHandle, _O_TEXT);
	m_OutCons = _fdopen( hConHandle, "w" );

	// Init window
    if( FAILED( InitWindow( hInstance, nCmdShow ) ) )  return 0;

	// Init device
	if( FAILED( InitDevice() ) )					{ CleanupDevice(); return 0; }
    
	// User init
	initialize ();

    // Main message loop
    MSG msg = {0};
    while( WM_QUIT != msg.message ) 
	{ 
		display ();        

		if( PeekMessage( &msg, NULL, 0, 0, PM_REMOVE ) ) {
            TranslateMessage( &msg );
            DispatchMessage( &msg );
        }
        
    }

    CleanupDevice();

    return ( int )msg.wParam;
}

//--------------------------------------------------------------------------------------
// Register class and create window
//--------------------------------------------------------------------------------------
HRESULT InitWindow( HINSTANCE hInstance, int nCmdShow )
{
	const wchar_t *str = L"DirectX 11";
	std::string utf8String = utf16ToUTF8( str );
	LPSTR name = const_cast<char*>(utf8String.c_str());

	const wchar_t *str2 = L"Instanced Tesselation Demo (c) NVIDIA, by R.Hoetzlein - DIRECTX 11";
	std::string utf8String2 = utf16ToUTF8( str2 );
	LPSTR win_name = const_cast<char*>(utf8String2.c_str());
	
    // Register class
    WNDCLASSEX wcex;
    wcex.cbSize = sizeof( WNDCLASSEX );
    wcex.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC | CS_DBLCLKS;
    wcex.lpfnWndProc = WndProc;
    wcex.cbClsExtra = 0;
    wcex.cbWndExtra = 0;
    wcex.hInstance = hInstance;
    wcex.hIcon = 0x0;
    wcex.hCursor = LoadCursor( NULL, IDC_ARROW );
    wcex.hbrBackground = ( HBRUSH )( COLOR_WINDOW + 1 );
    wcex.lpszMenuName = NULL;
    wcex.lpszClassName = name;
    wcex.hIconSm = 0x0;
    if( !RegisterClassEx( &wcex ) )
        return E_FAIL;

    // Create window
    g_hInst = hInstance;
    RECT rc = { 0, 0, window_width, window_height };
    AdjustWindowRect( &rc, WS_OVERLAPPEDWINDOW, FALSE );

	g_hWnd = CreateWindow( name, win_name, WS_OVERLAPPEDWINDOW,
                           CW_USEDEFAULT, CW_USEDEFAULT, rc.right - rc.left, rc.bottom - rc.top, NULL, NULL, hInstance,
                           NULL );
    if( !g_hWnd )
        return E_FAIL;

    ShowWindow( g_hWnd, nCmdShow );

    return S_OK;
}


//--------------------------------------------------------------------------------------
// Called every time the application receives a message
//--------------------------------------------------------------------------------------
LRESULT CALLBACK WndProc( HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam )
{
    PAINTSTRUCT ps;
    HDC hdc;
	
    switch( message )
    {
        case WM_PAINT:
            hdc = BeginPaint( hWnd, &ps );
            EndPaint( hWnd, &ps );
            break;
        case WM_DESTROY:
            PostQuitMessage( 0 );
            break;

		case WM_LBUTTONDOWN: {
			mState = GLUT_DOWN;
			mouse_click_func ( GLUT_LEFT_BUTTON, GLUT_DOWN, GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam) );		// invoke GLUT-style mouse move
			} break;
		case WM_RBUTTONDOWN: {
			mState = GLUT_DOWN;
			mouse_click_func ( GLUT_RIGHT_BUTTON, GLUT_DOWN, GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam) );		// invoke GLUT-style mouse move
			} break;
		case WM_LBUTTONUP: {
			mState = GLUT_UP;
			mouse_click_func ( GLUT_LEFT_BUTTON, GLUT_UP, GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam) );		// invoke GLUT-style mouse move
			} break;
		case WM_RBUTTONUP: {
			mState = GLUT_UP;
			mouse_click_func ( GLUT_RIGHT_BUTTON, GLUT_UP, GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam) );		// invoke GLUT-style mouse move
			} break;

		case WM_MOUSEMOVE: {
			int xpos = GET_X_LPARAM(lParam);
			int ypos = GET_Y_LPARAM(lParam);
			if ( mState == GLUT_DOWN )  mouse_drag_func ( xpos, ypos );// invoke GLUT-style mouse events
			else						mouse_move_func ( xpos, ypos ); 
			} break;

		case WM_KEYDOWN: {
			WPARAM param = wParam;
			char c = MapVirtualKey ( param, MAPVK_VK_TO_CHAR );
			keyboard_func ( c, 0, 0 );  // invoke GLUT-style mouse events
			} break;

        default:
            return DefWindowProc( hWnd, message, wParam, lParam );
    }

    return 0;
}


//--------------------------------------------------------------------------------------
// Create Direct3D device and swap chain
//--------------------------------------------------------------------------------------
HRESULT InitDevice()
{
    HRESULT hr = S_OK;

    RECT rc;
    GetClientRect( g_hWnd, &rc );
    UINT width = rc.right - rc.left;
    UINT height = rc.bottom - rc.top;

    UINT createDeviceFlags = 0;
#ifdef _DEBUG
    createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    D3D_DRIVER_TYPE driverTypes[] =
    {
        D3D_DRIVER_TYPE_HARDWARE,
        D3D_DRIVER_TYPE_WARP,
        D3D_DRIVER_TYPE_REFERENCE
    };
    UINT numDriverTypes = ARRAYSIZE( driverTypes );

    D3D_FEATURE_LEVEL featureLevels[] =
    {
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0
    };
	UINT numFeatureLevels = ARRAYSIZE( featureLevels );

	DEVMODE devMode;
    if (!EnumDisplaySettings(NULL, ENUM_CURRENT_SETTINGS, &devMode))
    {
        MessageBoxA(NULL, "ERROR/n Failed enumerate display settings", "DX11Window::createD3DDevice", MB_OK|MB_SETFOREGROUND|MB_TOPMOST);
        return 0;
    }

    DXGI_SWAP_CHAIN_DESC sd;
    ZeroMemory( &sd, sizeof( sd ) );
    sd.BufferCount = 1;
    sd.BufferDesc.Width = width;
    sd.BufferDesc.Height = height;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = devMode.dmDisplayFrequency;
    sd.BufferDesc.RefreshRate.Denominator = 1;	
	sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = g_hWnd;
    sd.SampleDesc.Count = 1;
    sd.SampleDesc.Quality = 0;
    sd.Windowed = TRUE;

	IDXGIFactory* pDXGIFactory;	
	HRESULT hRes;
	hRes = CreateDXGIFactory (__uuidof(IDXGIFactory), (void**) &pDXGIFactory);
	checkHR ( hRes );

	// Search for a PerfHUD adapter.  
	UINT nAdapter = 0;
	IDXGIAdapter* adapter = NULL;
	IDXGIAdapter* selectedAdapter = NULL;
	D3D_DRIVER_TYPE driverType = D3D_DRIVER_TYPE_HARDWARE;

	
	#ifdef PERF_HUD
		while (pDXGIFactory->EnumAdapters(nAdapter, &adapter) != DXGI_ERROR_NOT_FOUND) {
			if (adapter) {
				DXGI_ADAPTER_DESC adaptDesc;
				if (SUCCEEDED(adapter->GetDesc(&adaptDesc))) {
					const bool isPerfHUD = wcscmp(adaptDesc.Description, L"NVIDIA PerfHUD") == 0;
					// Select the first adapter in normal circumstances or the PerfHUD one if it exists.
					if (nAdapter == 0 || isPerfHUD) selectedAdapter = adapter;
					if (isPerfHUD)	driverType = D3D_DRIVER_TYPE_REFERENCE;
				}
			}
			++nAdapter;
		}	
		hr = D3D11CreateDeviceAndSwapChain( selectedAdapter, driverType, NULL, createDeviceFlags, featureLevels, numFeatureLevels,
												D3D11_SDK_VERSION, &sd, &g_pSwapChain, &g_pDevice, &g_featureLevel, &g_pContext );
		checkHR( hr );
	#else
		for( UINT driverTypeIndex = 0; driverTypeIndex < numDriverTypes; driverTypeIndex++ ) {
			g_driverType = driverTypes[driverTypeIndex];
			hr = D3D11CreateDeviceAndSwapChain( NULL, g_driverType, NULL, createDeviceFlags, featureLevels, numFeatureLevels,
												D3D11_SDK_VERSION, &sd, &g_pSwapChain, &g_pDevice, &g_featureLevel, &g_pContext );
			if( SUCCEEDED( hr ) ) break;
		}
	#endif 
    if( FAILED( hr ) ) return hr;

    // Create a render target view
    ID3D11Texture2D* pBackBuffer = NULL;
    hr = g_pSwapChain->GetBuffer( 0, __uuidof( ID3D11Texture2D ), ( LPVOID* )&pBackBuffer );
    if( FAILED( hr ) )
        return hr;

    hr = g_pDevice->CreateRenderTargetView( pBackBuffer, NULL, &g_pRenderTargetView );
    pBackBuffer->Release();
    if( FAILED( hr ) )
        return hr;

    g_pContext->OMSetRenderTargets( 1, &g_pRenderTargetView, NULL );

	 // Create depth stencil texture 
    D3D11_TEXTURE2D_DESC descDepth; 
    ZeroMemory( &descDepth, sizeof(descDepth) ); 
    descDepth.Width = width; 
    descDepth.Height = height; 
    descDepth.MipLevels = 1; 
    descDepth.ArraySize = 1; 
    descDepth.Format = DXGI_FORMAT_D24_UNORM_S8_UINT; 
    descDepth.SampleDesc.Count = 1; 
    descDepth.SampleDesc.Quality = 0; 
    descDepth.Usage = D3D11_USAGE_DEFAULT; 
    descDepth.BindFlags = D3D11_BIND_DEPTH_STENCIL; 
    descDepth.CPUAccessFlags = 0; 
    descDepth.MiscFlags = 0; 
    checkHR( g_pDevice->CreateTexture2D( &descDepth, NULL, &g_pDepthStencil ) );
 
    // Create the depth stencil view 
    D3D11_DEPTH_STENCIL_VIEW_DESC descDSV; 
    ZeroMemory( &descDSV, sizeof(descDSV) ); 
    descDSV.Format = descDepth.Format; 
    descDSV.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D; 
    descDSV.Texture2D.MipSlice = 0; 
    checkHR( g_pDevice->CreateDepthStencilView( g_pDepthStencil, &descDSV, &g_pDepthStencilView ) );
	D3D11_DEPTH_STENCIL_DESC dsDesc;

	// Depth test parameters
	dsDesc.DepthEnable = true;
	dsDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
	dsDesc.DepthFunc = D3D11_COMPARISON_LESS;
	dsDesc.StencilEnable = true;
	dsDesc.StencilReadMask = 0xFF;
	dsDesc.StencilWriteMask = 0xFF;
	dsDesc.FrontFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
	dsDesc.FrontFace.StencilDepthFailOp = D3D11_STENCIL_OP_INCR;
	dsDesc.FrontFace.StencilPassOp = D3D11_STENCIL_OP_KEEP;
	dsDesc.FrontFace.StencilFunc = D3D11_COMPARISON_ALWAYS;
	dsDesc.BackFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
	dsDesc.BackFace.StencilDepthFailOp = D3D11_STENCIL_OP_DECR;
	dsDesc.BackFace.StencilPassOp = D3D11_STENCIL_OP_KEEP;
	dsDesc.BackFace.StencilFunc = D3D11_COMPARISON_ALWAYS;
	checkHR( g_pDevice->CreateDepthStencilState (&dsDesc, &g_pDepthStencilState) );

	// Depth test parameters - OFF mode
	dsDesc.DepthEnable = false;
	dsDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
	dsDesc.DepthFunc = D3D11_COMPARISON_LESS;
	dsDesc.StencilEnable = true;
	dsDesc.StencilReadMask = 0xFF;
	dsDesc.StencilWriteMask = 0xFF;
	dsDesc.FrontFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
	dsDesc.FrontFace.StencilDepthFailOp = D3D11_STENCIL_OP_INCR;
	dsDesc.FrontFace.StencilPassOp = D3D11_STENCIL_OP_KEEP;
	dsDesc.FrontFace.StencilFunc = D3D11_COMPARISON_ALWAYS;
	dsDesc.BackFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
	dsDesc.BackFace.StencilDepthFailOp = D3D11_STENCIL_OP_DECR;
	dsDesc.BackFace.StencilPassOp = D3D11_STENCIL_OP_KEEP;
	dsDesc.BackFace.StencilFunc = D3D11_COMPARISON_ALWAYS;
	checkHR( g_pDevice->CreateDepthStencilState (&dsDesc, &g_pDepthOffState) );

	g_pContext->OMSetDepthStencilState( g_pDepthStencilState, 1 );

	// Set render target
	g_pContext->OMSetRenderTargets( 1, &g_pRenderTargetView, g_pDepthStencilView ); 

	// Blend state
	D3D11_BLEND_DESC descBS;
	ZeroMemory( &descBS, sizeof(descBS) ); 
	descBS.RenderTarget[0].BlendEnable = TRUE;
	descBS.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
	descBS.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
	descBS.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
	descBS.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    descBS.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
    descBS.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
	descBS.RenderTarget[0].RenderTargetWriteMask = 0x0F;
	checkHR( g_pDevice->CreateBlendState ( &descBS, &g_pBlendState ) );

	float blendFactor[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
	UINT sampleMask   = 0xffffffff;
	g_pContext->OMSetBlendState ( g_pBlendState, blendFactor, sampleMask );

	// Sampler state 
	D3D11_SAMPLER_DESC sampDesc;
	ZeroMemory( &sampDesc, sizeof(sampDesc) );
	sampDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
	sampDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
	sampDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
    sampDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
    sampDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
    sampDesc.MinLOD = 0;
    sampDesc.MaxLOD = D3D11_FLOAT32_MAX;
	checkHR ( g_pDevice->CreateSamplerState ( &sampDesc, &g_pSamplerState) );
	
	// Setup the raster description which will determine how and what polygons will be drawn.
	D3D11_RASTERIZER_DESC rasterDesc;
	ZeroMemory( &rasterDesc, sizeof(rasterDesc) );
	rasterDesc.AntialiasedLineEnable = false;
	rasterDesc.CullMode = D3D11_CULL_NONE;
	rasterDesc.DepthBias = 0;
	rasterDesc.DepthBiasClamp = 0.0f;
	rasterDesc.DepthClipEnable = false;
	rasterDesc.FillMode = D3D11_FILL_SOLID;
	rasterDesc.FrontCounterClockwise = true;
	rasterDesc.MultisampleEnable = false;
	rasterDesc.ScissorEnable = false;
	rasterDesc.SlopeScaledDepthBias = 0.0f;

	// Create the rasterizer state from the description we just filled out.
	checkHR( g_pDevice->CreateRasterizerState(&rasterDesc, &g_pRasterizerState) );	
	
	

    // Setup the viewport
    D3D11_VIEWPORT vp;
    vp.Width = (FLOAT)width;
    vp.Height = (FLOAT)height;
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    vp.TopLeftX = 0;
    vp.TopLeftY = 0;
    g_pContext->RSSetViewports( 1, &vp );

	// Create the transform buffers
	D3D11_BUFFER_DESC bd; 
    ZeroMemory( &bd, sizeof(bd) ); 
    bd.Usage = D3D11_USAGE_DEFAULT; 
    bd.ByteWidth = sizeof(MatrixBuffer); 
    bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER; 
    bd.CPUAccessFlags = 0;
	hr = g_pDevice->CreateBuffer( &bd, NULL, &g_pMatrixBuffer[0] ); if( FAILED( hr ) ) return hr; 
	hr = g_pDevice->CreateBuffer( &bd, NULL, &g_pMatrixBuffer[1] ); if( FAILED( hr ) ) return hr; 
	hr = g_pDevice->CreateBuffer( &bd, NULL, &g_pMatrixBuffer[2] ); if( FAILED( hr ) ) return hr; 

    return S_OK;
}




//--------------------------------------------------------------------------------------
// Clean up the objects we've created
//--------------------------------------------------------------------------------------
void CleanupDevice()
{
    if( g_pContext ) g_pContext->ClearState();

    if( g_pRenderTargetView ) g_pRenderTargetView->Release();
    if( g_pSwapChain )	g_pSwapChain->Release();
    if( g_pContext )	g_pContext->Release();
    if( g_pDevice )	g_pDevice->Release();
}

