cmake_minimum_required(VERSION 2.8)
set(PROJNAME fluids4.0)
Project(${PROJNAME})
Message(STATUS "-------------------------------")
Message(STATUS "Processing Project ${PROJNAME}:")

####################################################################################
# Bootstrap
#
set( BASE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )
find_path ( HELPERS_DIR "Helpers.cmake" HINTS "${BASE_DIRECTORY}/helpers" )
message ( "BASE: ${BASE_DIRECTORY}" )
message ( "HELPERS: ${HELPERS_DIR}" )
if ( ${HELPERS_DIR} STREQUAL "HELPERS-NOTFOUND" )
    message ( FATAL_ERROR "\n Please set the HELPERS_DIR for cmake helpers." )
endif()
set( CMAKE_MODULE_PATH ${HELPERS_DIR} ) 
include( ${HELPERS_DIR}/Helpers.cmake )  # Cross-Platform functions

#####################################################################################
# Sample requirements

set ( REQUIRE_COMMON "1" )
set ( REQUIRE_TGA "1" )
set ( REQUIRE_VEC "1" )
set ( REQUIRE_CAM "1" )
set ( REQUIRE_NVGUI "1" )
set ( REQUIRE_GLEW "1" )
set ( REQUIRE_OPENGL "1" )
set ( REQUIRE_TIMEX "1" )
set ( REQUIRE_MAIN "1" )

###################
# NSIGHT Markers
set ( USE_NVTX OFF CACHE BOOL "Use NSIGHT Markers" )
if ( USE_NVTX ) 
   add_definitions(-DUSE_NVTX)
endif()

######################
# CUDA Toolkit path
if (NOT CUDA_TOOLKIT_ROOT_DIR) 
   set ( CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2" CACHE PATH "CUDA Toolkit path")
endif()
if ( NOT DEFINED CUDA_ARCH_BIN )
   message( "CUDA_ARCH_BIN not set. Using default. Set this to target GPU architecture.")
   SET( CUDA_ARCH_BIN "compute_50" CACHE STRING "CUDA Architecture target")
   SET( CUDA_ARCH_PTX "sm_50" CACHE STRING "CUDA Code target")
endif ()

####################################################################################
# Find Helpers
#
find_package(Helpers)

#####################################################################################
# Require OpenGL
#
IF(WIN32)
  LIST(APPEND LIBRARIES_OPTIMIZED "opengl32.lib" )
  LIST(APPEND LIBRARIES_DEBUG "opengl32.lib" )
ENDIF()

####################################################################################
# Find CUDA
#
message( STATUS "CUDA:") 
find_package(CUDA)

if ( CUDA_FOUND )
	message( STATUS "  --> Using package CUDA (ver ${CUDA_VERSION})") 
	add_definitions(-DUSECUDA)    
	include_directories(${CUDA_TOOLKIT_INCLUDE})
	LIST(APPEND LIBRARIES_OPTIMIZED ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} )
	LIST(APPEND LIBRARIES_DEBUG ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} )
	LIST(APPEND PACKAGE_SOURCE_FILES ${CUDA_TOOLKIT_INCLUDE} )    
	source_group(CUDA FILES ${CUDA_TOOLKIT_INCLUDE} ) 
else()
	message ( FATAL_ERROR "  ---> Unable to find package CUDA")
endif()

####################################################################################
# Compile PTX Files
#
file(GLOB CUDA_FILES RELATIVE "${BASE_DIRECTORY}" *.cu *.cuh)
message ( STATUS "Build CUDA kernels: ${CUDA_FILES}" )

if ( USE_DEBUG_PTX )
	_COMPILEPTX ( SOURCES ${CUDA_FILES} TARGET_PATH ${EXECUTABLE_OUTPUT_PATH} GENERATED CUDA_PTX GENPATHS CUDA_PTX_PATHS INCLUDE "${CMAKE_CURRENT_SOURCE_DIR},${GVDB_INCLUDE_DIR}" OPTIONS --ptx -arch=${CUDA_ARCH_BIN} -code=${CUDA_ARCH_PTX} -dc -Xptxas -v -G -g --use_fast_math --maxrregcount=32 )
else()
	_COMPILEPTX ( SOURCES ${CUDA_FILES} TARGET_PATH ${EXECUTABLE_OUTPUT_PATH} GENERATED CUDA_PTX GENPATHS CUDA_PTX_PATHS INCLUDE "${CMAKE_CURRENT_SOURCE_DIR},${GVDB_INCLUDE_DIR}" OPTIONS --ptx -arch=${CUDA_ARCH_BIN} -code=${CUDA_ARCH_PTX} -dc -Xptxas -v -O3 --use_fast_math --maxrregcount=32 )
endif()

#####################################################################################
# Source files for this project
#
file(GLOB SOURCE_FILES *.cpp *.c *.h )


########################################################################3
# Collect shaders for reference
file(GLOB GLSL_FILES ${ASSET_PATH}/*.glsl )

#####################################################################################
# Executable
#
unset ( ALL_SOURCE_FILES )
list( APPEND ALL_SOURCE_FILES ${SOURCE_FILES} )
list( APPEND ALL_SOURCE_FILES ${COMMON_SOURCE_FILES} )
list( APPEND ALL_SOURCE_FILES ${PACKAGE_SOURCE_FILES} )
list( APPEND ALL_SOURCE_FILES ${UTIL_SOURCE_FILES} )
list( APPEND ALL_SOURCE_FILES ${CUDA_FILES} )

if ( NOT DEFINED WIN32 )
  find_library(NVTOOLSEXT nvToolsExt HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
  find_library(CUDART cudart HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
  find_library(GVDBLIB gvdb HINTS ${GVDB_LIB_DIR} )
  set(libdeps GL GLEW X11 cuda ${GVDBLIB} ${NVTOOLSEXT} ${CUDART})
  LIST(APPEND LIBRARIES_OPTIMIZED ${libdeps})
  LIST(APPEND LIBRARIES_DEBUG ${libdeps})
ENDIF()

include_directories ("${CMAKE_CURRENT_SOURCE_DIR}")    
add_definitions(-DGVDB_IMPORTS -DGLEW_STATIC -DGLEW_NO_GLU)  
add_definitions(-DASSET_PATH="${ASSET_PATH}/")
add_executable (${PROJNAME} ${ALL_SOURCE_FILES} ${CUDA_FILES} ${PTX_SOURCES} ${GLSL_FILES} )
set_property ( TARGET ${PROJNAME} APPEND PROPERTY DEPENDS ${PTX_SOURCES} )


if ( MSVC  )
    set_target_properties( ${PROJNAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH} )
    set_target_properties( ${PROJNAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${EXECUTABLE_OUTPUT_PATH} )
    set_target_properties( ${PROJNAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${EXECUTABLE_OUTPUT_PATH} )    	
	source_group(CUDA FILES ${CUDA_FILES})
	source_group(PTX FILES  ${PTX_FILES})
	source_group(Helpers FILES ${UTIL_SOURCE_FILES})

	source_group(Core FILES ${SOURCE_FILES})
endif ()

#####################################################################################
# Install kernels
#

_INSTALL_PTX ( FILES ${CUDA_PTX_PATHS} DESTINATION ${EXECUTABLE_OUTPUT_PATH} OUTPUT INSTALL_LIST )

install ( FILES ${CUDA_PTX_PATHS} DESTINATION ${EXECUTABLE_OUTPUT_PATH} )

#####################################################################################
# Library dependencies
#
set_property(GLOBAL PROPERTY DEBUG_CONFIGURATIONS Debug) 

foreach (loop_var IN ITEMS ${LIBRARIES_OPTIMIZED} )   
   target_link_libraries ( ${PROJNAME} optimized ${loop_var} )
endforeach()

foreach (loop_var IN ITEMS ${LIBRARIES_DEBUG} )
   target_link_libraries ( ${PROJNAME} debug ${loop_var} )
endforeach()

message ( STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}" )
message ( STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}" )
message ( STATUS "EXECUTABLE_OUTPUT_PATH: ${EXECUTABLE_OUTPUT_PATH}" )


