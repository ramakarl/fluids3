cmake_minimum_required(VERSION 2.8)
set(PROJNAME fluids3.2)
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
set ( REQUIRE_STRHELPER "1" )
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
   SET( CUDA_ARCH_BIN "compute_35" CACHE STRING "CUDA Architecture target")
   SET( CUDA_ARCH_PTX "compute_35" CACHE STRING "CUDA Code target")
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
	# required for fluids v3.x or older
	add_definitions(-DBUILD_CUDA)	

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
# Compile CU Files to OBJ
#
file(GLOB CUDA_FILES "${BASE_DIRECTORY}/fluids/*.cu" "${BASE_DIRECTORY}/fluids/*.cuh")
message ( STATUS "Build CUDA kernels: ${CUDA_FILES}" )

# Compile CU
set ( CUDA_RUNTIME_OUTPUTS "--device-c -Xcompiler \"/MDd\" " )     # ask nvcc to generate host .obj
if ( USE_DEBUG_PTX )
	_COMPILEPTX ( SOURCES ${CUDA_FILES} TARGET_PATH ${EXECUTABLE_OUTPUT_PATH} OUTEXT obj GENERATED CUDA_PTX GENPATHS CUDA_OBJS_PATHS INCLUDE "${CMAKE_CURRENT_SOURCE_DIR},${GVDB_INCLUDE_DIR}" OPTIONS ${CUDA_RUNTIME_OUTPUTS} -arch=${CUDA_ARCH_BIN} -code=${CUDA_ARCH_PTX} -dc -Xptxas -v -G -g --use_fast_math --maxrregcount=32 )
else()
	_COMPILEPTX ( SOURCES ${CUDA_FILES} TARGET_PATH ${EXECUTABLE_OUTPUT_PATH} OUTEXT obj GENERATED CUDA_PTX GENPATHS CUDA_OBJS_PATHS INCLUDE "${CMAKE_CURRENT_SOURCE_DIR},${GVDB_INCLUDE_DIR}" OPTIONS ${CUDA_RUNTIME_OUTPUTS} -arch=${CUDA_ARCH_BIN} -code=${CUDA_ARCH_PTX} -dc -Xptxas -v -O3 --use_fast_math --maxrregcount=32 )
endif()

#############################################################
#---------- Explicit obj link required for Fluids 3.x or older because of CUDA Runtime SDK
# cuda .cu codes must be built by nvcc because it contains host-side <<<launch>>> code resulting in .obj file
# the ptx source is linked into the obj
#    cu -> obj (nvcc) -> obj (host) -> C/C++ linker
# In Fluids 4.x or newer, only .ptx are generated and loaded dynamically by CUDA Driver SDK (much easier!)
set(CMAKE_VERBOSE_MAKEFILE ON)

set ( CUDA_OUT "${EXECUTABLE_OUTPUT_PATH}/fluid_system_kernels.obj" )
#list(APPEND CUDA_OBJS "${BASE_DIRECTORY}/fluids/fluid_system_kern.obj" )
#list(APPEND CUDA_OBJS "${BASE_DIRECTORY}/fluids/prefix_sum.obj" )
#list(APPEND CUDA_OBJS "${BASE_DIRECTORY}/fluids/radixsort.obj" )
#list(APPEND CUDA_OBJS "${BASE_DIRECTORY}/fluids/radixsort_kernel.obj" )

set ( CUDA_RUNTIME_OUTPUTS "-dlink -arch=${CUDA_ARCH_BIN} -Xcompiler \"/MDd\" ")     # ask nvcc to generate host .obj, relocatible device code, MDd libraries
if ( USE_DEBUG_PTX )
	_LINKPTX ( SRC "null.cpp" INPUTS ${CUDA_OBJS_PATHS} OUTPUT ${CUDA_OUT} INCLUDE "${CMAKE_CURRENT_SOURCE_DIR},${GVDB_INCLUDE_DIR}" OPTIONS "${CUDA_RUNTIME_OUTPUTS}" )
else()
	_LINKPTX ( SRC "null.cpp" INPUTS ${CUDA_OBJS_PATHS} OUTPUT ${CUDA_OUT} INCLUDE "${CMAKE_CURRENT_SOURCE_DIR},${GVDB_INCLUDE_DIR}" OPTIONS "${CUDA_RUNTIME_OUTPUTS}" )
endif()
message ( "CUDA LINK OBJ: ${CUDA_OUT}")
# see below for target_link_libraries command with ${CUDA_OBJ}


#####################################################################################
# Source files for this project
#
file(GLOB SOURCE_FILES *.cpp *.c *.h )
file(GLOB COMMON_FILES common/*.cpp common/*.c common/*.h )
file(GLOB FLUIDS_FILES fluids/*.cpp fluids/*.c fluids/*.h fluids/*.cuh)
include_directories ("${CMAKE_CURRENT_SOURCE_DIR}/common")    
include_directories ("${CMAKE_CURRENT_SOURCE_DIR}/fluids")    

########################################################################3
# Collect shaders for reference
file(GLOB GLSL_FILES ${ASSET_PATH}/*.glsl )

#####################################################################################
# Executable
#
unset ( ALL_SOURCE_FILES )
list( APPEND ALL_SOURCE_FILES ${SOURCE_FILES} )
list( APPEND ALL_SOURCE_FILES ${COMMON_FILES} )
list( APPEND ALL_SOURCE_FILES ${FLUIDS_FILES} )
list( APPEND ALL_SOURCE_FILES ${COMMON_SOURCE_FILES} )
list( APPEND ALL_SOURCE_FILES ${PACKAGE_SOURCE_FILES} )
list( APPEND ALL_SOURCE_FILES ${UTIL_SOURCE_FILES} )

# list( APPEND ALL_SOURCE_FILES ${CUDA_FILES} )

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
add_executable (${PROJNAME} ${ALL_SOURCE_FILES} ${PTX_SOURCES} ${GLSL_FILES} )
set_property ( TARGET ${PROJNAME} APPEND PROPERTY DEPENDS ${PTX_SOURCES} )

if ( MSVC  )
    set_target_properties( ${PROJNAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH} )
    set_target_properties( ${PROJNAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${EXECUTABLE_OUTPUT_PATH} )
    set_target_properties( ${PROJNAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${EXECUTABLE_OUTPUT_PATH} )    	
	source_group(CUDA FILES ${CUDA_FILES})

	source_group(PTX FILES  ${PTX_FILES})
	source_group(Helpers FILES ${UTIL_SOURCE_FILES})

	source_group(Main FILES ${SOURCE_FILES})
	source_group(Common FILES ${COMMON_FILES})
	source_group(Fluids FILES ${FLUIDS_FILES})
endif ()

#--------- CUDA RUN-TIME LINKING
FOREACH( _obj ${CUDA_OBJS_PATHS} )
  get_filename_component( _objname ${_obj} NAME_WE )						# Input base
  add_library ( "OBJ${_objname}" STATIC IMPORTED )
  set_property(TARGET "OBJ${_objname}" PROPERTY IMPORTED_LOCATION "${_obj}" )
  target_link_libraries ( ${PROJECT_NAME} "OBJ${_objname}" )
ENDFOREACH()

add_library ( hostObj STATIC IMPORTED )
set_property(TARGET hostObj PROPERTY IMPORTED_LOCATION "${CUDA_OUT}" )
target_link_libraries ( ${PROJECT_NAME} hostObj )

add_library ( cudartLib SHARED IMPORTED )
set_property(TARGET cudartLib PROPERTY IMPORTED_IMPLIB "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudart_static.lib" )
target_link_libraries ( ${PROJECT_NAME} cudartLib )

add_library ( cudadevLib SHARED IMPORTED )
set_property(TARGET cudadevLib PROPERTY IMPORTED_IMPLIB "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudadevrt.lib" )
target_link_libraries ( ${PROJECT_NAME} cudadevLib )
#
#---------

#####################################################################################
# Install kernels
#
LIST (APPEND FONTS "${BASE_DIRECTORY}/arial.tga" "${BASE_DIRECTORY}/arial.bin" )

_INSTALL_PTX ( FILES ${CUDA_PTX_PATHS} DESTINATION ${EXECUTABLE_OUTPUT_PATH} OUTPUT INSTALL_LIST )
_INSTALL ( FILES ${FONTS}  DESTINATION ${EXECUTABLE_OUTPUT_PATH} OUTPUT INSTALL_LIST)

install ( FILES ${CUDA_PTX_PATHS} DESTINATION ${EXECUTABLE_OUTPUT_PATH} )
install ( FILES ${FONTS} DESTINATION ${EXECUTABLE_OUTPUT_PATH} )

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


