
#
# Find TinyXML
#

unset(TINYXML_FOUND CACHE)
unset(TINYXML_INCLUDE_DIR CACHE)

if ( NOT DEFINED TINYXML_ROOT_DIR )
  if (WIN32)
    get_filename_component ( BASEDIR "${CMAKE_MODULE_PATH}/../../libraries/combined" REALPATH )
  else()
    get_filename_component ( BASEDIR "/usr/local/TINYXML/" REALPATH )
  endif()
  set ( TINYXML_ROOT_DIR ${BASEDIR} CACHE PATH "Location of TINYXML library" FORCE)
endif()
message ( STATUS "Searching for TINYXML at.. ${TINYXML_ROOT_DIR}")
set( TINYXML_FOUND "YES" )

if ( TINYXML_ROOT_DIR )

    #-- Paths to Port Audio
	set ( TINYXML_INCLUDE_DIR "${TINYXML_ROOT_DIR}/include" CACHE PATH "Path to include files" FORCE)
    if (WIN32)
       # Windows sub-path to libs
	   set ( TINYXML_LIB_DIR "${TINYXML_ROOT_DIR}/win64" CACHE PATH "Path to libraries" FORCE)	
    else()
       # Linux sub-path to libs
       set ( TINYXML_LIB_DIR "${TINYXML_ROOT_DIR}/" CACHE PATH "Path to libraries" FORCE)	
    endif()

	#-------- Locate Header files
    set ( OK_H "0" )
	_FIND_FILE ( TINYXML_FILES TINYXML_INCLUDE_DIR "TINYXML/TINYXML.h" "TINYXML/TINYXML.h" OK_H )
	if ( OK_H EQUAL 1 ) 
	    message ( STATUS "  Found. TINYXML Header files. ${TINYXML_INCLUDE_DIR}" )
	else()
	    message ( "  NOT FOUND. TINYXML Header files" )
  	    set ( TINYXML_FOUND "NO" )
	endif ()

    #-------- Locate Library	
    set ( OK_DLL 0 )	
    set ( OK_LIB 0 )	
	_FIND_FILE ( LIST_DLL TINYXML_LIB_DIR "TINYXML_x64.dll" "" OK_DLL )		

  	_FIND_FILE ( LIST_LIB TINYXML_LIB_DIR "TINYXML_x64.lib" "TINYXML.so" OK_LIB )  	
    
	if ( (${OK_DLL} EQUAL 1) AND (${OK_LIB} EQUAL 1) ) 
	   message ( STATUS "  Found. TINYXML Library. ${OPENSSL_LIB_DIR}" )	   
	else()
	   set ( TINYXML_FOUND "NO" )	   
	   message ( "  NOT FOUND. TINYXML  Library. (so/dll or lib missing)" )	   
	endif()

endif()
 
if ( ${TINYXML_FOUND} STREQUAL "NO" )
   message( FATAL_ERROR "
      Please set TINYXML_ROOT_DIR to the root location 
      of installed TINYXML library containing TINYXML_x64.lib/dll
      Not found at TINYXML_ROOT_DIR: ${TINYXML_ROOT_DIR}\n"
   )
endif()

set ( TINYXML_DLL ${LIST_DLL} CACHE INTERNAL "" FORCE)
set ( TINYXML_LIB ${LIST_LIB} CACHE INTERNAL "" FORCE)

#-- We do not want user to modified these vars, but helpful to show them
message ( STATUS "  TINYXML_ROOT_DIR: ${TINYXML_ROOT_DIR}" )
message ( STATUS "  TINYXML_DLL:  ${TINYXML_DLL}" )
message ( STATUS "  TINYXML_LIB:  ${TINYXML_LIB}" )

mark_as_advanced(TINYXML_FOUND)






