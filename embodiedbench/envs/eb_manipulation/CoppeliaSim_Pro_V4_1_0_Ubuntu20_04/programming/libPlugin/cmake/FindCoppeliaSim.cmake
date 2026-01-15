cmake_minimum_required(VERSION 3.16.3)
include(CMakeParseArguments)

function(COPPELIASIM_FIND_ERROR MESSAGE)
    if(CoppeliaSim_FIND_REQUIRED)
        message(FATAL_ERROR ${MESSAGE})
    elseif(NOT CoppeliaSim_FIND_QUIETLY)
        message(SEND_ERROR ${MESSAGE})
    endif()
endfunction()

# redefine LIBPLUGIN_DIR as the directory containing this module:

get_filename_component(LIBPLUGIN_DIR ${CMAKE_CURRENT_LIST_DIR} DIRECTORY)
if(NOT CoppeliaSim_FIND_QUIETLY)
    message(STATUS "CoppeliaSim: LIBPLUGIN_DIR: ${LIBPLUGIN_DIR}.")
endif()

# determine the value of COPPELIASIM_ROOT_DIR:

if(NOT COPPELIASIM_ROOT_DIR)
    if(NOT DEFINED ENV{COPPELIASIM_ROOT_DIR})
        if(EXISTS "${LIBPLUGIN_DIR}/../../programming/include")
            get_filename_component(COPPELIASIM_PROGRAMMING_DIR ${LIBPLUGIN_DIR} DIRECTORY)
        elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../programming/include")
            get_filename_component(COPPELIASIM_PROGRAMMING_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
        else()
            coppeliasim_find_error("Cannot figure out the value for COPPELIASIM_ROOT_DIR.")
            return()
        endif()
        get_filename_component(COPPELIASIM_ROOT_DIR ${COPPELIASIM_PROGRAMMING_DIR} DIRECTORY)
        unset(COPPELIASIM_PROGRAMMING_DIR)
    else()
        set(COPPELIASIM_ROOT_DIR "$ENV{COPPELIASIM_ROOT_DIR}")
    endif()
endif()

# check if COPPELIASIM_ROOT_DIR exists, is valid, etc...:

set(COPPELIASIM_FOUND FALSE)

if(EXISTS "${COPPELIASIM_ROOT_DIR}")
    file(TO_CMAKE_PATH "${COPPELIASIM_ROOT_DIR}" COPPELIASIM_ROOT_DIR)
    if(NOT CoppeliaSim_FIND_QUIETLY)
        message(STATUS "CoppeliaSim: COPPELIASIM_ROOT_DIR: ${COPPELIASIM_ROOT_DIR}.")
    endif()
else()
    coppeliasim_find_error("The specified COPPELIASIM_ROOT_DIR (${COPPELIASIM_ROOT_DIR}) does not exist")
    return()
endif()

foreach(D IN ITEMS
        "${COPPELIASIM_ROOT_DIR}/programming/include"
        "${COPPELIASIM_ROOT_DIR}/programming/common"
)
    if(NOT EXISTS "${D}" AND IS_DIRECTORY "${D}")
        coppeliasim_find_error("Directory ${D} does not exist.")
        return()
    endif()
endforeach()

set(COPPELIASIM_INCLUDE_DIR "${COPPELIASIM_ROOT_DIR}/programming/include")
set(COPPELIASIM_COMMON_DIR "${COPPELIASIM_ROOT_DIR}/programming/common")

foreach(F IN ITEMS
        "${COPPELIASIM_INCLUDE_DIR}/simLib.h"
        "${COPPELIASIM_INCLUDE_DIR}/simConst.h"
        "${COPPELIASIM_COMMON_DIR}/simLib.cpp"
        "${LIBPLUGIN_DIR}/simPlusPlus/Plugin.cpp"
)
    if(NOT EXISTS "${F}")
        coppeliasim_find_error("File ${F} does not exist.")
        return()
    endif()
endforeach()

set(COPPELIASIM_EXPORTED_SOURCES
    "${COPPELIASIM_EXPORTED_SOURCES}"
    "${COPPELIASIM_COMMON_DIR}/simLib.cpp"
    "${LIBPLUGIN_DIR}/simPlusPlus/Plugin.cpp")

if(NOT CoppeliaSim_FIND_QUIETLY)
    message(STATUS "Found CoppeliaSim installation at ${COPPELIASIM_ROOT_DIR}.")
endif()

# check required version:

if(DEFINED CoppeliaSim_FIND_VERSION)
    if(NOT CoppeliaSim_FIND_QUIETLY)
        message(STATUS "Checking CoppeliaSim header version...")
    endif()
    set(COPPELIASIM_VERSION_CHECK_SRC "${CMAKE_BINARY_DIR}/sim_version_check.cpp")
    set(COPPELIASIM_VERSION_CHECK_BIN "${CMAKE_BINARY_DIR}/sim_version_check")
    file(WRITE ${COPPELIASIM_VERSION_CHECK_SRC} "
#include <iostream>
#include <simConst.h>
int main() {
    char sep = ';';
    std::cout
        << SIM_PROGRAM_VERSION_NB/10000 << sep
        << SIM_PROGRAM_VERSION_NB/100%100 << sep
        << SIM_PROGRAM_VERSION_NB%100 << sep
        << SIM_PROGRAM_REVISION_NB << sep
        << 0 << std::endl;
}
")
    try_run(COPPELIASIM_VERSION_RUN_RESULT COPPELIASIM_VERSION_COMPILE_RESULT ${COPPELIASIM_VERSION_CHECK_BIN} ${COPPELIASIM_VERSION_CHECK_SRC} CMAKE_FLAGS -DINCLUDE_DIRECTORIES=${COPPELIASIM_INCLUDE_DIR} RUN_OUTPUT_VARIABLE COPPELIASIM_VERSION_CHECK_OUTPUT)
    if(${COPPELIASIM_VERSION_COMPILE_RESULT})
        if(${COPPELIASIM_VERSION_RUN_RESULT} EQUAL 0)
            list(GET COPPELIASIM_VERSION_CHECK_OUTPUT 0 COPPELIASIM_VERSION_MAJOR)
            list(GET COPPELIASIM_VERSION_CHECK_OUTPUT 1 COPPELIASIM_VERSION_MINOR)
            list(GET COPPELIASIM_VERSION_CHECK_OUTPUT 2 COPPELIASIM_VERSION_PATCH)
            list(GET COPPELIASIM_VERSION_CHECK_OUTPUT 3 COPPELIASIM_VERSION_TWEAK)
            set(COPPELIASIM_VERSION_COUNT 4)
            list(GET COPPELIASIM_VERSION_CHECK_OUTPUT 3 COPPELIASIM_REVISION)
            set(COPPELIASIM_VERSION "${COPPELIASIM_VERSION_MAJOR}.${COPPELIASIM_VERSION_MINOR}.${COPPELIASIM_VERSION_PATCH}.${COPPELIASIM_REVISION}")
            set(COPPELIASIM_VERSION_STR "${COPPELIASIM_VERSION_MAJOR}.${COPPELIASIM_VERSION_MINOR}.${COPPELIASIM_VERSION_PATCH} rev${COPPELIASIM_REVISION}")
            if(NOT CoppeliaSim_FIND_QUIETLY)
                message(STATUS "CoppeliaSim headers version ${COPPELIASIM_VERSION_STR}")
            endif()
            math(EXPR CoppeliaSim_FIND_VERSION_NB "1000000 * ${CoppeliaSim_FIND_VERSION_MAJOR} + 10000 * ${CoppeliaSim_FIND_VERSION_MINOR} + 100 * ${CoppeliaSim_FIND_VERSION_PATCH} + ${CoppeliaSim_FIND_VERSION_TWEAK}")
            add_compile_definitions(SIM_REQUIRED_PROGRAM_VERSION_NB=${CoppeliaSim_FIND_VERSION_NB})
            if(${COPPELIASIM_VERSION} VERSION_LESS ${CoppeliaSim_FIND_VERSION})
                coppeliasim_find_error("Found CoppeliaSim version ${COPPELIASIM_VERSION} but ${CoppeliaSim_FIND_VERSION} required.")
                return()
            endif()
        else()
            coppeliasim_find_error("Failed to run CoppeliaSim version check program")
            return()
        endif()
    else()
        coppeliasim_find_error("Failed to compile CoppeliaSim version check program")
        return()
    endif()
endif()

if(WIN32)
    add_definitions(-DWIN_SIM)
    add_definitions(-DNOMINMAX)
    add_definitions(-Dstrcasecmp=_stricmp)
    if((MSVC) AND (MSVC_VERSION GREATER_EQUAL 1914))
        add_compile_options("/Zc:__cplusplus")
    endif()
    set(COPPELIASIM_LIBRARIES shlwapi)
    set(COPPELIASIM_PLUGINS_DIR ${COPPELIASIM_ROOT_DIR})
elseif(UNIX AND NOT APPLE)
    add_definitions(-DLIN_SIM)
    set(COPPELIASIM_LIBRARIES "")
    set(COPPELIASIM_PLUGINS_DIR ${COPPELIASIM_ROOT_DIR})
elseif(UNIX AND APPLE)
    add_definitions(-DMAC_SIM)
    set(COPPELIASIM_LIBRARIES "")
    set(COPPELIASIM_PLUGINS_DIR ${COPPELIASIM_ROOT_DIR}/coppeliaSim.app/Contents/MacOS/)
endif()

include_directories(${LIBPLUGIN_DIR})

function(COPPELIASIM_GENERATE_STUBS GENERATED_OUTPUT_DIR)
    cmake_parse_arguments(COPPELIASIM_GENERATE_STUBS "" "XML_FILE;LUA_FILE" "" ${ARGN})
    if(NOT CoppeliaSim_FIND_QUIETLY)
        message(STATUS "Adding simStubsGen command...")
    endif()
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    if("${COPPELIASIM_GENERATE_STUBS_LUA_FILE}" STREQUAL "")
        add_custom_command(OUTPUT ${GENERATED_OUTPUT_DIR}/stubs.cpp ${GENERATED_OUTPUT_DIR}/stubs.h
            COMMAND ${Python3_EXECUTABLE} ${LIBPLUGIN_DIR}/simStubsGen/generate.py --xml-file ${COPPELIASIM_GENERATE_STUBS_XML_FILE} --gen-all ${GENERATED_OUTPUT_DIR}
            DEPENDS ${COPPELIASIM_GENERATE_STUBS_XML_FILE})
    else()
        add_custom_command(OUTPUT ${GENERATED_OUTPUT_DIR}/stubs.cpp ${GENERATED_OUTPUT_DIR}/stubs.h ${GENERATED_OUTPUT_DIR}/lua_calltips.cpp
            COMMAND ${Python3_EXECUTABLE} ${LIBPLUGIN_DIR}/simStubsGen/generate.py --xml-file ${COPPELIASIM_GENERATE_STUBS_XML_FILE} --lua-file ${COPPELIASIM_GENERATE_STUBS_LUA_FILE} --gen-all ${GENERATED_OUTPUT_DIR}
            DEPENDS ${COPPELIASIM_GENERATE_STUBS_XML_FILE})
    endif()
    set_property(SOURCE ${GENERATED_OUTPUT_DIR}/stubs.cpp PROPERTY SKIP_AUTOGEN ON)
    set_property(SOURCE ${GENERATED_OUTPUT_DIR}/stubs.h PROPERTY SKIP_AUTOGEN ON)
    include_directories("${GENERATED_OUTPUT_DIR}")
    set(COPPELIASIM_EXPORTED_SOURCES ${COPPELIASIM_EXPORTED_SOURCES} "${GENERATED_OUTPUT_DIR}/stubs.cpp" PARENT_SCOPE)
endfunction(COPPELIASIM_GENERATE_STUBS)

find_package(Git)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git AND GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE "BUILD_GIT_VERSION"
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE)
else()
    set(BUILD_GIT_VERSION "unknown")
endif()

set(COPPELIASIM_FOUND TRUE)
