find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})
set(SWIG_CXX_EXTENSION cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

option(BINDINGS_SWIG_CSHARP "Build C# bindings (using SWIG)" OFF)
if(BINDINGS_SWIG_CSHARP)
    add_subdirectory(csharp)
endif()

option(BINDINGS_SWIG_PYTHON "Build Python bindings (using SWIG)" OFF)
if(BINDINGS_SWIG_PYTHON)
    add_subdirectory(python)
endif()

