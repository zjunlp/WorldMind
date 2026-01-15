find_package(PythonInterp)
find_package(PythonLibs)
if (${PYTHON_VERSION_MAJOR} EQUAL 3)
    find_package(Boost COMPONENTS python3)
else()
    find_package(Boost COMPONENTS python)
endif()

include_directories(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
link_libraries(${Boost_LIBRARIES} ${PYTHON_LIBRARIES}) # Deprecated but so convenient!

add_library(pyb0 SHARED pyb0.cpp)
target_link_libraries(pyb0 b0)
set_target_properties(pyb0 PROPERTIES PREFIX "")
if(APPLE)
    set_target_properties(pyb0 PROPERTIES SUFFIX ".so")
endif()

