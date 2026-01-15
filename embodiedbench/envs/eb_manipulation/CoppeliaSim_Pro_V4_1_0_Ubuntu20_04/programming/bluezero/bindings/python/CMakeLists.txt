# Include python
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_PATH})

#set(CMAKE_SWIG_FLAGS "")

set_property(SOURCE ../bluezero.i PROPERTY CPLUSPLUS ON)
swig_add_library(
    bluezero_python
    LANGUAGE python
    SOURCES ../bluezero.i
)
swig_link_libraries(bluezero_python b0 ${PYTHON_LIBRARIES})
set_target_properties(_bluezero_python PROPERTIES OUTPUT_NAME "_bluezero")

set(PYTHON_INSTALL_FILES
        ${CMAKE_CURRENT_BINARY_DIR}/bluezero.py
        ${CMAKE_CURRENT_BINARY_DIR}/_bluezero.so)

set(SETUP_PY_IN ${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in)
set(SETUP_PY_OUT ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
configure_file(${SETUP_PY_IN} ${SETUP_PY_OUT})

add_custom_target(install-python
    DEPENDS _bluezero
    COMMAND python ${SETUP_PY_OUT} install)
