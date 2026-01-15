add_executable(print print.cpp)
target_link_libraries(print ${B0_LIBRARY})

add_executable(const const.cpp)
target_link_libraries(const ${B0_LIBRARY})

add_executable(operation operation.cpp)
target_link_libraries(operation ${B0_LIBRARY})

