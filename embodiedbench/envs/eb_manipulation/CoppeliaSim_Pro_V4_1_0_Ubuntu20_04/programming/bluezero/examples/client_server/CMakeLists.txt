add_executable(client_node client_node.cpp)
target_link_libraries(client_node ${B0_LIBRARY})

add_executable(server_node server_node.cpp)
target_link_libraries(server_node ${B0_LIBRARY})

