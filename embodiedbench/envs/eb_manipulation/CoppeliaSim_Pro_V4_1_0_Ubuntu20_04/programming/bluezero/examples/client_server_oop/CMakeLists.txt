add_executable(client_node_object client_node_object.cpp)
target_link_libraries(client_node_object ${B0_LIBRARY})

add_executable(server_node_object server_node_object.cpp)
target_link_libraries(server_node_object ${B0_LIBRARY})

