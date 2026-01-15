add_executable(client_node_msg client_node_msg.cpp)
target_link_libraries(client_node_msg ${B0_LIBRARY})

add_executable(server_node_msg server_node_msg.cpp)
target_link_libraries(server_node_msg ${B0_LIBRARY})

