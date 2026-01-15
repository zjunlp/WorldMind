add_executable(client_node_c client_node_c.c)
target_link_libraries(client_node_c ${B0_LIBRARY_SHARED})

add_executable(server_node_c server_node_c.c)
target_link_libraries(server_node_c ${B0_LIBRARY_SHARED})

