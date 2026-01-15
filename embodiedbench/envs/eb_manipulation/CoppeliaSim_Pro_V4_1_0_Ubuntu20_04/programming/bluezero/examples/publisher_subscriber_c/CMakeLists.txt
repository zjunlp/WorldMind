add_executable(publisher_node_c publisher_node_c.c)
target_link_libraries(publisher_node_c ${B0_LIBRARY_SHARED})

add_executable(subscriber_node_c subscriber_node_c.c)
target_link_libraries(subscriber_node_c ${B0_LIBRARY_SHARED})

