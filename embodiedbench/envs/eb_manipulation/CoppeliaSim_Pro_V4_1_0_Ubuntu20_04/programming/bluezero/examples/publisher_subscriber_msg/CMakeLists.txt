add_executable(publisher_node_msg publisher_node_msg.cpp)
target_link_libraries(publisher_node_msg ${B0_LIBRARY})

add_executable(subscriber_node_msg subscriber_node_msg.cpp)
target_link_libraries(subscriber_node_msg ${B0_LIBRARY})

