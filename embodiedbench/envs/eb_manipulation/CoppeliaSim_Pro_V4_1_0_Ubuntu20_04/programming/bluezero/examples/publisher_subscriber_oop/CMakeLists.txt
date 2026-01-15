add_executable(publisher_node_object publisher_node_object.cpp)
target_link_libraries(publisher_node_object ${B0_LIBRARY})

add_executable(subscriber_node_object subscriber_node_object.cpp)
target_link_libraries(subscriber_node_object ${B0_LIBRARY})

