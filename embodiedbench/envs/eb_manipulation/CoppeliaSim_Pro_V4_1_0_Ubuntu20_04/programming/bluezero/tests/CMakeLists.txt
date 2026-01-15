add_executable(pubsub pubsub.cpp)
target_link_libraries(pubsub ${B0_LIBRARY})
add_test(pubsub pubsub 0)
add_test(pubsub_with_compression pubsub 1)

add_executable(pubsub_slowsub_conflate pubsub_slowsub.cpp)
target_link_libraries(pubsub_slowsub_conflate ${B0_LIBRARY})
add_test(pubsub_slowsub_conflate_0 pubsub_slowsub_conflate 0)
add_test(pubsub_slowsub_conflate_1 pubsub_slowsub_conflate 1)

add_executable(pubsub_slowsub_conflate_zmq pubsub_slowsub_zmq.cpp)
target_link_libraries(pubsub_slowsub_conflate_zmq ${B0_LIBRARY})
add_test(pubsub_slowsub_conflate_zmq_0 pubsub_slowsub_conflate_zmq 0)
add_test(pubsub_slowsub_conflate_zmq_1 pubsub_slowsub_conflate_zmq 1)

# this test is known to fail because of a limitation in libzmq:
# "ZMQ_CONFLATE: [...] Does not supports multi-part messages, [...]"
# http://api.zeromq.org/4-2:zmq-setsockopt
#add_executable(pubsub_slowsub_conflate_zmq_multipart pubsub_slowsub_zmq_multipart.cpp)
#target_link_libraries(pubsub_slowsub_conflate_zmq_multipart ${B0_LIBRARY})
#add_test(pubsub_slowsub_conflate_zmq_multipart_0 pubsub_slowsub_conflate_zmq_multipart 0)
#add_test(pubsub_slowsub_conflate_zmq_multipart_1 pubsub_slowsub_conflate_zmq_multipart 1)

add_executable(pubsub2 pubsub2.cpp)
target_link_libraries(pubsub2 ${B0_LIBRARY})
add_test(pubsub2 pubsub2)

add_executable(pubsub_unmanaged pubsub_unmanaged.cpp)
target_link_libraries(pubsub_unmanaged ${B0_LIBRARY})
add_test(pubsub_unmanaged pubsub_unmanaged)

add_executable(pubsub_raw pubsub_raw.cpp)
target_link_libraries(pubsub_raw ${B0_LIBRARY})
add_test(pubsub_raw pubsub_raw)

add_executable(pubsub_many pubsub_many.cpp)
target_link_libraries(pubsub_many ${B0_LIBRARY})
add_test(pubsub_many pubsub_many)

add_executable(pubsub_poll pubsub_poll.cpp)
target_link_libraries(pubsub_poll ${B0_LIBRARY})
add_test(pubsub_poll pubsub_poll)

add_executable(clisrv clisrv.cpp)
target_link_libraries(clisrv ${B0_LIBRARY})
add_test(clisrv clisrv)

add_executable(clisrv2 clisrv2.cpp)
target_link_libraries(clisrv2 ${B0_LIBRARY})
add_test(clisrv2 clisrv2)

add_executable(clisrv_raw clisrv_raw.cpp)
target_link_libraries(clisrv_raw ${B0_LIBRARY})
add_test(clisrv_raw clisrv_raw)

add_executable(clisrv_badname clisrv_badname.cpp)
target_link_libraries(clisrv_badname ${B0_LIBRARY})
add_test(clisrv_badname clisrv_badname)

add_executable(clisrv_retry clisrv_retry.cpp)
target_link_libraries(clisrv_retry ${B0_LIBRARY})
add_test(clisrv_retry clisrv_retry)

add_executable(time_sync_offset_recover time_sync_offset_recover.cpp)
target_link_libraries(time_sync_offset_recover ${B0_LIBRARY})
add_test(time_sync_offset_recover time_sync_offset_recover)

add_executable(time_sync_clock_tracking time_sync_clock_tracking.cpp)
target_link_libraries(time_sync_clock_tracking ${B0_LIBRARY})
add_test(time_sync_clock_tracking_1 time_sync_clock_tracking 0.5 0.6 1)
add_test(time_sync_clock_tracking_2 time_sync_clock_tracking 0.5 1.5 1)
add_test(time_sync_clock_tracking_3 time_sync_clock_tracking 0.5 1.66 0)

add_executable(effective_spin_rate effective_spin_rate.cpp)
target_link_libraries(effective_spin_rate ${B0_LIBRARY})
add_test(effective_spin_rate effective_spin_rate)

add_executable(compress compress.cpp)
target_link_libraries(compress ${B0_LIBRARY})
add_test(compress-zlib compress zlib)
add_test(compress-lz4 compress lz4)

add_executable(announce_timeout announce_timeout.cpp)
target_link_libraries(announce_timeout ${B0_LIBRARY})
add_test(announce_timeout announce_timeout 1000 1)
add_test(announce_notimeout announce_timeout --timeout=-1 -e0)

add_executable(two_nodes_one_thread two_nodes_one_thread.cpp)
target_link_libraries(two_nodes_one_thread ${B0_LIBRARY})
add_test(two_nodes_one_thread two_nodes_one_thread)

add_executable(check_resolver_status check_resolver_status.cpp)
target_link_libraries(check_resolver_status ${B0_LIBRARY})
add_test(check_resolver_status_when_not_running check_resolver_status 0)
add_test(check_resolver_status_when_running check_resolver_status 1)

add_executable(protocol protocol.cpp)
target_link_libraries(protocol ${B0_LIBRARY})
add_test(protocol protocol)

add_executable(json json.cpp)
target_link_libraries(json ${B0_LIBRARY})
add_test(json json)

add_executable(args_parse args.cpp)
target_link_libraries(args_parse ${B0_LIBRARY})
add_test(args args_parse -a 1 -a 2 -a 3 -b 0.5 -c 281474976710656 -n 4 w x y z)

add_executable(multi_resolver multi_resolver.cpp)
target_link_libraries(multi_resolver ${B0_LIBRARY})
add_test(multi_resolver multi_resolver)

add_executable(many_nodes many_nodes.cpp)
target_link_libraries(many_nodes ${B0_LIBRARY})
add_test(many_nodes many_nodes)

add_executable(resolver_shutdown resolver_shutdown.cpp)
target_link_libraries(resolver_shutdown ${B0_LIBRARY})
add_test(resolver_shutdown resolver_shutdown)

add_executable(c_clisrv_timeout c_clisrv_timeout.c)
target_link_libraries(c_clisrv_timeout ${B0_LIBRARY})
#add_test(c_clisrv_timeout_1 c_clisrv_timeout -w 0 -f 0)
#add_test(c_clisrv_timeout_2 c_clisrv_timeout -w 10 -f 1)

if(ENABLE_PROTOBUF)
    protobuf_generate_cpp(PROTO_TEST_SRCS PROTO_TEST_HDRS test_protobuf.proto)
    add_executable(test_protobuf test_protobuf.cpp ${PROTO_TEST_SRCS} ${PROTO_TEST_HDRS})
    target_link_libraries(test_protobuf ${B0_LIBRARY} ${PROTOBUF_LIBRARIES})
    add_test(test_protobuf test_protobuf)
endif()
