#include "test_protobuf.pb.h"

#include <iostream>
#include <boost/thread.hpp>

#include <b0/resolver/resolver.h>
#include <b0/node.h>
#include <b0/protobuf/publisher.h>
#include <b0/protobuf/subscriber.h>

void resolver_thread()
{
    b0::resolver::Resolver node;
    node.init();
    node.spin();
    node.cleanup();
}

void pub_thread()
{
    b0::Node node("pub");
    b0::test::protobuf::Message msg;
    b0::protobuf::Publisher<b0::test::protobuf::Message> pub(&node, "topicp1");
    node.init();
    msg.set_a("Hello");
    msg.set_b(42);
    for(;;)
        pub.publish(msg);
}

void sub_thread()
{
    b0::Node node("sub");
    b0::test::protobuf::Message msg;
    b0::protobuf::Subscriber<b0::test::protobuf::Message> sub(&node, "topicp1");
    node.init();
    sub.read(&sub, msg);
    if(msg.a() == "Hello" && msg.b() == 42)
    {
        std::cerr << "test passed" << std::endl;
        exit(0);
    }
    else
    {
        std::cerr << "test failed: bad payload (a=" << msg.a() << ", b=" << msg.b() << ")" << std::endl;
        exit(1);
    }
}

void timeout_thread()
{
    boost::this_thread::sleep_for(boost::chrono::seconds{4});
    std::cerr << "test failed: timeout" << std::endl;
    exit(1);
}

int main(int argc, char **argv)
{
    b0::init(argc, argv);

    boost::thread t0(&timeout_thread);
    boost::thread t1(&resolver_thread);
    boost::this_thread::sleep_for(boost::chrono::seconds{1});
    boost::thread t2(&sub_thread);
    boost::this_thread::sleep_for(boost::chrono::seconds{1});
    boost::thread t3(&pub_thread);
    t0.join();
}

