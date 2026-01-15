#include <iostream>
#include <boost/thread.hpp>

#include <b0/resolver/resolver.h>
#include <b0/node.h>
#include <b0/publisher.h>
#include <b0/subscriber.h>

int use_compression;

std::string payload0("foo", 4);
std::string payload1("\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f", 16);
std::string payload2("&CV^$%*&^VR%^v8b&^r5V#*&%*#76v58", 33);

void resolver_thread()
{
    b0::resolver::Resolver node;
    node.init();
    node.spin();
}

void pub_thread()
{
    b0::Node node("pub");
    b0::Publisher pub(&node, "topic1");
    if(use_compression)
    {
        node.info("Using compression");
        pub.setCompression("zlib", 9);
    }
    node.init();
    for(;;)
    {
        pub.publish(payload0);
        pub.publish(payload1);
        pub.publish(payload2);
    }
}

void sub_thread()
{
    int received[] = {0, 0, 0};

    b0::Node node("sub");
    b0::Subscriber sub(&node, "topic1");
    node.init();
    while(1)
    {
        std::string m;
        sub.readRaw(m);
        if(payload0 == m) received[0]++;
        if(payload1 == m) received[1]++;
        if(payload2 == m) received[2]++;
        if(received[0] && received[1] && received[2])
            exit(0);
    }
}

void timeout_thread()
{
    boost::this_thread::sleep_for(boost::chrono::seconds{4});
    exit(1);
}

int main(int argc, char **argv)
{
    b0::addOptionInt("use-compression,c", "use compression", &use_compression, true);
    b0::setPositionalOption("use-compression");
    b0::init(argc, argv);

    boost::thread t0(&timeout_thread);
    boost::thread t1(&resolver_thread);
    boost::this_thread::sleep_for(boost::chrono::seconds{1});
    boost::thread t2(&sub_thread);
    boost::this_thread::sleep_for(boost::chrono::seconds{1});
    boost::thread t3(&pub_thread);
    t0.join();
}

