#include <iostream>
#include <boost/thread.hpp>

#include <b0/resolver/resolver.h>
#include <b0/node.h>
#include <b0/publisher.h>
#include <b0/subscriber.h>

void resolver_thread()
{
    b0::resolver::Resolver node;
    node.init();
    node.spin();
}

void pub_thread()
{
    b0::Node node("pub2");
    b0::Publisher pub(&node, "topic2");
    node.init();
    std::vector<b0::message::MessagePart> parts;
    parts.resize(2);
    parts[0].payload = "foo";
    parts[1].payload = "bar";
    for(;;) pub.publish(parts);
}

void sub_callback(const std::vector<b0::message::MessagePart> &parts)
{
    exit(parts.size() == 2 && parts[0].payload == "foo" && parts[1].payload == "bar" ? 0 : 1);
}

void sub_thread()
{
    b0::Node node("sub2");
    b0::Subscriber sub(&node, "topic2", &sub_callback);
    node.init();
    node.spin();
}

void timeout_thread()
{
    boost::this_thread::sleep_for(boost::chrono::seconds{4});
    std::cout << "FAILED: timeout reached" << std::endl;
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

