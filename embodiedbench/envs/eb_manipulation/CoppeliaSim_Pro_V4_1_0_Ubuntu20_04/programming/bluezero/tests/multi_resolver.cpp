#include <map>
#include <mutex>
#include <iostream>
#include <boost/thread.hpp>

#include <b0/resolver/resolver.h>
#include <b0/node.h>
#include <b0/publisher.h>
#include <b0/subscriber.h>

std::map<int, bool> passed;
std::mutex passed_mutex;

void resolver_thread(int port)
{
    b0::resolver::Resolver node;
    node.setResolverPort(port);
    node.init();
    node.spin();
}

void pub_thread(int port)
{
    b0::Node node("pub");
    node.setResolverAddress((boost::format("tcp://localhost:%d") % port).str());
    b0::Publisher pub(&node, (boost::format("topic-%d") % port).str());
    node.init();
    std::string payload = (boost::format("payload-%d") % port).str();
    for(;;)
        pub.publish(payload);
}

void sub_thread(int port)
{
    b0::Node node("sub");
    node.setResolverAddress((boost::format("tcp://localhost:%d") % port).str());
    b0::Subscriber sub(&node, (boost::format("topic-%d") % port).str());
    node.init();
    std::string expected = (boost::format("payload-%d") % port).str();
    while(1)
    {
        std::string payload;
        sub.readRaw(payload);
        if(payload != expected)
            throw "received unexpected payload";
        std::lock_guard<std::mutex> guard(passed_mutex);
        passed[port] = true;
        bool all_passed = true;
        for(auto &x : passed) all_passed = all_passed && x.second;
        if(all_passed) exit(0);
    }
}

void timeout_thread()
{
    boost::this_thread::sleep_for(boost::chrono::seconds{8});
    exit(1);
}

int main(int argc, char **argv)
{
    b0::init(argc, argv);

    passed[22000] = false;
    passed[24000] = false;

    boost::thread t0(&timeout_thread);
    for(auto &x : passed)
    {
        int port = x.first;
        boost::thread t1(&resolver_thread, port);
        boost::this_thread::sleep_for(boost::chrono::seconds{1});
        boost::thread t2(&sub_thread, port);
        boost::this_thread::sleep_for(boost::chrono::seconds{1});
    }
    for(auto &x : passed)
    {
        int port = x.first;
        boost::thread t3(&pub_thread, port);
        boost::this_thread::sleep_for(boost::chrono::seconds{1});
    }
    t0.join();
}

