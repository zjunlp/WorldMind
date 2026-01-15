#include <iostream>
#include <boost/thread.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

#include <b0/resolver/resolver.h>
#include <b0/node.h>
#include <b0/publisher.h>
#include <b0/subscriber.h>

const int num_publishers = 100;

void resolver_thread()
{
    b0::resolver::Resolver node;
    node.init();
    node.spin();
}

void pub_thread(int i)
{
    std::string name = (boost::format("pub-%d") % i).str();
    b0::Node node(name);
    b0::Publisher pub(&node, "topic1");
    node.init();
    std::string payload = (boost::format("%d") % i).str();
    for(;;)
    {
        pub.publish(payload);
        boost::this_thread::sleep_for(boost::chrono::seconds{1});
    }
}

void sub_thread()
{
    bool received[num_publishers];

    b0::Node node("sub");
    b0::Subscriber sub(&node, "topic1");
    node.init();
    while(1)
    {
        std::string m;
        sub.readRaw(m);
        int i = boost::lexical_cast<int>(m);
        if(i < 0 || i >= num_publishers)
            throw std::runtime_error("bad payload");
        received[i] = true;

        bool all_received = true;
        for(int i = 0; i < num_publishers; i++)
            all_received = all_received && received[i];
        if(all_received)
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
    b0::init(argc, argv);

    boost::thread t0(&timeout_thread);
    boost::thread t1(&resolver_thread);
    boost::this_thread::sleep_for(boost::chrono::seconds{1});
    boost::thread t2(&sub_thread);
    boost::this_thread::sleep_for(boost::chrono::seconds{1});
    for(int i = 0; i < num_publishers; i++)
    {
        boost::thread t3(&pub_thread, i);
    }
    t0.join();
}

