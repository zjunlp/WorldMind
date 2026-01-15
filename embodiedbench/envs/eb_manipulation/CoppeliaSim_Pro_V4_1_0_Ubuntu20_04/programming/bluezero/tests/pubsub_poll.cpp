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
    node.cleanup();
}

void pub_thread()
{
    b0::Node node("pub");
    b0::Publisher pub(&node, "topic1");
    node.init();
    node.sleepUSec(1000000);
    std::string m("foo");
    bool p = true;
    for(;;)
    {
        if(p)
        {
            std::cout << "pub: publish message: " << m << std::endl;
            p = false;
        }
        pub.publish(m);
    }
}

void sub_thread()
{
    b0::Node node("sub");
    b0::Subscriber sub(&node, "topic1", b0::Subscriber::CallbackRaw{});
    node.init();
    for(;;) if(sub.poll(500))
    {
        std::string m;
        sub.readRaw(m);
        std::cout << "sub: got message: " << m << std::endl;
        exit(m == "foo" ? 0 : 1);
    }
    else
    {
        std::cout << "sub: waiting" << std::endl;
    }
}

void timeout_thread()
{
    boost::this_thread::sleep_for(boost::chrono::seconds{6});
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

