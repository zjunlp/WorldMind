#include <atomic>
#include <boost/thread.hpp>

#include <b0/resolver/resolver.h>
#include <b0/node.h>
#include <b0/publisher.h>
#include <b0/subscriber.h>

std::atomic<bool> quit;
int result;

void resolver_thread()
{
    b0::resolver::Resolver node;
    node.init();
    node.spin();
}

void pub_thread()
{
    b0::Node node("pub");
    b0::Publisher pub1(&node, "topic1");
    b0::Publisher pub2(&node, "topic2");
    b0::Publisher pub3(&node, "topic3");
    node.init();
    std::string m = "foo";
    while(!quit.load())
    {
        pub1.publish(m);
        pub2.publish(m);
        pub3.publish(m);
    }
    node.cleanup();
    exit(0);
}

void sub_thread()
{
    b0::Node node("sub");
    b0::Subscriber sub1(&node, "topic1");
    b0::Subscriber sub2(&node, "topic2");
    b0::Subscriber sub3(&node, "topic3");
    node.init();
    std::string m1, m2, m3;
    sub1.readRaw(m1);
    sub2.readRaw(m2);
    sub3.readRaw(m3);
    std::string e = "foo";
    if(m1 == e && m2 == e && m3 == e)
        result = 0;
    quit.store(true);
    node.cleanup();
}

void timeout_thread()
{
    boost::this_thread::sleep_for(boost::chrono::seconds{4});
    exit(1);
}

int main(int argc, char **argv)
{
    b0::init(argc, argv);

    result = 1;
    quit.store(true);
    boost::thread t0(&timeout_thread);
    boost::thread t1(&resolver_thread);
    boost::this_thread::sleep_for(boost::chrono::seconds{1});
    boost::thread t2(&sub_thread);
    boost::this_thread::sleep_for(boost::chrono::seconds{1});
    boost::thread t3(&pub_thread);
    t0.join();
}

