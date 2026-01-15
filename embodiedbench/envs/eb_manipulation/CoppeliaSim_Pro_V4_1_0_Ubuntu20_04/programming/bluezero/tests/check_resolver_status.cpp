#include <iostream>
#include <boost/thread.hpp>

#include <b0/resolver/resolver.h>
#include <b0/node.h>
#include <b0/exceptions.h>
#include <b0/publisher.h>
#include <b0/subscriber.h>

int run_resolver = 0;

void resolver_thread()
{
    if(!run_resolver) return;
    b0::resolver::Resolver node;
    node.init();
    node.spin();
}

void node_thread()
{
    b0::Node node("testnode");
    node.setAnnounceTimeout(500);
    b0::Publisher pub(&node, "testtopic");
    b0::Subscriber sub(&node, "testtopic");
    try
    {
        node.init();
        node.cleanup();
        std::cout << "RESOLVER IS RUNNING" << std::endl;
    }
    catch(b0::exception::SocketReadError &ex)
    {
        std::cout << "RESOLVER IS NOT RUNNING" << std::endl;
    }
    exit(0);
}

void timeout_thread()
{
    boost::this_thread::sleep_for(boost::chrono::seconds{4});
    std::cout << "TIMEOUT" << std::endl;
    exit(1);
}

int main(int argc, char **argv)
{
    b0::addOptionInt("run-resolver,r", "run resolver", &run_resolver, true);
    b0::setPositionalOption("run-resolver");
    b0::init(argc, argv);

    boost::thread t0(&timeout_thread);
    boost::thread t1(&resolver_thread);
    boost::thread t2(&node_thread);
    t0.join();
}

