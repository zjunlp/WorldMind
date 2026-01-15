#include <boost/thread.hpp>

#include <b0/resolver/resolver.h>
#include <b0/node.h>
#include <b0/service_client.h>
#include <b0/service_server.h>
#include <b0/exceptions.h>

void resolver_thread()
{
    b0::resolver::Resolver node;
    node.init();
    node.spin();
}

void cli_thread()
{
    b0::Node node("cli");
    b0::ServiceClient cli(&node, "service1b");
    try
    {
        node.init();
    }
    catch(b0::exception::NameResolutionError &ex)
    {
        // correct behavior: throw NameResolutionError
        exit(0);
    }
    exit(1);
}

void srv_thread()
{
    b0::Node node("srv");
    b0::ServiceServer srv(&node, "service1a");
    node.init();
    node.spin();
}

void timeout_thread()
{
    boost::this_thread::sleep_for(boost::chrono::seconds{4});
    exit(2);
}

int main(int argc, char **argv)
{
    b0::init(argc, argv);

    boost::thread t0(&timeout_thread);
    boost::thread t1(&resolver_thread);
    boost::this_thread::sleep_for(boost::chrono::seconds{1});
    boost::thread t2(&srv_thread);
    boost::this_thread::sleep_for(boost::chrono::seconds{1});
    boost::thread t3(&cli_thread);
    t0.join();
}

