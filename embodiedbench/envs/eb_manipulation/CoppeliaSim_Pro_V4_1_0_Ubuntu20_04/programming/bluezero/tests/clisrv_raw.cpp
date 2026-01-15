#include <iostream>

#include <boost/thread.hpp>

#include <b0/resolver/resolver.h>
#include <b0/node.h>
#include <b0/service_client.h>
#include <b0/service_server.h>

void resolver_thread()
{
    b0::resolver::Resolver node;
    node.init();
    node.spin();
}

void cli_thread()
{
    b0::Node node("cli");
    b0::ServiceClient cli(&node, "service1");
    node.init();
    std::string req = "hello";
    std::string rep;
    cli.writeRaw(req);
    cli.readRaw(rep);
    std::cout << "server response: " << rep << std::endl;
    exit(rep == "world" ? 0 : 1);
}

void srv_thread()
{
    b0::Node node("srv");
    b0::ServiceServer srv(&node, "service1");
    node.init();
    std::string req;
    std::string resp = "world";
    srv.readRaw(req);
    srv.writeRaw(resp);
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
    boost::thread t2(&srv_thread);
    boost::this_thread::sleep_for(boost::chrono::seconds{1});
    boost::thread t3(&cli_thread);
    t0.join();
}

