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
    cli.setReadTimeout(1000);
    std::string req = "foo";
    std::string rep;
    std::cout << "CLI: client request: " << req << std::endl;
    cli.writeRaw(req);
    for(int i = 0;; i++)
    {
        try
        {
            // Using ServiceClient::call() here will fail the second time, because
            // a Socket::write() has already happened without being followed by a read(),
            // so the ZMQ socket is still in the state where it expect a read() to be called,
            // but  call() will call write() first
            //
            // cli.call(req, rep);
            cli.readRaw(rep);
            std::cout << "CLI: server response: " << rep << std::endl;
            break;
        }
        catch(b0::exception::Exception &ex)
        {
            std::cout << "CLI: read timed out!" << std::endl;
        }

        if(i < 10)
            std::cout << "CLI: retrying..." <<  std::endl;
        else break;
    }
    exit(rep == "foo_" ? 0 : 1);
}

void srv_thread()
{
    b0::Node node("srv");
    b0::ServiceServer srv(&node, "service1");
    node.init();
    std::string req;
    std::string rep;
    int w = 3000000;
    while(1)
    {
        srv.readRaw(req);
        std::cout << "SRV: got request: " << req << std::endl;
        rep = req + "_";
        node.sleepUSec(w);
        std::cout << "SRV: sending reply: " << rep << std::endl;
        srv.writeRaw(rep);
        w /= 2;
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

    boost::thread t0(&timeout_thread);
    boost::thread t1(&resolver_thread);
    boost::this_thread::sleep_for(boost::chrono::seconds{1});
    boost::thread t2(&srv_thread);
    boost::this_thread::sleep_for(boost::chrono::seconds{1});
    boost::thread t3(&cli_thread);
    t0.join();
}

