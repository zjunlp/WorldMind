#include <b0/node.h>
#include <b0/service_client.h>

#include <iostream>

/*! \example client_server_oop/client_node_object.cpp
 * This is an example of creating a node with a service client by subclassing b0::Node
 */

//! \cond HIDDEN_SYMBOLS

class TestClientNode : public b0::Node
{
public:
    TestClientNode()
        : cli_(this, "control")
    {
    }

    void run()
    {
        std::string req = "hello";
        std::string rep;

        std::cout << "Sending: " << req << std::endl;

        cli_.call(req, rep);

        std::cout << "Received: " << rep << std::endl;
    }

protected:
    b0::ServiceClient cli_;
};

int main(int argc, char **argv)
{
    b0::init(argc, argv);
    TestClientNode node;
    node.init();
    node.run();
    node.cleanup();
    return 0;
}

//! \endcond

