#include <b0/node.h>
#include <b0/service_client.h>

#include <iostream>

/*! \example client_server/client_node.cpp
 * This is an example of a simple node with a service client
 */

//! \cond HIDDEN_SYMBOLS

int main(int argc, char **argv)
{
    /*
     * Initialize B0
     */
    b0::init(argc, argv);

    /*
     * Create a node named "client"
     */
    b0::Node node("client");

    /*
     * Create a ServiceClient that will connect to the service "control"
     */
    b0::ServiceClient cli(&node, "control");

    /*
     * Initialize the node (will announce the node name to the network, and do other nice things)
     */
    node.init();

    /*
     * Create a request message
     */
    std::string req = "hello";
    std::cout << "Sending: " << req << std::endl;

    /*
     * The response will be written here
     */
    std::string rep;

    /*
     * Call the service (blocking)
     */
    cli.call(req, rep);
    std::cout << "Received: " << rep << std::endl;

    /*
     * Perform cleanup (stop threads, notify resolver that this node has quit, ...)
     */
    node.cleanup();

    return 0;
}

//! \endcond

