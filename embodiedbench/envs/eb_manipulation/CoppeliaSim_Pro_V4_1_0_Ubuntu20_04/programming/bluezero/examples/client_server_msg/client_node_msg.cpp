#include <b0/node.h>
#include <b0/service_client.h>

#include <iostream>

#include "msg.h"

/*! \example client_server_msg/client_node_msg.cpp
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
    AddRequest req;
    req.a = 1;
    req.b = 2;
    std::cout << "Sending: a=" << req.a << " b=" << req.b << std::endl;

    /*
     * The response will be written here
     */
    AddReply rep;

    /*
     * Call the service (blocking)
     */
    cli.call(req, rep);
    std::cout << "Received: c=" << rep.c << std::endl;

    /*
     * Perform cleanup (stop threads, notify resolver that this node has quit, ...)
     */
    node.cleanup();

    return 0;
}

//! \endcond

