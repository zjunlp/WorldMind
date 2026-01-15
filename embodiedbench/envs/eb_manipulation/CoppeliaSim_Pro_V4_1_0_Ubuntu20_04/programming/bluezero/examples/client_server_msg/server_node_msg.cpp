#include <b0/node.h>
#include <b0/service_server.h>

#include <iostream>

#include "msg.h"

/*! \example client_server_msg/server_node_msg.cpp
 * This is an example of a simple node with a service server.
 */

//! \cond HIDDEN_SYMBOLS

/*
 * This callback will be called whenever a request message is read from the socket
 */
void callback(const AddRequest &req, AddReply &rep)
{
    std::cout << "Received: a=" << req.a << " b=" << req.b << std::endl;
    rep.c = req.a + req.b;
    std::cout << "Sending: c=" << rep.c << std::endl;
}

int main(int argc, char **argv)
{
    /*
     * Initialize B0
     */
    b0::init(argc, argv);

    /*
     * Create a node named "server"
     *
     * Note: if another node with the same name exists on the network, this node will
     *       get a different name
     */
    b0::Node node("server");

    /*
     * Create a ServiceServer for a service named "control"
     * It will call the specified callback upon receiving requests.
     */
    b0::ServiceServer srv(&node, "control", &callback);

    /*
     * Initialize the node (will announce node name to the network, and do other nice things)
     */
    node.init();

    /*
     * Spin the node (continuously process incoming requests and call callbacks)
     */
    node.spin();

    /*
     * Perform cleanup (stop threads, notify resolver that this node has quit, ...)
     */
    node.cleanup();

    return 0;
}

//! \endcond

