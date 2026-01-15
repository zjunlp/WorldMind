#include <b0/node.h>
#include <b0/subscriber.h>

#include <iostream>

#include "msg.h"

/*! \example publisher_subscriber_msg/subscriber_node_msg.cpp
 * This is an example of a simple node with one callback-based subscriber
 */

//! \cond HIDDEN_SYMBOLS

/*
 * This callback will be called whenever a message is received on any
 * of the subscribed topics
 */
void callback(const MyMsg &msg)
{
    std::cout << "Received: " << msg.greeting << msg.n << std::endl;
}

int main(int argc, char **argv)
{
    /*
     * Initialize B0
     */
    b0::init(argc, argv);

    /*
     * Create a node named "subscriber"
     */
    b0::Node node("subscriber");

    /*
     * Create a Subscriber to subscribe to topic "A"
     * It will call the specified callback upon receiving messages
     */
    b0::Subscriber sub(&node, "A", &callback);

    /*
     * Initialize the node (will announce node name to the network, and do other nice things)
     */
    node.init();

    /*
     * Spin the node (continuously process incoming messages and call callbacks)
     */
    node.spin();

    /*
     * Perform cleanup (stop threads, notify resolver that this node has quit, ...)
     */
    node.cleanup();

    return 0;
}

//! \endcond

