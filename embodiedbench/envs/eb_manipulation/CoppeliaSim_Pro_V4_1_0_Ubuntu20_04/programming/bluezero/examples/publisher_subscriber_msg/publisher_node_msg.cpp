#include <b0/node.h>
#include <b0/publisher.h>

#include <iostream>

#include "msg.h"

/*! \example publisher_subscriber_msg/publisher_node_msg.cpp
 * This is an example of creating a simple node with one publisher
 */

//! \cond HIDDEN_SYMBOLS

int main(int argc, char **argv)
{
    /*
     * Initialize B0
     */
    b0::init(argc, argv);

    /*
     * Create a node named "publisher"
     */
    b0::Node node("publisher");

    /*
     * Create a Publisher to publish on topic "A"
     */
    b0::Publisher pub(&node, "A");

    /*
     * Initialize the node (will announce node name to the network, and do other nice things)
     */
    node.init();

    int i = 0;
    while(!node.shutdownRequested())
    {
        /*
         * Process messages from node's sockets
         */
        node.spinOnce();

        /*
         * Create a message to send
         */
        MyMsg msg;
        msg.greeting = "Hello, world!";
        msg.n = 6345;

        /*
         * Send the message on the "A" topic
         */
        std::cout << "Sending: " << msg.greeting << msg.n << std::endl;
        pub.publish(msg);

        /*
         * Wait some time
         */
        node.sleepUSec(500000);
    }

    /*
     * Perform cleanup (stop threads, notify resolver that this node has quit, ...)
     */
    node.cleanup();

    return 0;
}

//! \endcond

