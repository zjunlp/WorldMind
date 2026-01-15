#include <b0/node.h>
#include <b0/publisher.h>

#include <iostream>

/*! \example publisher_subscriber/publisher_node.cpp
 * This is an example of creating a simple node with one publisher
 */

//! \cond HIDDEN_SYMBOLS

/*
 * This callback will be called continuously by the b0::Node::spin() method
 */
void callback(b0::Publisher &pub)
{
    static int i = 0;

    /*
     * Create a message to send
     */
    std::string msg = (boost::format("msg-%d") % i++).str();

    /*
     * Send the message on the "A" topic
     */
    std::cout << "Sending: " << msg << std::endl;
    pub.publish(msg);
}

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

    /*
     * Spin the node (continuously process messages, and call callback() to send a message)
     */
    node.spin([&]() { callback(pub); });

    /*
     * Perform cleanup (stop threads, notify resolver that this node has quit, ...)
     */
    node.cleanup();

    return 0;
}

//! \endcond

