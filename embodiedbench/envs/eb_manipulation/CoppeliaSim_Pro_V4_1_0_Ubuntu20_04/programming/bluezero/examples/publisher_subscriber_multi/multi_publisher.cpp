#include <b0/node.h>
#include <b0/publisher.h>

#include <iostream>

/*! \example publisher_subscriber_multi/multi_publisher.cpp
 * This is an example of having multiple publishers in one node
 */

//! \cond HIDDEN_SYMBOLS

/*
 * This callback will be called continuously by the b0::Node::spin() method
 */
void callback(b0::Publisher &pubA, b0::Publisher &pubB)
{
    static int i = 0;

    /*
     * Publish some data on "A"
     */
    std::string msg1 = (boost::format("meow-%d") % i++).str();
    pubA.publish(msg1);

    /*
     * Publish some data on "B"
     */
    std::string msg2 = (boost::format("woof-%d") % i++).str();
    pubB.publish(msg2);
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
     * Create two publishers:
     *  - pubA publishes messages on topic "A"
     *  - pubB publishes messages on topic "B"
     */
    b0::Publisher pubA(&node, "A");
    b0::Publisher pubB(&node, "B");

    /*
     * Initialize the node (will announce node name to the network, and do other nice things)
     */
    node.init();

    /*
     * Spin the node (continuously process messages, and call callback() to send messages)
     */
    node.spin([&]() { callback(pubA, pubB); });

    /*
     * Perform cleanup (stop threads, notify resolver that this node has quit, ...)
     */
    node.cleanup();

    return 0;
}

//! \endcond

