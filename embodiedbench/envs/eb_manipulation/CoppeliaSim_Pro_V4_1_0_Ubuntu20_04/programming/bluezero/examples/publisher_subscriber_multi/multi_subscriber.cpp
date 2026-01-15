#include <b0/node.h>
#include <b0/subscriber.h>

#include <iostream>

/*! \example publisher_subscriber_multi/multi_subscriber.cpp
 * This is an example of having multiple subscribers inside one node
 */

//! \cond HIDDEN_SYMBOLS

void callback1(const std::string &msg)
{
    std::cout << "Received: " << msg << std::endl;
}

void callback2(const std::string &msg)
{
    std::cout << "Received: " << msg << std::endl;
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
     * Subscribe on topic "A" and call callback1(const std::string &msg) when a message is received.
     */
    b0::Subscriber subA(&node, "A", &callback1);

    /*
     * Subscribe on topic "B" and call callback2(const std::string &msg) when a message is received.
     */
    b0::Subscriber subB(&node, "B", &callback2);

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

