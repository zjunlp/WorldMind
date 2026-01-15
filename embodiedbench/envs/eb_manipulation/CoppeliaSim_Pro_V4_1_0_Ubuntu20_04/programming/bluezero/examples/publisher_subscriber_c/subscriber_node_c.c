#include <stdio.h>
#include <stdlib.h>
#include <b0/bindings/c.h>

/*! \example publisher_subscriber_c/subscriber_node_c.c
 * This is an example of a simple node with one callback-based subscriber
 * using the C API
 */

//! \cond HIDDEN_SYMBOLS

/*
 * This callback will be called whenever a message is received on any
 * of the subscribed topics
 */
void callback(const void *msg, size_t size)
{
    printf("Received: %s\n", (const char *)msg);
}

int main(int argc, char **argv)
{
    /*
     * Initialize B0
     */
    b0_init(&argc, argv);

    /*
     * Create a node named "subscriber"
     */
    b0_node *node = b0_node_new("subscriber");

    /*
     * Create a Subscriber to subscribe to topic "A"
     * It will call the specified callback upon receiving messages
     */
    b0_subscriber *sub = b0_subscriber_new(node, "A", &callback);

    /*
     * Initialize the node (will announce node name to the network, and do other nice things)
     */
    b0_node_init(node);

    /*
     * Spin the node (continuously process incoming messages and call callbacks)
     */
    b0_node_spin(node);

    /*
     * Perform cleanup (stop threads, notify resolver that this node has quit, ...)
     */
    b0_node_cleanup(node);

    /*
     * Free objects
     */
    b0_subscriber_delete(sub);
    b0_node_delete(node);

    return 0;
}

//! \endcond

