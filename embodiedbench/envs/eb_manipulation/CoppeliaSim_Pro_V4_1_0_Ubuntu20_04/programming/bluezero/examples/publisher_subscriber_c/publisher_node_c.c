#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <b0/bindings/c.h>

/*! \example publisher_subscriber_c/publisher_node_c.c
 * This is an example of creating a simple node with one publisher
 * using the C API
 */

//! \cond HIDDEN_SYMBOLS

int main(int argc, char **argv)
{
    /*
     * Initialize B0
     */
    b0_init(&argc, argv);

    /*
     * Create a node named "publisher"
     */
    b0_node *node = b0_node_new("publisher");

    /*
     * Create a Publisher to publish on topic "A"
     */
    b0_publisher *pub = b0_publisher_new(node, "A");

    /*
     * Initialize the node (will announce node name to the network, and do other nice things)
     */
    b0_node_init(node);

    int i = 0;
    while(!b0_node_shutdown_requested(node))
    {
        /*
         * Process messages from node's sockets
         */
        b0_node_spin_once(node);

        /*
         * Create a message to send
         */
        char msg[100];
        snprintf(msg, 100, "msg-%d", i++);

        /*
         * Send the message on the "A" topic
         */
        printf("Sending: %s\n", msg);
        b0_publisher_publish(pub, msg, strlen(msg) + 1);

        /*
         * Wait some time
         */
        b0_node_sleep_usec(node, 1000000);
    }

    /*
     * Perform cleanup (stop threads, notify resolver that this node has quit, ...)
     */
    b0_node_cleanup(node);

    /*
     * Free objects
     */
    b0_publisher_delete(pub);
    b0_node_delete(node);

    return 0;
}

//! \endcond

