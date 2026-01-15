#include <string.h>
#include <stdio.h>
#include <b0/bindings/c.h>

/*! \example client_server_c/client_node_c.c
 * This is an example of a simple node with a service client
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
     * Create a node named "client"
     */
    b0_node *node = b0_node_new("client");

    /*
     * Create a ServiceClient that will connect to the service "control"
     */
    b0_service_client *cli = b0_service_client_new(node, "control");

    /*
     * Initialize the node (will announce the node name to the network, and do other nice things)
     */
    b0_node_init(node);

    /*
     * Create a request message
     */
    const char *req = "hello";
    printf("Sending: %s\n", req);

    /*
     * The response will be written here
     */
    char *rep;
    size_t rep_sz;

    /*
     * Call the service (blocking)
     */
    rep = b0_service_client_call(cli, req, strlen(req) + 1, &rep_sz);
    printf("Received: %s\n", rep);

    /*
     * Free reply buffer
     */
    b0_buffer_delete(rep);

    /*
     * Perform cleanup (stop threads, notify resolver that this node has quit, ...)
     */
    b0_node_cleanup(node);

    /*
     * Free objects
     */
    b0_service_client_delete(cli);
    b0_node_delete(node);

    return 0;
}

//! \endcond

