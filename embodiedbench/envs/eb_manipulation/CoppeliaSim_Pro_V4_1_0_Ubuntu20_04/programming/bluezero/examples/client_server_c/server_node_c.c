#include <stdio.h>
#include <string.h>
#include <b0/bindings/c.h>

/*! \example client_server_c/server_node_c.c
 * This is an example of a simple node with a service server.
 * using the C API
 */

//! \cond HIDDEN_SYMBOLS

/*
 * This callback will be called whenever a request message is read from the socket
 */
void * callback(const void *req, size_t sz, size_t *out_sz)
{
    printf("Received: %s\n", (const char*)req);

    const char *repmsg = "hi";
    printf("Sending: %s\n", repmsg);

    *out_sz = strlen(repmsg);
    void *rep = b0_buffer_new(*out_sz);
    memcpy(rep, repmsg, *out_sz);
    return rep;
}

int main(int argc, char **argv)
{
    /*
     * Initialize B0
     */
    b0_init(&argc, argv);

    /*
     * Create a node named "server"
     *
     * Note: if another node with the same name exists on the network, this node will
     *       get a different name
     */
    b0_node *node = b0_node_new("server");

    /*
     * Create a ServiceServer for a service named "control"
     * It will call the specified callback upon receiving requests.
     */
    b0_service_server *srv = b0_service_server_new(node, "control", &callback);

    /*
     * Initialize the node (will announce node name to the network, and do other nice things)
     */
    b0_node_init(node);

    /*
     * Spin the node (continuously process incoming requests and call callbacks)
     */
    b0_node_spin(node);

    /*
     * Perform cleanup (stop threads, notify resolver that this node has quit, ...)
     */
    b0_node_cleanup(node);

    /*
     * Free objects
     */
    b0_service_server_delete(srv);
    b0_node_delete(node);

    return 0;
}

//! \endcond

