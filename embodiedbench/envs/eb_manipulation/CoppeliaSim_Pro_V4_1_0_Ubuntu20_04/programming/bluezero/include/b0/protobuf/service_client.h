#ifndef B0__PROTOBUF__SERVICE_CLIENT_H__INCLUDED
#define B0__PROTOBUF__SERVICE_CLIENT_H__INCLUDED

#include <string>

#include <b0/service_client.h>
#include <b0/protobuf/socket.h>

namespace b0
{

namespace protobuf
{

/*!
 * \brief The service client template class
 *
 * This template class specializes b0::AbstractServiceClient to a specific request/response type.
 * It implements the call() method as well.
 *
 * The remote service is invoked with ServiceClient::call() and the call is blocking.
 * It will unblock as soon as the server sends out a reply.
 *
 * You can make it work asynchronously by directly using ServiceClient::write(), and polling
 * for the reply with ServiceClient::poll(), followed by ServiceClient::read().
 *
 * \sa b0::ServiceClient, b0::ServiceServer, b0::AbstractServiceClient, b0::AbstractServiceServer
 */
template<typename TReq, typename TRep>
class ServiceClient : public b0::ServiceClient, public SocketProtobuf
{
public:
    /*!
     * \brief Construct a ServiceClient child of a specific Node, which will connect to the specified socket in the specified node
     */
    ServiceClient(Node *node, std::string service_name, bool managed = true, bool notify_graph = true)
        : b0::ServiceClient(node, service_name, managed, notify_graph)
    {
    }

    /*!
     * \brief Write a request and read a reply from the underlying ZeroMQ REQ socket
     * \sa ServiceServer::read(), ServiceServer::write()
     */
    virtual void call(const TReq &req, TRep &rep)
    {
        write(this, req);
        read(this, rep);
    }
};

} // namespace protobuf

} // namespace b0

#endif // B0__PROTOBUF__SERVICE_CLIENT_H__INCLUDED
