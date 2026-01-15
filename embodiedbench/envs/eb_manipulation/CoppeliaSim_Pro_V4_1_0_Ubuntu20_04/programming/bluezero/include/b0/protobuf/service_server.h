#ifndef B0__PROTOBUF__SERVICE_SERVER_H__INCLUDED
#define B0__PROTOBUF__SERVICE_SERVER_H__INCLUDED

#include <string>

#include <boost/function.hpp>
#include <boost/bind.hpp>

#include <b0/service_server.h>
#include <b0/protobuf/socket.h>

namespace b0
{

namespace protobuf
{

/*!
 * \brief The service server template class
 *
 * This template class specializes b0::AbstractServiceClient to a specific request/response type.
 *
 * If using a callback, when a request is received, it will be handed to the callback, as long as
 * a spin method is called (e.g. Node::spin(), Node::spinOnce() or ServiceServer::spinOnce()).
 *
 * You can directly read requests and write replies from the underlying socket, by using
 * ServiceServer::poll(), ServiceServer::read() and ServiceServer::write().
 *
 * \sa b0::ServiceClient, b0::ServiceServer, b0::AbstractServiceClient, b0::AbstractServiceServer
 */
template<typename TReq, typename TRep>
class ServiceServer : public b0::ServiceServer, public SocketProtobuf
{
public:
    /*!
     * \brief Construct a ServiceServer child of a specific Node, using a boost::function as callback
     */
    ServiceServer(Node *node, std::string service_name, boost::function<void(const TReq&, TRep&)> callback = 0, bool managed = true, bool notify_graph = true)
        : b0::ServiceServer(node, service_name, 0, managed, notify_graph),
          callback_(callback)
    {
    }

    /*!
     * \brief Construct a ServiceServer child of a specific Node, using a method (of the Node subclass) as callback
     */
    template<class TNode>
    ServiceServer(TNode *node, std::string service_name, void (TNode::*callbackMethod)(const TReq&, TRep&), bool managed = true, bool notify_graph = true)
        : ServiceServer(node, service_name, boost::bind(callbackMethod, node, _1, _2), managed, notify_graph)
    {
        // delegate constructor. leave empty
    }

    /*!
     * \brief Construct a ServiceServer child of a specific Node, using a method as callback
     */
    template<class T>
    ServiceServer(Node *node, std::string service_name, void (T::*callbackMethod)(const TReq&, TRep&), T *callbackObject, bool managed = true, bool notify_graph = true)
        : ServiceServer(node, service_name, boost::bind(callbackMethod, callbackObject, _1, _2), managed, notify_graph)
    {
        // delegate constructor. leave empty
    }

    /*!
     * \brief Poll and read incoming messages, and dispatch them (called by b0::Node::spinOnce())
     */
    virtual void spinOnce() override
    {
        if(callback_.empty()) return;

        while(poll())
        {
            TReq req;
            TRep rep;
            read(this, req);
            callback_(req, rep);
            write(this, rep);
        }
    }

protected:
    /*!
     * \brief Callback which will be called when a new message is read from the socket
     */
    boost::function<void(const TReq&, TRep&)> callback_;
};

} // namespace protobuf

} // namespace b0

#endif // B0__PROTOBUF__SERVICE_SERVER_H__INCLUDED
