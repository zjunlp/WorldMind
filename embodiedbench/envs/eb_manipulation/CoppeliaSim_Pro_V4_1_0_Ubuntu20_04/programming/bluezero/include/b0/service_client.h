#ifndef B0__SERVICE_CLIENT_H__INCLUDED
#define B0__SERVICE_CLIENT_H__INCLUDED

#include <string>

#include <b0/b0.h>
#include <b0/socket.h>
#include <b0/message/message_part.h>

namespace b0
{

class Node;

/*!
 * \brief The service client class
 *
 * This class wraps a REQ socket. It will automatically resolve the address
 * of service name.
 *
 * \sa b0::ServiceClient, b0::ServiceServer
 */
class ServiceClient : public Socket
{
public:
    using logger::LogInterface::log;

    /*!
     * \brief Construct an ServiceClient child of the specified Node
     */
    ServiceClient(Node *node, const std::string &service_name, bool managed = true, bool notify_graph = true);

    /*!
     * \brief ServiceClient destructor
     */
    virtual ~ServiceClient();

    /*!
     * \brief Log a message using node's logger, prepending this service client informations
     */
    void log(logger::Level level, const std::string &message) const override;

    /*!
     * \brief Perform initialization and optionally send graph notify
     */
    virtual void init() override;

    /*!
     * \brief Perform cleanup and optionally send graph notify
     */
    virtual void cleanup() override;

    /*!
     * \brief Return the name of this client's service
     */
    std::string getServiceName();

    /*!
     * \brief Write a request and read a reply from the underlying ZeroMQ REQ socket
     * \sa ServiceServer::read(), ServiceServer::write()
     */
    virtual void call(const std::string &req, std::string &rep);

    /*!
     * \brief Write a request and read a reply from the underlying ZeroMQ REQ socket
     * \sa ServiceServer::read(), ServiceServer::write()
     */
    virtual void call(const std::string &req, const std::string &reqtype, std::string &rep, std::string &reptype);

    /*!
     * \brief Write a request and read a reply from the underlying ZeroMQ REQ socket
     * \sa ServiceServer::read(), ServiceServer::write()
     */
    virtual void call(const std::vector<b0::message::MessagePart> &reqparts, std::vector<b0::message::MessagePart> &repparts);

    /*!
     * \brief Write a request and read a reply from the underlying ZeroMQ REQ socket
     * \sa ServiceServer::read(), ServiceServer::write()
     */
    template<class TReq, class TRep>
    void call(const TReq &req, TRep &rep)
    {
        writeMsg(req);
        readMsg(rep);
    }

    /*!
     * \brief Write a request and read a reply from the underlying ZeroMQ REQ socket
     * \sa ServiceServer::read(), ServiceServer::write()
     */
    template<class TReq, class TRep>
    void call(const TReq &req, const std::vector<b0::message::MessagePart> &reqparts, TRep &rep, std::vector<b0::message::MessagePart> &repparts)
    {
        writeMsg(req, reqparts);
        readMsg(rep, repparts);
    }

protected:
    /*!
     * \brief Perform service address resolution
     */
    virtual void resolve();

    /*!
     * \brief Connect to service server endpoint
     */
    virtual void connect();

    /*!
     * \brief Disconnect from service server endpoint
     */
    virtual void disconnect();

    //! If false this socket will not send announcement to resolv (i.e. it will be "invisible")
    const bool notify_graph_;
};

} // namespace b0

#endif // B0__SERVICE_CLIENT_H__INCLUDED
