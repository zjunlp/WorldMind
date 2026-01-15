#ifndef B0__SERVICE_SERVER_H__INCLUDED
#define B0__SERVICE_SERVER_H__INCLUDED

#include <string>

#include <boost/function.hpp>
#include <boost/bind.hpp>

#include <b0/b0.h>
#include <b0/socket.h>
#include <b0/message/message_part.h>

namespace b0
{

class Node;

/*!
 * \brief The service server class
 *
 * This class wraps a REP socket. It will automatically
 * announce the socket name to resolver.
 *
 * \sa b0::ServiceClient, b0::ServiceServer
 */
class ServiceServer : public Socket
{
public:
    using logger::LogInterface::log;

    //! \brief Alias for function
    template<typename T> using function = boost::function<T>;

    //! \brief Alias for callback raw without type
    using CallbackRaw = function<void(const std::string&, std::string&)>;

    //! \brief Alias for callback raw with type
    using CallbackRawType = function<void(const std::string&, const std::string&, std::string&, std::string&)>;

    //! \brief Alias for callback raw message parts
    using CallbackParts = function<void(const std::vector<b0::message::MessagePart>&, std::vector<b0::message::MessagePart>&)>;

    //! \brief Alias for callback message class
    template<class TReq, class TRep> using CallbackMsg = function<void(const TReq&, TRep&)>;

    //! \brief Alias for callback message class + raw extra parts
    template<class TReq, class TRep> using CallbackMsgParts = function<void(const TReq&, const std::vector<b0::message::MessagePart>&, TRep&, std::vector<b0::message::MessagePart>&)>;

    /*!
     * \brief Construct an ServiceServer child of the specified Node, without a callback
     */
    ServiceServer(Node *node, const std::string &service_name, bool managed = true, bool notify_graph = true);

    /*!
     * \brief Construct an ServiceServer child of the specified Node, using a function as a callback (raw without type)
     */
    ServiceServer(Node *node, const std::string &service_name, CallbackRaw callback, bool managed = true, bool notify_graph = true);

    /*!
     * \brief Construct an ServiceServer child of the specified Node, using a function as a callback (raw with type)
     */
    ServiceServer(Node *node, const std::string &service_name, CallbackRawType callback, bool managed = true, bool notify_graph = true);

    /*!
     * \brief Construct an ServiceServer child of the specified Node, using a function as a callback (raw message parts)
     */
    ServiceServer(Node *node, const std::string &service_name, CallbackParts callback, bool managed = true, bool notify_graph = true);

    /*!
     * \brief Construct an ServiceServer child of the specified Node, using a function as a callback (message class)
     */
    template<class TReq, class TRep>
    ServiceServer(Node *node, const std::string &service_name, CallbackMsg<TReq, TRep> callback, bool managed = true, bool notify_graph = true);

    /*!
     * \brief Construct an ServiceServer child of the specified Node, using a function as a callback (message class + raw extra parts)
     */
    template<class TReq, class TRep>
    ServiceServer(Node *node, const std::string &service_name, CallbackMsgParts<TReq, TRep> callback, bool managed = true, bool notify_graph = true);

    /*!
     * \brief Construct an ServiceServer child of the specified Node, using a function ptr as a callback (raw without type)
     */
    ServiceServer(Node *node, const std::string &service_name, void (*callback)(const std::string&, std::string&), bool managed = true, bool notify_graph = true);

    /*!
     * \brief Construct an ServiceServer child of the specified Node, using a function ptr as a callback (raw with type)
     */
    ServiceServer(Node *node, const std::string &service_name, void (*callback)(const std::string&, const std::string&, std::string&, std::string&), bool managed = true, bool notify_graph = true);

    /*!
     * \brief Construct an ServiceServer child of the specified Node, using a function ptr as a callback (raw message parts)
     */
    ServiceServer(Node *node, const std::string &service_name, void (*callback)(const std::vector<b0::message::MessagePart>&, std::vector<b0::message::MessagePart>&), bool managed = true, bool notify_graph = true);

    /*!
     * \brief Construct an ServiceServer child of the specified Node, using a function ptr as a callback (message class)
     */
    template<class TReq, class TRep>
    ServiceServer(Node *node, const std::string &service_name, void (*callback)(const TReq&, TRep&), bool managed = true, bool notify_graph = true);

    /*!
     * \brief Construct an ServiceServer child of the specified Node, using a function ptr as a callback (message class and raw extra parts)
     */
    template<class TReq, class TRep>
    ServiceServer(Node *node, const std::string &service_name, void (*callback)(const TReq&, const std::vector<b0::message::MessagePart>&, TRep&, std::vector<b0::message::MessagePart>&), bool managed = true, bool notify_graph = true);

    /*!
     * \brief Construct an ServiceServer child of the specified Node, using a method ptr as a callback (raw without type)
     */
    template<class T>
    ServiceServer(Node *node, const std::string &service_name, void (T::*callback)(const std::string&, std::string&), T *obj, bool managed = true, bool notify_graph = true);

    /*!
     * \brief Construct an ServiceServer child of the specified Node, using a method ptr as a callback (raw with type)
     */
    template<class T>
    ServiceServer(Node *node, const std::string &service_name, void (T::*callback)(const std::string&, const std::string&, std::string&, std::string&), T *obj, bool managed = true, bool notify_graph = true);

    /*!
     * \brief Construct an ServiceServer child of the specified Node, using a method ptr as a callback (raw message parts)
     */
    template<class T>
    ServiceServer(Node *node, const std::string &service_name, void (T::*callback)(const std::vector<b0::message::MessagePart>&, std::vector<b0::message::MessagePart>&), T *obj, bool managed = true, bool notify_graph = true);

    /*!
     * \brief Construct an ServiceServer child of the specified Node, using a method ptr as a callback (message class)
     */
    template<class T, class TReq, class TRep>
    ServiceServer(Node *node, const std::string &service_name, void (T::*callback)(const TReq&, TRep&), T *obj, bool managed = true, bool notify_graph = true);

    /*!
     * \brief Construct an ServiceServer child of the specified Node, using a method ptr as a callback (message class + raw extra parts)
     */
    template<class T, class TReq, class TRep>
    ServiceServer(Node *node, const std::string &service_name, void (T::*callback)(const TReq&, const std::vector<b0::message::MessagePart>&, TRep&, std::vector<b0::message::MessagePart>&), T *obj, bool managed = true, bool notify_graph = true);

    /*!
     * \brief ServiceServer destructor
     */
    virtual ~ServiceServer();

    /*!
     * \brief Log a message using node's logger, prepending this service server informations
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
     * \brief Poll and read incoming messages, and dispatch them (called by b0::Node::spinOnce())
     */
    virtual void spinOnce() override;

    /*!
     * \brief Return the name of this server's service
     */
    std::string getServiceName();

    /*!
     * \brief Bind to an additional address
     */
    virtual void bind(const std::string &address);

protected:
    /*!
     * \brief Bind socket to the address
     */
    virtual void bind();

    /*!
     * \brief Unbind socket from the address
     */
    virtual void unbind();

    /*!
     * \brief Announce service to resolver
     */
    virtual void announce();

    //! The ZeroMQ address to bind the service socket on
    std::string bind_addr_;

    //! If false this socket will not send announcement to resolv (i.e. it will be "invisible")
    const bool notify_graph_;

    /*!
     * \brief Callback which will be called when a new message is read from the socket (raw)
     */
    CallbackRaw callback_;

    /*!
     * \brief Callback which will be called when a new message is read from the socket (raw with type)
     */
    CallbackRawType callback_with_type_;

    /*!
     * \brief Callback which will be called when a new message is read from the socket (raw multipart)
     */
    CallbackParts callback_multipart_;
};

template<class TReq, class TRep>
ServiceServer::ServiceServer(Node *node, const std::string &service_name, CallbackMsg<TReq, TRep> callback, bool managed, bool notify_graph)
    : ServiceServer(node, service_name,
            static_cast<CallbackParts>([&, callback](const std::vector<b0::message::MessagePart> &reqparts, std::vector<b0::message::MessagePart> &repparts) {
                TReq req; TRep rep;
                parse(req, reqparts[0].payload, reqparts[0].content_type);
                callback(req, rep);
                repparts.resize(1);
                serialize(rep, repparts[0].payload, repparts[0].content_type);
            }), managed, notify_graph)
{}

template<class TReq, class TRep>
ServiceServer::ServiceServer(Node *node, const std::string &service_name, CallbackMsgParts<TReq, TRep> callback, bool managed, bool notify_graph)
    : ServiceServer(node, service_name,
            static_cast<CallbackParts>([&, callback](const std::vector<b0::message::MessagePart> &reqparts, std::vector<b0::message::MessagePart> &repparts) {
                std::vector<b0::message::MessagePart> reqparts1(reqparts);
                TReq req;
                parse(req, reqparts1[0].payload, reqparts1[0].content_type);
                reqparts1.erase(reqparts1.begin());
                TRep rep;
                std::vector<b0::message::MessagePart> repparts1;
                callback(req, reqparts1, rep, repparts);
                b0::message::MessagePart reppart0;
                serialize(rep, reppart0.payload, reppart0.content_type);
                repparts.insert(repparts.begin(), reppart0);
            }), managed, notify_graph)
{}

template<class TReq, class TRep>
ServiceServer::ServiceServer(Node *node, const std::string &service_name, void (*callback)(const TReq&, TRep&), bool managed, bool notify_graph)
    : ServiceServer(node, service_name, static_cast<CallbackMsg<TReq, TRep> >(callback), managed, notify_graph)
{}

template<class TReq, class TRep>
ServiceServer::ServiceServer(Node *node, const std::string &service_name, void (*callback)(const TReq&, const std::vector<b0::message::MessagePart>&, TRep&, std::vector<b0::message::MessagePart>&), bool managed, bool notify_graph)
    : ServiceServer(node, service_name, static_cast<CallbackMsgParts<TReq, TRep> >(callback), managed, notify_graph)
{}

template<class T>
ServiceServer::ServiceServer(Node *node, const std::string &service_name, void (T::*callback)(const std::string&, std::string&), T *obj, bool managed, bool notify_graph)
    : ServiceServer(node, service_name, static_cast<CallbackRaw>(boost::bind(callback, obj, _1, _2)), managed, notify_graph)
{}

template<class T>
ServiceServer::ServiceServer(Node *node, const std::string &service_name, void (T::*callback)(const std::string&, const std::string&, std::string&, std::string&), T *obj, bool managed, bool notify_graph)
    : ServiceServer(node, service_name, static_cast<CallbackRawType>(boost::bind(callback, obj, _1, _2, _3, _4)), managed, notify_graph)
{}

template<class T>
ServiceServer::ServiceServer(Node *node, const std::string &service_name, void (T::*callback)(const std::vector<b0::message::MessagePart>&, std::vector<b0::message::MessagePart>&), T *obj, bool managed, bool notify_graph)
    : ServiceServer(node, service_name, static_cast<CallbackParts>(boost::bind(callback, obj, _1, _2)), managed, notify_graph)
{}

template<class T, class TReq, class TRep>
ServiceServer::ServiceServer(Node *node, const std::string &service_name, void (T::*callback)(const TReq&, TRep&), T *obj, bool managed, bool notify_graph)
    : ServiceServer(node, service_name, static_cast<CallbackMsg<TReq, TRep> >(boost::bind(callback, obj, _1, _2)), managed, notify_graph)
{}

template<class T, class TReq, class TRep>
ServiceServer::ServiceServer(Node *node, const std::string &service_name, void (T::*callback)(const TReq&, const std::vector<b0::message::MessagePart>&, TRep&, std::vector<b0::message::MessagePart>&), T *obj, bool managed, bool notify_graph)
    : ServiceServer(node, service_name, static_cast<CallbackMsgParts<TReq, TRep> >(boost::bind(callback, obj, _1, _2)), managed, notify_graph)
{}

} // namespace b0

#endif // B0__SERVICE_SERVER_H__INCLUDED
