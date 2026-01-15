#ifndef B0__SUBSCRIBER_H__INCLUDED
#define B0__SUBSCRIBER_H__INCLUDED

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
 * \brief The subscriber class
 *
 * This class wraps a SUB socket. It will automatically connect to the
 * XPUB socket of the proxy (note: the proxy is started by the resolver node).
 *
 * In order to receive some message, you must set a topic subscription with Subscriber::subscribe.
 * You can set multiple subscription, and the incoming messages will match any of those.
 *
 * The subscription topics are strings and are matched on a prefix basis.
 * (i.e. if the topic of the incoming message is "AAA", a subscription for "A" will match it)
 *
 * \sa b0::Publisher, b0::Subscriber
 */
class Subscriber : public Socket
{
public:
    using logger::LogInterface::log;

    //! \brief Alias for function
    template<typename T> using function = boost::function<T>;

    //! \brief Alias for callback raw without type
    using CallbackRaw = function<void(const std::string&)>;

    //! \brief Alias for callback raw with type
    using CallbackRawType = function<void(const std::string&, const std::string&)>;

    //! \brief Alias for callback raw message parts
    using CallbackParts = function<void(const std::vector<b0::message::MessagePart>&)>;

    //! \brief Alias for callback message class
    template<class TMsg> using CallbackMsg = function<void(const TMsg&)>;

    //! \brief Alias for callback message class + raw extra parts
    template<class TMsg> using CallbackMsgParts = function<void(const TMsg&, const std::vector<b0::message::MessagePart>&)>;

    /*!
     * \brief Construct an Subscriber child of the specified Node without a callback
     */
    Subscriber(Node *node, const std::string &topic_name, bool managed = true, bool notify_graph = true);

    /*!
     * \brief Construct an Subscriber child of the specified Node, optionally using a function as callback (raw without type)
     */
    Subscriber(Node *node, const std::string &topic_name, CallbackRaw callback, bool managed = true, bool notify_graph = true);

    /*!
     * \brief Construct an Subscriber child of the specified Node, using a function as callback (raw with type)
     */
    Subscriber(Node *node, const std::string &topic_name, CallbackRawType callback, bool managed = true, bool notify_graph = true);

    /*!
     * \brief Construct an Subscriber child of the specified Node, using a function as callback (raw message parts)
     */
    Subscriber(Node *node, const std::string &topic_name, CallbackParts callback, bool managed = true, bool notify_graph = true);

    /*!
     * \brief Construct an Subscriber child of the specified Node, using a function as callback (message class)
     */
    template<class TMsg>
    Subscriber(Node *node, const std::string &topic_name, CallbackMsg<TMsg> callback, bool managed = true, bool notify_graph = true);

    /*!
     * \brief Construct an Subscriber child of the specified Node, using a function as callback (message class and raw extra parts)
     */
    template<class TMsg>
    Subscriber(Node *node, const std::string &topic_name, CallbackMsgParts<TMsg> callback, bool managed = true, bool notify_graph = true);

    /*!
     * \brief Construct an Subscriber child of the specified Node, using a function ptr as callback (raw without type)
     */
    Subscriber(Node *node, const std::string &topic_name, void (*callback)(const std::string&), bool managed = true, bool notify_graph = true);

    /*!
     * \brief Construct an Subscriber child of the specified Node, using a function ptr as callback (raw with type)
     */
    Subscriber(Node *node, const std::string &topic_name, void (*callback)(const std::string&, const std::string&), bool managed = true, bool notify_graph = true);

    /*!
     * \brief Construct an Subscriber child of the specified Node, using a function ptr as callback (raw message parts)
     */
    Subscriber(Node *node, const std::string &topic_name, void (*callback)(const std::vector<b0::message::MessagePart>&), bool managed = true, bool notify_graph = true);

    /*!
     * \brief Construct an Subscriber child of the specified Node, using a function ptr as callback (message class)
     */
    template<class TMsg>
    Subscriber(Node *node, const std::string &topic_name, void (*callback)(const TMsg&), bool managed = true, bool notify_graph = true);

    /*!
     * \brief Construct an Subscriber child of the specified Node, using a function ptr as callback (message class and raw extra parts)
     */
    template<class TMsg>
    Subscriber(Node *node, const std::string &topic_name, void (*callback)(const TMsg&, const std::vector<b0::message::MessagePart>&), bool managed = true, bool notify_graph = true);

    /*!
     * \brief Construct an Subscriber child of the specified Node, using a method ptr as callback (raw without type)
     */
    template<class T>
    Subscriber(Node *node, const std::string &topic_name, void (T::*callback)(const std::string&), T *obj, bool managed = true, bool notify_graph = true);

    /*!
     * \brief Construct an Subscriber child of the specified Node, using a method ptr as callback (raw with type)
     */
    template<class T>
    Subscriber(Node *node, const std::string &topic_name, void (T::*callback)(const std::string&, const std::string&), T *obj, bool managed = true, bool notify_graph = true);

    /*!
     * \brief Construct an Subscriber child of the specified Node, using a method ptr as callback (raw message parts)
     */
    template<class T>
    Subscriber(Node *node, const std::string &topic_name, void (T::*callback)(const std::vector<b0::message::MessagePart>&), T *obj, bool managed = true, bool notify_graph = true);

    /*!
     * \brief Construct an Subscriber child of the specified Node, using a method ptr as callback (message class)
     */
    template<class T, class TMsg>
    Subscriber(Node *node, const std::string &topic_name, void (T::*callback)(const TMsg&), T *obj, bool managed = true, bool notify_graph = true);

    /*!
     * \brief Construct an Subscriber child of the specified Node, using a method ptr as callback (message class and raw extra parts)
     */
    template<class T, class TMsg>
    Subscriber(Node *node, const std::string &topic_name, void (T::*callback)(const TMsg&, const std::vector<b0::message::MessagePart>&), T *obj, bool managed = true, bool notify_graph = true);

    /*!
     * \brief Subscriber destructor
     */
    virtual ~Subscriber();

    /*!
     * \brief Log a message using node's logger, prepending this subscriber informations
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
     * \brief Process incoming messages and call callbacks
     */
    virtual void spinOnce() override;

    /*!
     * \brief Return the name of this subscriber's topic
     */
    std::string getTopicName();

protected:
    /*!
     * \brief Connect to the remote address
     */
    virtual void connect();

    /*!
     * \brief Disconnect from the remote address
     */
    virtual void disconnect();

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

template<class TMsg>
Subscriber::Subscriber(Node *node, const std::string &topic_name, CallbackMsg<TMsg> callback, bool managed, bool notify_graph)
    : Subscriber(node, topic_name,
            static_cast<CallbackParts>([&, callback](const std::vector<b0::message::MessagePart> &parts) {
                TMsg msg;
                parse(msg, parts[0].payload, parts[0].content_type);
                callback(msg);
            }), managed, notify_graph)
{}

template<class TMsg>
Subscriber::Subscriber(Node *node, const std::string &topic_name, CallbackMsgParts<TMsg> callback, bool managed, bool notify_graph)
    : Subscriber(node, topic_name,
            static_cast<CallbackParts>([&, callback](const std::vector<b0::message::MessagePart> &parts) {
                std::vector<b0::message::MessagePart> parts1(parts);
                TMsg msg;
                parse(msg, parts1[0].payload, parts1[0].content_type);
                parts1.erase(parts1.begin());
                callback(msg, parts1);
            }), managed, notify_graph)
{}

template<class TMsg>
Subscriber::Subscriber(Node *node, const std::string &topic_name, void (*callback)(const TMsg&), bool managed, bool notify_graph)
    : Subscriber(node, topic_name, static_cast<CallbackMsg<TMsg> >(callback), managed, notify_graph)
{}

template<class TMsg>
Subscriber::Subscriber(Node *node, const std::string &topic_name, void (*callback)(const TMsg&, const std::vector<b0::message::MessagePart>&), bool managed, bool notify_graph)
    : Subscriber(node, topic_name, static_cast<CallbackMsgParts<TMsg> >(callback), managed, notify_graph)
{}

template<class T>
Subscriber::Subscriber(Node *node, const std::string &topic_name, void (T::*callback)(const std::string&), T *obj, bool managed, bool notify_graph)
    : Subscriber(node, topic_name, static_cast<CallbackRaw>(boost::bind(callback, obj, _1)))
{}

template<class T>
Subscriber::Subscriber(Node *node, const std::string &topic_name, void (T::*callback)(const std::string&, const std::string&), T *obj, bool managed, bool notify_graph)
    : Subscriber(node, topic_name, static_cast<CallbackRawType>(boost::bind(callback, obj, _1, _2)))
{}

template<class T>
Subscriber::Subscriber(Node *node, const std::string &topic_name, void (T::*callback)(const std::vector<b0::message::MessagePart>&), T *obj, bool managed, bool notify_graph)
    : Subscriber(node, topic_name, static_cast<CallbackRawType>(boost::bind(callback, obj, _1, _2)))
{}

template<class T, class TMsg>
Subscriber::Subscriber(Node *node, const std::string &topic_name, void (T::*callback)(const TMsg&), T *obj, bool managed, bool notify_graph)
    : Subscriber(node, topic_name, static_cast<CallbackMsg<TMsg> >(boost::bind(callback, obj, _1)))
{}

template<class T, class TMsg>
Subscriber::Subscriber(Node *node, const std::string &topic_name, void (T::*callback)(const TMsg&, const std::vector<b0::message::MessagePart>&), T *obj, bool managed, bool notify_graph)
    : Subscriber(node, topic_name, static_cast<CallbackMsgParts<TMsg> >(boost::bind(callback, obj, _1)))
{}

} // namespace b0

#endif // B0__SUBSCRIBER_H__INCLUDED
