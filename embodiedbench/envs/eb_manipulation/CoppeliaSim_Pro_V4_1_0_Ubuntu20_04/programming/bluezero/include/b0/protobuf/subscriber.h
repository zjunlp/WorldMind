#ifndef B0__PROTOBUF__SUBSCRIBER_H__INCLUDED
#define B0__PROTOBUF__SUBSCRIBER_H__INCLUDED

#include <string>

#include <boost/function.hpp>
#include <boost/bind.hpp>

#include <b0/subscriber.h>
#include <b0/protobuf/socket.h>

namespace b0
{

namespace protobuf
{

/*!
 * \brief The subscriber template class
 *
 * This template class specializes b0::AbstractSubscriber to a specific message type,
 * and it implements the spinOnce method as well.
 *
 * Important when using a callback: you must call b0::Node::spin(), or periodically call
 * b0::Node::spinOnce(), otherwise no message will be delivered.
 *
 * Otherwise, you can directly read from the SUB socket, by using Subscriber::read().
 * Note: read operation is blocking. If you do not want to block, use Subscriber::poll() first.
 *
 * \sa b0::Publisher, b0::Subscriber, b0::AbstractPublisher, b0::AbstractSubscriber
 */
template<typename TMsg>
class Subscriber : public b0::Subscriber, public SocketProtobuf
{
public:
    /*!
     * \brief Construct a Subscriber child of a specified Node, with a boost::function as callback
     */
    Subscriber(Node *node, std::string topic_name, boost::function<void(const TMsg&)> callback = 0, bool managed = true, bool notify_graph = true)
        : b0::Subscriber(node, topic_name, managed, notify_graph),
          callback_(callback)
    {
    }

    /*!
     * \brief Construct a Subscriber child of a specified Node, with a method (of the Node subclass) as a callback
     */
    template<class TNode>
    Subscriber(TNode *node, std::string topic_name, void (TNode::*callbackMethod)(const TMsg&), bool managed = true, bool notify_graph = true)
        : Subscriber(node, topic_name, boost::bind(callbackMethod, node, _1), managed, notify_graph)
    {
        // delegate constructor. leave empty
    }

    /*!
     * \brief Construct a Subscriber child of a specified Node, with a method as a callback
     */
    template<class T>
    Subscriber(Node *node, std::string topic_name, void (T::*callbackMethod)(const TMsg&), T *callbackObject, bool managed = true, bool notify_graph = true)
        : Subscriber(node, topic_name, boost::bind(callbackMethod, callbackObject, _1), managed, notify_graph)
    {
        // delegate constructor. leave empty
    }

    /*!
     * \brief Poll and read incoming messages, and dispatch them (called by b0::Node::spinOnce())
     */
    void spinOnce() override
    {
        if(callback_.empty()) return;

        while(poll())
        {
            TMsg msg;
            read(this, msg);
            callback_(msg);
        }
    }
 
protected:
    /*!
     * \brief Callback which will be called when a new message is read from the socket
     */
    boost::function<void(TMsg&)> callback_;
};

} // namespace protobuf

} // namespace b0

#endif // B0__PROTOBUF__SUBSCRIBER_H__INCLUDED
