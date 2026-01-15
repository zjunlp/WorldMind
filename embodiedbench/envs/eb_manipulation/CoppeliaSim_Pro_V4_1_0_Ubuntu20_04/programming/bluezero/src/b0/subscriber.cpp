#include <b0/subscriber.h>
#include <b0/node.h>

#include <zmq.hpp>

namespace b0
{

Subscriber::Subscriber(Node *node, const std::string &topic_name, bool managed, bool notify_graph)
    : Subscriber(node, topic_name, CallbackRaw{}, managed, notify_graph)
{
}

Subscriber::Subscriber(Node *node, const std::string &topic_name, CallbackRaw callback, bool managed, bool notify_graph)
    : Socket(node, ZMQ_SUB, topic_name, managed),
      notify_graph_(notify_graph),
      callback_(callback)
{
}

Subscriber::Subscriber(Node *node, const std::string &topic_name, CallbackRawType callback, bool managed, bool notify_graph)
    : Socket(node, ZMQ_SUB, topic_name, managed),
      notify_graph_(notify_graph),
      callback_with_type_(callback)
{
}

Subscriber::Subscriber(Node *node, const std::string &topic_name, CallbackParts callback, bool managed, bool notify_graph)
    : Socket(node, ZMQ_SUB, topic_name, managed),
      notify_graph_(notify_graph),
      callback_multipart_(callback)
{
}

Subscriber::Subscriber(Node *node, const std::string &topic_name, void (*callback)(const std::string&), bool managed, bool notify_graph)
    : Subscriber(node, topic_name, static_cast<CallbackRaw>(callback), managed, notify_graph)
{
}

Subscriber::Subscriber(Node *node, const std::string &topic_name, void (*callback)(const std::string&, const std::string&), bool managed, bool notify_graph)
    : Subscriber(node, topic_name, static_cast<CallbackRawType>(callback), managed, notify_graph)
{
}

Subscriber::Subscriber(Node *node, const std::string &topic_name, void (*callback)(const std::vector<b0::message::MessagePart>&), bool managed, bool notify_graph)
    : Subscriber(node, topic_name, static_cast<CallbackParts>(callback), managed, notify_graph)
{
}

Subscriber::~Subscriber()
{
}

void Subscriber::log(logger::Level level, const std::string &message) const
{
    boost::format fmt("Subscriber(%s): %s");
    Socket::log(level, (fmt % name_ % message).str());
}

void Subscriber::init()
{
    if(Global::getInstance().remapTopicName(getNode(), orig_name_, name_))
        info("Topic name '%s' remapped to '%s'", orig_name_, name_);

    if(remote_addr_.empty())
        remote_addr_ = node_.getXPUBSocketAddress();
    connect();

    if(notify_graph_)
        node_.notifyTopic(name_, true, true);
}

void Subscriber::cleanup()
{
    disconnect();

    if(notify_graph_)
        node_.notifyTopic(name_, true, false);
}

void Subscriber::spinOnce()
{
    if(!callback_ && !callback_with_type_ && !callback_multipart_) return;

    while(poll())
    {
        if(callback_)
        {
            std::string msg;
            readRaw(msg);
            callback_(msg);
        }
        if(callback_with_type_)
        {
            std::string msg, type;
            readRaw(msg, type);
            callback_with_type_(msg, type);
        }
        if(callback_multipart_)
        {
            std::vector<b0::message::MessagePart> parts;
            readRaw(parts);
            callback_multipart_(parts);
        }
    }
}

std::string Subscriber::getTopicName()
{
    return name_;
}

void Subscriber::connect()
{
    trace("Connecting to %s...", remote_addr_);
    Socket::connect(remote_addr_);
    Socket::setsockopt(ZMQ_SUBSCRIBE, name_.data(), name_.size());
}

void Subscriber::disconnect()
{
    trace("Disconnecting from %s...", remote_addr_);
    Socket::setsockopt(ZMQ_UNSUBSCRIBE, name_.data(), name_.size());
    Socket::disconnect(remote_addr_);
}

} // namespace b0

