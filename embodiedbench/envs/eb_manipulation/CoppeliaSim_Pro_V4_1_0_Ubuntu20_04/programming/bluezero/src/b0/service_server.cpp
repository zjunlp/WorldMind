#include <b0/service_server.h>
#include <b0/node.h>

#include <zmq.hpp>

namespace b0
{

ServiceServer::ServiceServer(Node *node, const std::string &service_name, bool managed, bool notify_graph)
    : ServiceServer(node, service_name, CallbackRaw{}, managed, notify_graph)
{
}

ServiceServer::ServiceServer(Node *node, const std::string &service_name, CallbackRaw callback, bool managed, bool notify_graph)
    : Socket(node, ZMQ_REP, service_name, managed),
      notify_graph_(notify_graph),
      bind_addr_(""),
      callback_(callback)
{
}

ServiceServer::ServiceServer(Node *node, const std::string &service_name, CallbackRawType callback, bool managed, bool notify_graph)
    : Socket(node, ZMQ_REP, service_name, managed),
      notify_graph_(notify_graph),
      bind_addr_(""),
      callback_with_type_(callback)
{
}

ServiceServer::ServiceServer(Node *node, const std::string &service_name, CallbackParts callback, bool managed, bool notify_graph)
    : Socket(node, ZMQ_REP, service_name, managed),
      notify_graph_(notify_graph),
      bind_addr_(""),
      callback_multipart_(callback)
{
}

ServiceServer::ServiceServer(Node *node, const std::string &service_name, void (*callback)(const std::string&, std::string&), bool managed, bool notify_graph)
    : ServiceServer(node, service_name, static_cast<CallbackRaw>(callback), managed, notify_graph)
{
}

ServiceServer::ServiceServer(Node *node, const std::string &service_name, void (*callback)(const std::string&, const std::string&, std::string&, std::string&), bool managed, bool notify_graph)
    : ServiceServer(node, service_name, static_cast<CallbackRawType>(callback), managed, notify_graph)
{
}

ServiceServer::ServiceServer(Node *node, const std::string &service_name, void (*callback)(const std::vector<b0::message::MessagePart>&, std::vector<b0::message::MessagePart>&), bool managed, bool notify_graph)
    : ServiceServer(node, service_name, static_cast<CallbackParts>(callback), managed, notify_graph)
{
}

ServiceServer::~ServiceServer()
{
}

void ServiceServer::log(logger::Level level, const std::string &message) const
{
    boost::format fmt("ServiceServer(%s): %s");
    Socket::log(level, (fmt % name_ % message).str());
}

void ServiceServer::init()
{
    if(Global::getInstance().remapServiceName(getNode(), orig_name_, name_))
        info("Service name '%s' remapped to '%s'", orig_name_, name_);

    bind();
    announce();

    if(notify_graph_)
        node_.notifyService(name_, false, true);
}

void ServiceServer::cleanup()
{
    unbind();

    if(notify_graph_)
        node_.notifyService(name_, false, false);
}

void ServiceServer::spinOnce()
{
    if(!callback_ && !callback_with_type_ && !callback_multipart_) return;

    while(poll())
    {
        if(callback_)
        {
            std::string req, rep;
            readRaw(req);
            callback_(req, rep);
            writeRaw(rep);
        }
        if(callback_with_type_)
        {
            std::string req, reqtype, rep, reptype;
            readRaw(req, reqtype);
            callback_with_type_(req, reqtype, rep, reptype);
            writeRaw(rep, reptype);
        }
        if(callback_multipart_)
        {
            std::vector<b0::message::MessagePart> reqparts, repparts;
            readRaw(reqparts);
            callback_multipart_(reqparts, repparts);
            writeRaw(repparts);
        }
    }
}

std::string ServiceServer::getServiceName()
{
    return name_;
}

void ServiceServer::bind()
{
    boost::format fmt("tcp://%s:%d");
    std::string host = node_.hostname();
    int port = node_.freeTCPPort();
    bind_addr_ = (fmt % "*" % port).str();
    remote_addr_ = (fmt % host % port).str();
    Socket::bind(bind_addr_);
    debug("Bound to %s", bind_addr_);
}

void ServiceServer::unbind()
{
    //Socket::unbind(bind_addr_); // FIXME: causes a zmq error on node shutdown
}

void ServiceServer::announce()
{
    trace("Announcing %s to resolver...", remote_addr_);
    node_.announceService(name_, remote_addr_);
}

void ServiceServer::bind(const std::string &address)
{
    Socket::bind(address);
    debug("Bound to additional address %s", address);
}

} // namespace b0

