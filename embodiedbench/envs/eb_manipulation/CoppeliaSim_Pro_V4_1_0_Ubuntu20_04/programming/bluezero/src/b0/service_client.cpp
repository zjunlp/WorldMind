#include <b0/service_client.h>
#include <b0/node.h>

#include <zmq.hpp>

namespace b0
{

ServiceClient::ServiceClient(Node *node, const std::string &service_name, bool managed, bool notify_graph)
    : Socket(node, ZMQ_REQ, service_name, managed),
      notify_graph_(notify_graph)
{
}

ServiceClient::~ServiceClient()
{
}

void ServiceClient::log(logger::Level level, const std::string &message) const
{
    boost::format fmt("ServiceClient(%s): %s");
    Socket::log(level, (fmt % name_ % message).str());
}

void ServiceClient::init()
{
    if(Global::getInstance().remapServiceName(getNode(), orig_name_, name_))
        info("Service name '%s' remapped to '%s'", orig_name_, name_);

    resolve();
    connect();

    if(notify_graph_)
        node_.notifyService(name_, true, true);
}

void ServiceClient::cleanup()
{
    disconnect();

    if(notify_graph_)
        node_.notifyService(name_, true, false);
}

std::string ServiceClient::getServiceName()
{
    return name_;
}

void ServiceClient::call(const std::string &req, std::string &rep)
{
    writeRaw(req);
    readRaw(rep);
}

void ServiceClient::call(const std::string &req, const std::string &reqtype, std::string &rep, std::string &reptype)
{
    writeRaw(req, reqtype);
    readRaw(rep, reptype);
}

void ServiceClient::call(const std::vector<b0::message::MessagePart> &reqparts, std::vector<b0::message::MessagePart> &repparts)
{
    writeRaw(reqparts);
    readRaw(repparts);
}

void ServiceClient::resolve()
{
    if(!remote_addr_.empty())
    {
        debug("Skipping resolution because remote address (%s) was given", remote_addr_);
        return;
    }

    node_.resolveService(name_, remote_addr_);

    trace("Resolved address: %s", remote_addr_);
}

void ServiceClient::connect()
{
    trace("Connecting to %s...", remote_addr_);
    Socket::connect(remote_addr_);
}

void ServiceClient::disconnect()
{
    trace("Disconnecting from %s...", remote_addr_);
    Socket::disconnect(remote_addr_);
}

} // namespace b0

