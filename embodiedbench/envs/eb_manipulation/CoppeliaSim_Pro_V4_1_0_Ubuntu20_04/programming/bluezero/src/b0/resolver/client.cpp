#include <b0/resolver/client.h>
#include <b0/message/resolv/request.h>
#include <b0/message/resolv/response.h>
#include <b0/node.h>
#include <b0/exceptions.h>
#include <b0/utils/env.h>

#include <zmq.hpp>

namespace b0
{

namespace resolver
{

Client::Client(b0::Node *node)
    : ServiceClient(node, "resolv", false, false),
      announce_timeout_(-1)
{
    if(!node)
        throw exception::Exception("node cannot be null");

    std::string resolver_addr = b0::env::get("B0_RESOLVER", "tcp://localhost:22000");
    setRemoteAddress(resolver_addr);
}

Client::~Client()
{
}

void Client::setAnnounceTimeout(int timeout)
{
    announce_timeout_ = timeout;
}

void Client::announceNode(const std::string &host_id, int process_id, std::string &node_name, std::string &xpub_sock_addr, std::string &xsub_sock_addr, int64_t &minimum_heartbeat_interval)
{
    int old_timeout = getReadTimeout();
    setReadTimeout(announce_timeout_);

    trace("Announcing node '%s' to resolver...", node_name);
    b0::message::resolv::Request rq0;
    rq0.announce_node.emplace();
    b0::message::resolv::AnnounceNodeRequest &rq = *rq0.announce_node;
    rq.host_id = host_id;
    rq.process_id = process_id;
    rq.node_name = node_name;

    b0::message::resolv::Response rsp0;
    rsp0.announce_node.emplace();
    b0::message::resolv::AnnounceNodeResponse &rsp = *rsp0.announce_node;
    trace("Waiting for response from resolver...");
    call(rq0, rsp0);

    setReadTimeout(old_timeout);

    if(!rsp.ok)
        throw exception::Exception("announceNode failed");

    if(node_name != rsp.node_name)
    {
        warn("Warning: resolver changed this node name to '%s'", rsp.node_name);
    }
    node_name = rsp.node_name;

    xpub_sock_addr = rsp.xpub_sock_addr;
    trace("Proxy's XPUB socket address: %s", xpub_sock_addr);

    xsub_sock_addr = rsp.xsub_sock_addr;
    trace("Proxy's XSUB socket address: %s", xsub_sock_addr);

    minimum_heartbeat_interval = rsp.minimum_heartbeat_interval;
}

void Client::notifyShutdown()
{
    b0::message::resolv::Request rq0;
    rq0.shutdown_node.emplace();
    b0::message::resolv::ShutdownNodeRequest &rq = *rq0.shutdown_node;
    rq.node_name = node_.getName();

    b0::message::resolv::Response rsp0;
    rsp0.shutdown_node.emplace();
    b0::message::resolv::ShutdownNodeResponse &rsp = *rsp0.shutdown_node;
    call(rq0, rsp0);

    if(!rsp.ok)
        throw exception::Exception("notifyShutdown failed");
}

void Client::sendHeartbeat(int64_t *time_usec)
{
    b0::message::resolv::Request rq0;
    rq0.heartbeat.emplace();
    b0::message::resolv::HeartbeatRequest &rq = *rq0.heartbeat;
    rq.node_name = node_.getName();
    int64_t sendTime = node_.hardwareTimeUSec();

    b0::message::resolv::Response rsp0;
    rsp0.heartbeat.emplace();
    b0::message::resolv::HeartbeatResponse &rsp = *rsp0.heartbeat;
    call(rq0, rsp0);

    if(!rsp.ok)
        throw exception::Exception("sendHeartbeat failed");

    if(time_usec)
    {
        int64_t recvTime = node_.hardwareTimeUSec();
        int64_t rtt = recvTime - sendTime;
        *time_usec = rsp.time_usec + rtt / 2;
    }
}

void Client::notifyTopic(std::string topic_name, bool reverse, bool active)
{
    b0::message::resolv::Request rq0;
    rq0.node_topic.emplace();
    b0::message::graph::NodeTopicRequest &rq = *rq0.node_topic;
    rq.node_name = node_.getName();
    rq.topic_name = topic_name;
    rq.reverse = reverse;
    rq.active = active;

    b0::message::resolv::Response rsp0;
    rsp0.node_topic.emplace();
    b0::message::graph::NodeTopicResponse &rsp = *rsp0.node_topic;
    call(rq0, rsp0);
}

void Client::notifyService(std::string service_name, bool reverse, bool active)
{
    b0::message::resolv::Request rq0;
    rq0.node_service.emplace();
    b0::message::graph::NodeServiceRequest &rq = *rq0.node_service;
    rq.node_name = node_.getName();
    rq.service_name = service_name;
    rq.reverse = reverse;
    rq.active = active;

    b0::message::resolv::Response rsp0;
    rsp0.node_service.emplace();
    b0::message::graph::NodeServiceResponse &rsp = *rsp0.node_service;
    call(rq0, rsp0);
}

void Client::announceService(std::string name, std::string addr)
{
    b0::message::resolv::Request rq0;
    rq0.announce_service.emplace();
    b0::message::resolv::AnnounceServiceRequest &rq = *rq0.announce_service;
    rq.node_name = node_.getName();
    rq.service_name = name;
    rq.sock_addr = addr;

    b0::message::resolv::Response rsp0;
    rsp0.announce_service.emplace();
    b0::message::resolv::AnnounceServiceResponse &rsp = *rsp0.announce_service;
    call(rq0, rsp0);

    if(!rsp.ok)
        throw exception::Exception("announceService failed");
}

void Client::resolveService(std::string name, std::string &addr)
{
    b0::message::resolv::Request rq0;
    rq0.resolve_service.emplace();
    b0::message::resolv::ResolveServiceRequest &rq = *rq0.resolve_service;
    rq.service_name = name;

    b0::message::resolv::Response rsp0;
    rsp0.resolve_service.emplace();
    b0::message::resolv::ResolveServiceResponse &rsp = *rsp0.resolve_service;
    call(rq0, rsp0);

    if(!rsp.ok)
        throw exception::NameResolutionError(name);

    addr = rsp.sock_addr;
}

void Client::getGraph(b0::message::graph::Graph &graph)
{
    b0::message::resolv::Request rq0;
    rq0.get_graph.emplace();
    b0::message::graph::GetGraphRequest &rq = *rq0.get_graph;

    b0::message::resolv::Response rsp0;
    rsp0.get_graph.emplace();
    b0::message::graph::GetGraphResponse &rsp = *rsp0.get_graph;
    call(rq0, rsp0);

    graph = rsp.graph;
}

} // namespace resolver

} // namespace b0

