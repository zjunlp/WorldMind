#include <string>
#include <vector>
#include <map>

#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <b0/resolver/resolver.h>
#include <b0/resolver/client.h>
#include <b0/logger/logger.h>
#include <b0/utils/env.h>
#include <b0/utils/thread_name.h>

#include <zmq.hpp>

namespace b0
{

namespace resolver
{

ResolverServiceServer::ResolverServiceServer(Resolver *resolver)
    : ServiceServer(resolver, "resolv", &Resolver::handle, resolver, true, false),
      resolver_(resolver)
{
    setPort(-1);
}

void ResolverServiceServer::announce()
{
    b0::message::resolv::AnnounceServiceRequest rq;
    rq.node_name = node_.getName();
    rq.service_name = name_;
    rq.sock_addr = remote_addr_;
    b0::message::resolv::AnnounceServiceResponse rsp;
    resolver_->handleAnnounceService(rq, rsp);
    resolver_->onNodeServiceOfferStart(resolver_->getName(), name_);
}

void ResolverServiceServer::setPort(int port)
{
    if(port == -1)
        port_ = b0::env::getInt("B0_RESOLVER_PORT", 22000);
    else
        port_ = port;
}

int ResolverServiceServer::port() const
{
    return port_;
}

Resolver::Resolver()
    : Node("resolver"),
      resolv_server_(this),
      graph_pub_(this, "graph", true, false),
      minimum_heartbeat_interval_resolver_(5000000)
{
}

Resolver::~Resolver()
{
    heartbeat_sweeper_thread_.interrupt();
    pub_proxy_thread_.interrupt();
    //pub_proxy_thread_.join(); // FIXME: this makes the process hang on quit
}

void Resolver::init()
{
    setResolverAddress(address(hostname(), resolv_server_.port()));

    // setup XPUB-XSUB proxy addresses
    // those will be sent to nodes in response to announce
    int xsub_proxy_port_ = freeTCPPort();
    xsub_proxy_addr_ = address(hostname(), xsub_proxy_port_);
    trace("XSUB address is %s", xsub_proxy_addr_);
    int xpub_proxy_port_ = freeTCPPort();
    xpub_proxy_addr_ = address(hostname(), xpub_proxy_port_);
    trace("XPUB address is %s", xpub_proxy_addr_);
    // run XPUB-XSUB proxy:
    pub_proxy_thread_ = boost::thread(&Resolver::pubProxy, this, xsub_proxy_port_, xpub_proxy_port_);

    Node::init();

    resolv_server_.bind(address("*", resolv_server_.port()));

    // run heartbeat sweeper (to detect when nodes go offline):
    if(minimum_heartbeat_interval_resolver_ > 0)
        heartbeat_sweeper_thread_ = boost::thread(&Resolver::heartbeatSweeper, this);

    // we have to manually notify that 'resolver' is publishing on the 'graph' topic,
    // because the graph_pub_ socket doesn't send graph notify:
    // (has to be disabled because resolver is a special kind of node)
    onNodeTopicPublishStart(getName(), graph_pub_.getTopicName());

    info("Ready.");
}

void Resolver::cleanup()
{
    Node::cleanup();

    // stop auxiliary threads
    if(minimum_heartbeat_interval_resolver_ > 0)
        heartbeat_sweeper_thread_.interrupt();
    pub_proxy_thread_.interrupt(); // XXX: this will have no effect; anyway we'll use
                                   //      each time different port numbers, so, alas.
}

std::string Resolver::getXPUBSocketAddress() const
{
    return xpub_proxy_addr_;
}

std::string Resolver::getXSUBSocketAddress() const
{
    return xsub_proxy_addr_;
}

void Resolver::announceNode()
{
    // directly route this call to the handler, otherwise it will cause a deadlock
    b0::message::resolv::AnnounceNodeRequest rq;
    rq.host_id = hostname();
    rq.process_id = pid();
    rq.node_name = getName();
    b0::message::resolv::AnnounceNodeResponse rsp;
    handleAnnounceNode(rq, rsp);

    if(logger::Logger *p_logger = dynamic_cast<logger::Logger*>(p_logger_))
        p_logger->connect(xsub_proxy_addr_);
}

void Resolver::notifyShutdown()
{
    // directly route this call to the handler, otherwise it will cause a deadlock
#if 0
    b0::message::resolv::ShutdownNodeRequest rq;
    rq.node_name = getName();
    b0::message::resolv::ShutdownNodeResponse rsp;
    handleShutdownNode(rq, rsp);
#else
    // nothing to do really
#endif
}

void Resolver::startHeartbeatThread()
{
}

void Resolver::stopHeartbeatThread()
{
}

void Resolver::spinOnce()
{
    try
    {
        Node::spinOnce();
    }
    catch(std::exception &ex)
    {
        error("Exception in Resolver::spinOnce(): %s", ex.what());
    }
}

void Resolver::onNodeConnected(std::string name)
{
}

void Resolver::onNodeDisconnected(std::string name)
{
    resolver::NodeEntry *e = nodeByName(name);

    for(resolver::ServiceEntry *s : e->services)
        services_by_name_.erase(s->name);
    nodes_by_name_.erase(name);

    std::set<std::pair<std::string, std::string> > npt, nst, nos, nus;

    for(auto x : node_publishes_topic_)
        if(x.first == name)
            npt.insert(x);

    for(auto x : node_subscribes_topic_)
        if(x.first == name)
            nst.insert(x);

    for(auto x : node_offers_service_)
        if(x.first == name)
            nos.insert(x);

    for(auto x : node_uses_service_)
        if(x.first == name)
            nus.insert(x);

    for(auto x : npt) onNodeTopicPublishStop(x.first, x.second);
    for(auto x : nst) onNodeTopicSubscribeStop(x.first, x.second);
    for(auto x : nos) onNodeServiceOfferStop(x.first, x.second);
    for(auto x : nus) onNodeServiceUseStop(x.first, x.second);

    onGraphChanged();
}

void Resolver::onNodeTopicPublishStart(std::string node_name, std::string topic_name)
{
    info("Graph: node '%s' publishes on topic '%s'", node_name, topic_name);
    node_publishes_topic_.insert(std::make_pair(node_name, topic_name));
}

void Resolver::onNodeTopicPublishStop(std::string node_name, std::string topic_name)
{
    info("Graph: node '%s' stops publishing on topic '%s'", node_name, topic_name);
    node_publishes_topic_.erase(std::make_pair(node_name, topic_name));
}

void Resolver::onNodeTopicSubscribeStart(std::string node_name, std::string topic_name)
{
    info("Graph: node '%s' subscribes to topic '%s'", node_name, topic_name);
    node_subscribes_topic_.insert(std::make_pair(node_name, topic_name));
}

void Resolver::onNodeTopicSubscribeStop(std::string node_name, std::string topic_name)
{
    info("Graph: node '%s' stops subscribing to topic '%s'", node_name, topic_name);
    node_subscribes_topic_.erase(std::make_pair(node_name, topic_name));
}

void Resolver::onNodeServiceOfferStart(std::string node_name, std::string service_name)
{
    info("Graph: node '%s' offers service '%s'", node_name, service_name);
    node_offers_service_.insert(std::make_pair(node_name, service_name));
}

void Resolver::onNodeServiceOfferStop(std::string node_name, std::string service_name)
{
    info("Graph: node '%s' stops offering service '%s'", node_name, service_name);
    node_offers_service_.erase(std::make_pair(node_name, service_name));
}

void Resolver::onNodeServiceUseStart(std::string node_name, std::string service_name)
{
    info("Graph: node '%s' connects to service '%s'", node_name, service_name);
    node_uses_service_.insert(std::make_pair(node_name, service_name));
}

void Resolver::onNodeServiceUseStop(std::string node_name, std::string service_name)
{
    info("Graph: node '%s' disconnects from service '%s'", node_name, service_name);
    node_uses_service_.erase(std::make_pair(node_name, service_name));
}

void Resolver::pubProxy(int xsub_proxy_port, int xpub_proxy_port)
{
    set_thread_name("XPROXY");
    b0::logger::LocalLogger logger(this);
    logger.trace("XPROXY: started");

    zmq::context_t &context_ = *reinterpret_cast<zmq::context_t*>(getContext());

    zmq::socket_t proxy_in_sock_(context_, ZMQ_XSUB);
    std::string xsub_proxy_addr = address(xsub_proxy_port);
    proxy_in_sock_.bind(xsub_proxy_addr);

    zmq::socket_t proxy_out_sock_(context_, ZMQ_XPUB);
    std::string xpub_proxy_addr = address(xpub_proxy_port);
    proxy_out_sock_.bind(xpub_proxy_addr);

    try
    {
#ifdef __GNUC__
        zmq::proxy(static_cast<void*>(proxy_in_sock_), static_cast<void*>(proxy_out_sock_), nullptr);
#else
        zmq::proxy(proxy_in_sock_, proxy_out_sock_, nullptr);
#endif
    }
    catch(zmq::error_t &ex)
    {
        if(getState() == Ready)
            logger.error("XPROXY: %s", ex.what());
    }

    logger.trace("XPROXY: finished");
}

bool Resolver::nodeNameExists(std::string name)
{
    return name == "node" || nodes_by_name_.find(name) != nodes_by_name_.end();
}

void Resolver::setResolverPort(int port)
{
    return resolv_server_.setPort(port);
}

std::string Resolver::address(std::string host, int port)
{
    boost::format f("tcp://%s:%d");
    return (f % host % port).str();
}

std::string Resolver::address(int port)
{
    return address("*", port);
}

resolver::NodeEntry * Resolver::nodeByName(std::string node_name)
{
    auto it = nodes_by_name_.find(node_name);
    return it == nodes_by_name_.end() ? 0 : it->second;
}

resolver::ServiceEntry * Resolver::serviceByName(std::string service_name)
{
    auto it = services_by_name_.find(service_name);
    return it == services_by_name_.end() ? 0 : it->second;
}

void Resolver::heartbeat(resolver::NodeEntry *node_entry)
{
    node_entry->last_heartbeat = boost::posix_time::second_clock::local_time();
}

void Resolver::handle(const b0::message::resolv::Request &rq, b0::message::resolv::Response &rsp)
{
#define MAP_METHOD(m, f, t) \
    if(rq.f) { \
        if(t) trace("Received a " #m "Request"); \
        rsp.f.emplace(); \
        handle##m(*rq.f, *rsp.f); \
    }
    MAP_METHOD(AnnounceNode, announce_node, 1)
    MAP_METHOD(ShutdownNode, shutdown_node, 1)
    MAP_METHOD(AnnounceService, announce_service, 1)
    MAP_METHOD(ResolveService, resolve_service, 1)
    MAP_METHOD(Heartbeat, heartbeat, 0)
    MAP_METHOD(NodeTopic, node_topic, 1)
    MAP_METHOD(NodeService, node_service, 1)
    MAP_METHOD(GetGraph, get_graph, 1)
#undef MAP_METHOD
}

std::string Resolver::makeUniqueNodeName(std::string nodeName)
{
    if(nodeName == "") nodeName = "node";
    std::string uniqueNodeName = nodeName;
    for(int i = 1; true; i++)
    {
        if(!nodeNameExists(uniqueNodeName)) break;
        uniqueNodeName = (boost::format("%s-%d") % nodeName % i).str();
    }
    return uniqueNodeName;
}

void Resolver::handleAnnounceNode(const b0::message::resolv::AnnounceNodeRequest &rq, b0::message::resolv::AnnounceNodeResponse &rsp)
{
    std::string nodeName = makeUniqueNodeName(rq.node_name);
    resolver::NodeEntry *e = new resolver::NodeEntry;
    e->host_id = rq.host_id;
    e->process_id = rq.process_id;
    e->name = nodeName;
    heartbeat(e);
    nodes_by_name_[nodeName] = e;
    onNodeConnected(nodeName);
    onGraphChanged();
    rsp.node_name = e->name;
    rsp.xsub_sock_addr = xsub_proxy_addr_;
    rsp.xpub_sock_addr = xpub_proxy_addr_;
    rsp.minimum_heartbeat_interval = minimum_heartbeat_interval_resolver_;
    rsp.ok = true;
    info("New node has joined: '%s'", e->name);
}

void Resolver::handleShutdownNode(const b0::message::resolv::ShutdownNodeRequest &rq, b0::message::resolv::ShutdownNodeResponse &rsp)
{
    resolver::NodeEntry *ne = nodeByName(rq.node_name);
    if(!ne)
    {
        rsp.ok = false;
        error("Invalid node name: %s", rq.node_name);
        return;
    }
    std::string node_name = ne->name;
    onNodeDisconnected(node_name);
    delete ne;
    rsp.ok = true;
    info("Node '%s' has left", node_name);
}

void Resolver::handleAnnounceService(const b0::message::resolv::AnnounceServiceRequest &rq, b0::message::resolv::AnnounceServiceResponse &rsp)
{
    resolver::NodeEntry *ne = nodeByName(rq.node_name);
    if(!ne)
    {
        rsp.ok = false;
        error("Invalid node name: %s", rq.node_name);
        return;
    }
    if(serviceByName(rq.service_name))
    {
        rsp.ok = false;
        error("A service with name '%s' already exists", rq.service_name);
        return;
    }
    resolver::ServiceEntry *se = new resolver::ServiceEntry;
    se->node = ne;
    se->name = rq.service_name;
    se->addr = rq.sock_addr;
    services_by_name_[se->name] = se;
    ne->services.push_back(se);
    //onNodeNewService(...);
    rsp.ok = true;
    trace("Node '%s' announced service '%s' (%s)", ne->name, rq.service_name, rq.sock_addr);
}

void Resolver::handleResolveService(const b0::message::resolv::ResolveServiceRequest &rq, b0::message::resolv::ResolveServiceResponse &rsp)
{
    auto it = services_by_name_.find(rq.service_name);
    if(it == services_by_name_.end())
    {
        rsp.sock_addr = "";
        rsp.ok = false;
        error("Failed to resolve service '%s'", rq.service_name);
        return;
    }
    resolver::ServiceEntry *se = it->second;
    trace("Resolution: '%s' -> %s", rq.service_name, se->addr);
    rsp.ok = true;
    rsp.sock_addr = se->addr;
}

void Resolver::handleHeartbeat(const b0::message::resolv::HeartbeatRequest &rq, b0::message::resolv::HeartbeatResponse &rsp)
{
    if(rq.node_name == "resolver")
    {
        // a HeartbeatRequest from "resolver" means to actually perform
        // the detection and purging of dead nodes
        std::set<std::string> nodes_shutdown;
        for(auto i = nodes_by_name_.begin(); i != nodes_by_name_.end(); ++i)
        {
            resolver::NodeEntry *e = i->second;
            bool is_alive = (boost::posix_time::second_clock::local_time() - e->last_heartbeat) < boost::posix_time::microseconds{minimum_heartbeat_interval_resolver_};
            if(!is_alive && e->name != this->getName())
                nodes_shutdown.insert(e->name);
        }
        if(minimum_heartbeat_interval_resolver_ > 0)
        {
            for(auto node_name : nodes_shutdown)
            {
                info("Node '%s' disconnected due to timeout.", node_name);
                resolver::NodeEntry *e = nodeByName(node_name);
                onNodeDisconnected(node_name);
                delete e;
            }
        }
    }
    else
    {
        resolver::NodeEntry *ne = nodeByName(rq.node_name);
        if(!ne)
        {
            rsp.ok = false;
            error("Received a heartbeat from an invalid node name: %s", rq.node_name);
            return;
        }
        heartbeat(ne);
    }
    rsp.ok = true;
    rsp.time_usec = hardwareTimeUSec();
}

void Resolver::handleNodeTopic(const b0::message::graph::NodeTopicRequest &req, b0::message::graph::NodeTopicResponse &resp)
{
    size_t old_sz1 = node_publishes_topic_.size(), old_sz2 = node_subscribes_topic_.size();
    if(req.reverse)
    {
        if(req.active)
            onNodeTopicSubscribeStart(req.node_name, req.topic_name);
        else
            onNodeTopicSubscribeStop(req.node_name, req.topic_name);
    }
    else
    {
        if(req.active)
            onNodeTopicPublishStart(req.node_name, req.topic_name);
        else
            onNodeTopicPublishStop(req.node_name, req.topic_name);
    }
    if(old_sz1 != node_publishes_topic_.size() || old_sz2 != node_subscribes_topic_.size())
        onGraphChanged();
}

void Resolver::handleNodeService(const b0::message::graph::NodeServiceRequest &req, b0::message::graph::NodeServiceResponse &resp)
{
    size_t old_sz1 = node_offers_service_.size(), old_sz2 = node_uses_service_.size();
    if(req.reverse)
    {
        if(req.active)
            onNodeServiceUseStart(req.node_name, req.service_name);
        else
            onNodeServiceUseStop(req.node_name, req.service_name);
    }
    else
    {
        if(req.active)
            onNodeServiceOfferStart(req.node_name, req.service_name);
        else
            onNodeServiceOfferStop(req.node_name, req.service_name);
    }
    if(old_sz1 != node_offers_service_.size() || old_sz2 != node_uses_service_.size())
        onGraphChanged();
}

void Resolver::handleGetGraph(const b0::message::graph::GetGraphRequest &req, b0::message::graph::GetGraphResponse &resp)
{
    getGraph(resp.graph);
}

void Resolver::getGraph(b0::message::graph::Graph &graph)
{
    for(auto x : nodes_by_name_)
    {
        b0::message::graph::GraphNode n;
        n.host_id = x.second->host_id;
        n.process_id = x.second->process_id;
        n.node_name = x.second->name;
        graph.nodes.push_back(n);
    }
    for(auto x : node_publishes_topic_)
    {
        b0::message::graph::GraphLink l;
        l.node_name = x.first;
        l.other_name = x.second;
        l.reversed = false;
        graph.node_topic.push_back(l);
    }
    for(auto x : node_subscribes_topic_)
    {
        b0::message::graph::GraphLink l;
        l.node_name = x.first;
        l.other_name = x.second;
        l.reversed = true;
        graph.node_topic.push_back(l);
    }
    for(auto x : node_offers_service_)
    {
        b0::message::graph::GraphLink l;
        l.node_name = x.first;
        l.other_name = x.second;
        l.reversed = false;
        graph.node_service.push_back(l);
    }
    for(auto x : node_uses_service_)
    {
        b0::message::graph::GraphLink l;
        l.node_name = x.first;
        l.other_name = x.second;
        l.reversed = true;
        graph.node_service.push_back(l);
    }
}

void Resolver::onGraphChanged()
{
    b0::message::graph::Graph g;
    getGraph(g);
    graph_pub_.publish(g);
}

void Resolver::heartbeatSweeper()
{
    set_thread_name("HBsweep");
    b0::logger::LocalLogger logger(this);
    logger.trace("HBsweep: started");

    while(!shutdownRequested())
    {
        resolver::Client resolv_cli(this);
        resolv_cli.init();

        try
        {
            while(!shutdownRequested())
            {
                // send a heartbeat to resolv itself trigger the sweeping:
                resolv_cli.sendHeartbeat(nullptr);
                sleepUSec(minimum_heartbeat_interval_resolver_ / 3);
            }

            resolv_cli.cleanup();
        }
        catch(std::exception &ex)
        {
            if(getState() == Ready)
                logger.error("HBsweep: %s", ex.what());
        }
    }

    logger.trace("HBsweep: finished");
}

void Resolver::setMinimumHeartbeatInterval(int64_t interval)
{
    minimum_heartbeat_interval_resolver_ = interval;
}

} // namespace resolver

} // namespace b0

