#ifndef B0__MESSAGE__RESOLV__RESOLV_REQUEST_H__INCLUDED
#define B0__MESSAGE__RESOLV__RESOLV_REQUEST_H__INCLUDED

#include <boost/serialization/string.hpp>

#include <b0/b0.h>
#include <b0/message/message.h>
#include <b0/message/resolv/announce_node_request.h>
#include <b0/message/resolv/shutdown_node_request.h>
#include <b0/message/resolv/announce_service_request.h>
#include <b0/message/resolv/resolve_service_request.h>
#include <b0/message/resolv/heartbeat_request.h>
#include <b0/message/graph/node_topic_request.h>
#include <b0/message/graph/node_service_request.h>
#include <b0/message/graph/get_graph_request.h>

namespace b0
{

namespace message
{

namespace resolv
{

/*!
 * \brief Message sent to resolver
 *
 * \sa Response, \ref protocol
 */
class Request : public Message
{
public:
    //! \brief Message for the AnnounceNodeRequest
    boost::optional<AnnounceNodeRequest> announce_node;

    //! \brief Message for the ShutdownNodeRequest
    boost::optional<ShutdownNodeRequest> shutdown_node;

    //! \brief Message for the AnnounceServiceRequest
    boost::optional<AnnounceServiceRequest> announce_service;

    //! \brief Message for the ResolveServiceRequest
    boost::optional<ResolveServiceRequest> resolve_service;

    //! \brief Message for the HeartbeatRequest
    boost::optional<HeartbeatRequest> heartbeat;

    //! \brief Message for the NodeTopicRequest
    boost::optional<graph::NodeTopicRequest> node_topic;

    //! \brief Message for the NodeServiceRequest
    boost::optional<graph::NodeServiceRequest> node_service;

    //! \brief Message for the GetGraphRequest
    boost::optional<graph::GetGraphRequest> get_graph;

public:
    std::string type() const override {return "b0.message.resolv.Request";}
};

} // namespace resolv

} // namespace message

} // namespace b0

//! \cond HIDDEN_SYMBOLS

namespace spotify
{

namespace json
{

using b0::message::resolv::Request;

template <>
struct default_codec_t<Request>
{
    static codec::object_t<Request> codec()
    {
        auto codec = codec::object<Request>();
        codec.optional("announce_node", &Request::announce_node);
        codec.optional("shutdown_node", &Request::shutdown_node);
        codec.optional("announce_service", &Request::announce_service);
        codec.optional("resolve_service", &Request::resolve_service);
        codec.optional("heartbeat", &Request::heartbeat);
        codec.optional("node_topic", &Request::node_topic);
        codec.optional("node_service", &Request::node_service);
        codec.optional("get_graph", &Request::get_graph);
        return codec;
    }
};

} // namespace json

} // namespace spotify

//! \endcond

#endif // B0__MESSAGE__RESOLV__RESOLV_REQUEST_H__INCLUDED
