#ifndef B0__MESSAGE__RESOLV__RESOLV_RESPONSE_H__INCLUDED
#define B0__MESSAGE__RESOLV__RESOLV_RESPONSE_H__INCLUDED

#include <boost/serialization/string.hpp>

#include <b0/b0.h>
#include <b0/message/message.h>
#include <b0/message/resolv/announce_node_response.h>
#include <b0/message/resolv/shutdown_node_response.h>
#include <b0/message/resolv/announce_service_response.h>
#include <b0/message/resolv/resolve_service_response.h>
#include <b0/message/resolv/heartbeat_response.h>
#include <b0/message/graph/node_topic_response.h>
#include <b0/message/graph/node_service_response.h>
#include <b0/message/graph/get_graph_response.h>

namespace b0
{

namespace message
{

namespace resolv
{

/*!
 * \brief Response to Request message
 *
 * \sa Request, \ref protocol
 */
class Response : public Message
{
public:
    //! \brief Message for the AnnounceNodeResponse
    boost::optional<AnnounceNodeResponse> announce_node;

    //! \brief Message for the ShutdownNodeResponse
    boost::optional<ShutdownNodeResponse> shutdown_node;

    //! \brief Message for the AnnounceServiceResponse
    boost::optional<AnnounceServiceResponse> announce_service;

    //! \brief Message for the ResolveServiceResponse
    boost::optional<ResolveServiceResponse> resolve_service;

    //! \brief Message for the HeartbeatResponse
    boost::optional<HeartbeatResponse> heartbeat;

    //! \brief Message for the NodeTopicResponse
    boost::optional<graph::NodeTopicResponse> node_topic;

    //! \brief Message for the NodeServiceResponse
    boost::optional<graph::NodeServiceResponse> node_service;

    //! \brief Message for the GetGraphResponse
    boost::optional<graph::GetGraphResponse> get_graph;

public:
    std::string type() const override {return "b0.message.resolv.Response";}
};

} // namespace resolv

} // namespace message

} // namespace b0

//! \cond HIDDEN_SYMBOLS

namespace spotify
{

namespace json
{

using b0::message::resolv::Response;

template <>
struct default_codec_t<Response>
{
    static codec::object_t<Response> codec()
    {
        auto codec = codec::object<Response>();
        codec.optional("announce_node", &Response::announce_node);
        codec.optional("shutdown_node", &Response::shutdown_node);
        codec.optional("announce_service", &Response::announce_service);
        codec.optional("resolve_service", &Response::resolve_service);
        codec.optional("heartbeat", &Response::heartbeat);
        codec.optional("node_topic", &Response::node_topic);
        codec.optional("node_service", &Response::node_service);
        codec.optional("get_graph", &Response::get_graph);
        return codec;
    }
};

} // namespace json

} // namespace spotify

//! \endcond

#endif // B0__MESSAGE__RESOLV__RESOLV_RESPONSE_H__INCLUDED
