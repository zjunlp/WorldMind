#ifndef B0__MESSAGE__GRAPH__NODE_SERVICE_REQUEST_H__INCLUDED
#define B0__MESSAGE__GRAPH__NODE_SERVICE_REQUEST_H__INCLUDED

#include <boost/serialization/string.hpp>

#include <b0/b0.h>
#include <b0/message/message.h>

namespace b0
{

namespace message
{

namespace graph
{

/*!
 * \brief Sent by node to tell a service it is offering
 *
 * \mscfile graph-service.msc
 *
 * \sa NodeServiceResponse, \ref protocol, \ref graph
 */
class NodeServiceRequest : public Message
{
public:
    //! The name of the node
    std::string node_name;

    //! The name of the service
    std::string service_name;

    //! If true, node is a client, otherwise a server
    bool reverse;

    //! If true, the relationship is starting, otherwise it is ending
    bool active;

public:
    std::string type() const override {return "b0.message.graph.NodeServiceRequest";}
};

} // namespace graph

} // namespace message

} // namespace b0

//! \cond HIDDEN_SYMBOLS

namespace spotify
{

namespace json
{

using b0::message::graph::NodeServiceRequest;

template <>
struct default_codec_t<NodeServiceRequest>
{
    static codec::object_t<NodeServiceRequest> codec()
    {
        auto codec = codec::object<NodeServiceRequest>();
        codec.required("node_name", &NodeServiceRequest::node_name);
        codec.required("service_name", &NodeServiceRequest::service_name);
        codec.required("reverse", &NodeServiceRequest::reverse);
        codec.required("active", &NodeServiceRequest::active);
        return codec;
    }
};

} // namespace json

} // namespace spotify

//! \endcond

#endif // B0__MESSAGE__GRAPH__NODE_SERVICE_REQUEST_H__INCLUDED
