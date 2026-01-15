#ifndef B0__MESSAGE__GRAPH__NODE_TOPIC_REQUEST_H__INCLUDED
#define B0__MESSAGE__GRAPH__NODE_TOPIC_REQUEST_H__INCLUDED

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
 * \brief Sent by node to tell a topic it is publishing onto/subscribing to
 *
 * \mscfile graph-topic.msc
 *
 * \sa NodeTopicResponse, \ref protocol, \ref graph
 */
class NodeTopicRequest : public Message
{
public:
    //! The name of the node
    std::string node_name;

    //! The name of the topic
    std::string topic_name;

    //! If true, node is a subscriber, otherwise a publisher
    bool reverse;

    //! If true, the relationship is starting, otherwise is ending
    bool active;

public:
    std::string type() const override {return "b0.message.graph.NodeTopicRequest";}
};

} // namespace graph

} // namespace message

} // namespace b0

//! \cond HIDDEN_SYMBOLS

namespace spotify
{

namespace json
{

using b0::message::graph::NodeTopicRequest;

template <>
struct default_codec_t<NodeTopicRequest>
{
    static codec::object_t<NodeTopicRequest> codec()
    {
        auto codec = codec::object<NodeTopicRequest>();
        codec.required("node_name", &NodeTopicRequest::node_name);
        codec.required("topic_name", &NodeTopicRequest::topic_name);
        codec.required("reverse", &NodeTopicRequest::reverse);
        codec.required("active", &NodeTopicRequest::active);
        return codec;
    }
};

} // namespace json

} // namespace spotify

//! \endcond

#endif // B0__MESSAGE__GRAPH__NODE_TOPIC_REQUEST_H__INCLUDED
