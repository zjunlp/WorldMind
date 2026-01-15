#ifndef B0__MESSAGE__GRAPH__NODE_TOPIC_RESPONSE_H__INCLUDED
#define B0__MESSAGE__GRAPH__NODE_TOPIC_RESPONSE_H__INCLUDED

#include <b0/b0.h>
#include <b0/message/message.h>

namespace b0
{

namespace message
{

namespace graph
{

/*!
 * \brief Response to NodeTopicRequest message
 *
 * \mscfile graph-topic.msc
 *
 * \sa NodeTopicRequest, \ref protocol, \ref graph
 */
class NodeTopicResponse : public Message
{
public:

public:
    std::string type() const override {return "b0.message.graph.NodeTopicResponse";}
};

} // namespace graph

} // namespace message

} // namespace b0

//! \cond HIDDEN_SYMBOLS

namespace spotify
{

namespace json
{

using b0::message::graph::NodeTopicResponse;

template <>
struct default_codec_t<NodeTopicResponse>
{
    static codec::object_t<NodeTopicResponse> codec()
    {
        auto codec = codec::object<NodeTopicResponse>();
        return codec;
    }
};

} // namespace json

} // namespace spotify

//! \endcond

#endif // B0__MESSAGE__GRAPH__NODE_TOPIC_RESPONSE_H__INCLUDED
