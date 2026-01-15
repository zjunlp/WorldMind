#ifndef B0__MESSAGE__GRAPH__NODE_SERVICE_RESPONSE_H__INCLUDED
#define B0__MESSAGE__GRAPH__NODE_SERVICE_RESPONSE_H__INCLUDED

#include <b0/b0.h>
#include <b0/message/message.h>

namespace b0
{

namespace message
{

namespace graph
{

/*!
 * \brief Response to NodeServiceRequest message
 *
 * \mscfile graph-service.msc
 *
 * \sa NodeServiceRequest, \ref protocol, \ref graph
 */
class NodeServiceResponse : public Message
{
public:

public:
    std::string type() const override {return "b0.message.graph.NodeServiceResponse";}
};

} // namespace graph

} // namespace message

} // namespace b0

//! \cond HIDDEN_SYMBOLS

namespace spotify
{

namespace json
{

using b0::message::graph::NodeServiceResponse;

template <>
struct default_codec_t<NodeServiceResponse>
{
    static codec::object_t<NodeServiceResponse> codec()
    {
        auto codec = codec::object<NodeServiceResponse>();
        return codec;
    }
};

} // namespace json

} // namespace spotify

//! \endcond

#endif // B0__MESSAGE__GRAPH__NODE_SERVICE_RESPONSE_H__INCLUDED
