#ifndef B0__MESSAGE__GRAPH__GET_GRAPH_RESPONSE_H__INCLUDED
#define B0__MESSAGE__GRAPH__GET_GRAPH_RESPONSE_H__INCLUDED

#include <b0/b0.h>
#include <b0/message/message.h>
#include <b0/message/graph/graph.h>

namespace b0
{

namespace message
{

namespace graph
{

/*!
 * \brief Response to GetGraphRequest message
 *
 * \mscfile graph-get.msc
 *
 * \sa GetGraphRequest, \ref protocol, \ref graph
 */
class GetGraphResponse : public Message
{
public:
    //! The graph of the network
    Graph graph;

public:
    std::string type() const override {return "b0.message.graph.GetGraphResponse";}
};

} // namespace graph

} // namespace message

} // namespace b0

//! \cond HIDDEN_SYMBOLS

namespace spotify
{

namespace json
{

using b0::message::graph::GetGraphResponse;

template <>
struct default_codec_t<GetGraphResponse>
{
    static codec::object_t<GetGraphResponse> codec()
    {
        auto codec = codec::object<GetGraphResponse>();
        codec.required("graph", &GetGraphResponse::graph);
        return codec;
    }
};

} // namespace json

} // namespace spotify

//! \endcond

#endif // B0__MESSAGE__GRAPH__GET_GRAPH_RESPONSE_H__INCLUDED
