#ifndef B0__MESSAGE__GRAPH__GET_GRAPH_REQUEST_H__INCLUDED
#define B0__MESSAGE__GRAPH__GET_GRAPH_REQUEST_H__INCLUDED

#include <b0/b0.h>
#include <b0/message/message.h>

namespace b0
{

namespace message
{

namespace graph
{

/*!
 * \brief Sent by node to resolver, for getting the full graph
 *
 * \mscfile graph-get.msc
 *
 * \sa GetGraphResponse, \ref protocol, \ref graph
 */
class GetGraphRequest : public Message
{
public:

public:
    std::string type() const override {return "b0.message.graph.GetGraphRequest";}
};

} // namespace graph

} // namespace message

} // namespace b0

//! \cond HIDDEN_SYMBOLS

namespace spotify
{

namespace json
{

using b0::message::graph::GetGraphRequest;

template <>
struct default_codec_t<GetGraphRequest>
{
    static codec::object_t<GetGraphRequest> codec()
    {
        auto codec = codec::object<GetGraphRequest>();
        return codec;
    }
};

} // namespace json

} // namespace spotify

//! \endcond

#endif // B0__MESSAGE__GRAPH__GET_GRAPH_REQUEST_H__INCLUDED
