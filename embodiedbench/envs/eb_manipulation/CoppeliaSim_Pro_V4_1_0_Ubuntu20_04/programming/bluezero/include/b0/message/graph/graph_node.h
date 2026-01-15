#ifndef B0__MESSAGE__GRAPH__GRAPH_NODE_H__INCLUDED
#define B0__MESSAGE__GRAPH__GRAPH_NODE_H__INCLUDED

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
 * \brief A link in the network
 *
 * \sa Graph, \ref protocol, \ref graph
 */
class GraphNode : public Message
{
public:
    //! The hostname
    std::string host_id;

    //! The process id containing the node
    int process_id;

    //! The name of the node
    std::string node_name;

public:
    std::string type() const override {return "b0.message.graph.GraphNode";}
};

} // namespace graph

} // namespace message

} // namespace b0

//! \cond HIDDEN_SYMBOLS

namespace spotify
{

namespace json
{

using b0::message::graph::GraphNode;

template <>
struct default_codec_t<GraphNode>
{
    static codec::object_t<GraphNode> codec()
    {
        auto codec = codec::object<GraphNode>();
        codec.required("host_id", &GraphNode::host_id);
        codec.required("process_id", &GraphNode::process_id);
        codec.required("node_name", &GraphNode::node_name);
        return codec;
    }
};

} // namespace json

} // namespace spotify

//! \endcond

#endif // B0__MESSAGE__GRAPH__GRAPH_NODE_H__INCLUDED
