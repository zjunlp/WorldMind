#ifndef B0__MESSAGE__GRAPH__GRAPH_H__INCLUDED
#define B0__MESSAGE__GRAPH__GRAPH_H__INCLUDED

#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>

#include <b0/b0.h>
#include <b0/message/message.h>
#include <b0/message/graph/graph_node.h>
#include <b0/message/graph/graph_link.h>

namespace b0
{

namespace message
{

namespace graph
{

/*!
 * \brief A complete graph of the network
 *
 * \sa GraphLink, \ref protocol, \ref graph
 */
class Graph : public Message
{
public:
    //! List of node names
    std::vector<GraphNode> nodes;

    //! List of topic links
    std::vector<GraphLink> node_topic;

    //! List of service links
    std::vector<GraphLink> node_service;

public:
    std::string type() const override {return "b0.message.graph.Graph";}
};

} // namespace graph

} // namespace message

} // namespace b0

//! \cond HIDDEN_SYMBOLS

namespace spotify
{

namespace json
{

using b0::message::graph::Graph;

template <>
struct default_codec_t<Graph>
{
    static codec::object_t<Graph> codec()
    {
        auto codec = codec::object<Graph>();
        codec.required("nodes", &Graph::nodes);
        codec.required("node_topic", &Graph::node_topic);
        codec.required("node_service", &Graph::node_service);
        return codec;
    }
};

} // namespace json

} // namespace spotify

//! \endcond

#endif // B0__MESSAGE__GRAPH__GRAPH_H__INCLUDED
