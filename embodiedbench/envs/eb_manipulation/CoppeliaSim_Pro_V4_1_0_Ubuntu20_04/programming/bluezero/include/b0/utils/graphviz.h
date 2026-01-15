#ifndef B0__UTILS__GRAPHVIZ_H
#define B0__UTILS__GRAPHVIZ_H

#include <string>

#include <b0/b0.h>
#include <b0/message/graph/graph.h>

namespace b0
{

namespace graph
{

struct GraphvizOutputOptions
{
    std::string outline_color{"black"};
    std::string topic_color{"blue"};
    std::string service_color{"red"};
    bool cluster_hosts{true};

    inline GraphvizOutputOptions & setOutlineColor(const std::string &c)
    {
        outline_color = c;
        return *this;
    }

    inline GraphvizOutputOptions & setTopicColor(const std::string &c)
    {
        topic_color = c;
        return *this;
    }

    inline GraphvizOutputOptions & setServiceColor(const std::string &c)
    {
        service_color = c;
        return *this;
    }

    inline GraphvizOutputOptions & setClusterHosts(bool c)
    {
        cluster_hosts = c;
        return *this;
    }
};

void toGraphviz(const b0::message::graph::Graph &graph, const std::string &filename, const GraphvizOutputOptions &opts = {});

struct GraphvizRenderOptions
{
    std::string output_format{"png"};
    std::string program{"dot"};

    inline GraphvizRenderOptions & setOutputFormat(const std::string &f)
    {
        output_format = f;
        return *this;
    }

    inline GraphvizRenderOptions & setProgram(const std::string &p)
    {
        program = p;
        return *this;
    }
};

int renderGraphviz(const std::string &input, const std::string &output, const GraphvizRenderOptions &opts = {});

} // namespace graph

} // namespace b0

#endif // B0__UTILS__GRAPHVIZ_H
