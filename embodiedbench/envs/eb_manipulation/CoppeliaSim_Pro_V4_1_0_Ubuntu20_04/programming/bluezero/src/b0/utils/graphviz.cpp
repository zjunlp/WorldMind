#include <b0/utils/graphviz.h>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iostream>
#include <boost/format.hpp>
#ifdef HAVE_BOOST_PROCESS
#include <boost/process.hpp>
#endif

namespace b0
{

namespace graph
{

static inline char normalize(char c)
{
    if(c >= 'a' && c <= 'z') return c;
    if(c >= 'A' && c <= 'Z') return c;
    if(c >= '0' && c <= '9') return c;
    if(c == '_') return c;
    return '_';
}

static std::string normalize(const std::string &s)
{
    std::stringstream ss;
    for(int i = 0; i < s.size(); i++)
        ss << normalize(s[i]);
    return ss.str();
}

static std::string id(const std::string &t, const std::string &name)
{
    boost::format fmt("%s_%s");
    return (fmt % t % normalize(name)).str();
}

void toGraphviz(const b0::message::graph::Graph &graph, const std::string &filename, const GraphvizOutputOptions &opts)
{
    std::ofstream f;
    f.open(filename);
    f << "digraph G {" << std::endl;
    f << "    graph [overlap=false, splines=true, bgcolor=\"transparent\"];" << std::endl;
    f << std::endl;
    std::set<std::string> hosts;
    std::map<std::string, std::set<std::string> > nodes_by_host;
    for(auto x : graph.nodes)
    {
        hosts.insert(x.host_id);
        nodes_by_host[x.host_id].insert(x.node_name);
    }
    std::map<std::string, std::set<std::string> > services_by_node;
    for(auto x : graph.node_service)
    {
        services_by_node[x.node_name].insert(x.other_name);
    }
    for(auto host : hosts)
    {
        if(opts.cluster_hosts)
        {
            f << "    subgraph cluster_" << id("H", host) << " {" << std::endl;
            f << "        label=\"" << host << "\";" << std::endl;
            f << "        color=" << opts.outline_color << ";" << std::endl;
            f << "        fontcolor=" << opts.outline_color << ";" << std::endl;
        }
        f << "        node [shape=box, color=" << opts.outline_color << ", fontcolor=" << opts.outline_color << "];" << std::endl;
        for(auto node : nodes_by_host[host])
        {
            f << "        " << id("N", node) << " [label=\"" << node << "\"];" << std::endl;
        }
        f << "        node [shape=diamond, color=" << opts.service_color << "];" << std::endl;
        for(auto node : nodes_by_host[host])
        for(auto service : services_by_node[node])
        {
            f << "        " << id("S", service) << " [label=\"" << service << "\", fontcolor=" << opts.service_color << "];" << std::endl;
        }
        if(opts.cluster_hosts)
        {
            f << "    }" << std::endl;
        }
    }
    f << std::endl;
    f << "    node [shape=ellipse, color=" << opts.topic_color << "];" << std::endl;
    for(auto x : graph.node_topic)
    {
        f << "    " << id("T", x.other_name) << " [label=\"" << x.other_name << "\", fontcolor=" << opts.topic_color << "];" << std::endl;
    }
    f << std::endl;
    f << "    edge [color=" << opts.outline_color << "];" << std::endl;
    for(auto x : graph.node_topic)
    {
        if(x.reversed)
            f << "    " << id("T", x.other_name) << " -> " << id("N", x.node_name) << ";" << std::endl;
        else
            f << "    " << id("N", x.node_name) << " -> " << id("T", x.other_name) << ";" << std::endl;
    }
    for(auto x : graph.node_service)
    {
        if(x.reversed)
            f << "    " << id("S", x.other_name) << " -> " << id("N", x.node_name) << ";" << std::endl;
        else
            f << "    " << id("N", x.node_name) << " -> " << id("S", x.other_name) << ";" << std::endl;
    }
    f << "}" << std::endl;
    f.close();
}

int renderGraphviz(const std::string &input, const std::string &output, const GraphvizRenderOptions &opts)
{
#ifdef HAVE_BOOST_PROCESS
    boost::process::child c(boost::process::search_path(opts.program), "-T", opts.output_format, boost::process::std_out > output, boost::process::std_in < input);
    c.wait();
    return c.exit_code();
#else
    std::cerr << "boost/process.hpp is needed for executing " << opts.program << std::endl;
    return 1;
#endif
}

} // namespace graph

} // namespace b0

