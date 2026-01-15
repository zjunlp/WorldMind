#include <set>
#include <iostream>
#include <b0/node.h>
#include <b0/resolver/client.h>

int main(int argc, char **argv)
{
    b0::init(argc, argv);
    b0::Node node("topic_list");
    b0::resolver::Client resolv_cli(&node);
    node.init();
    resolv_cli.init();
    b0::message::graph::Graph graph;
    resolv_cli.getGraph(graph);
    std::set<std::string> topics;
    for(auto &link : graph.node_topic)
        topics.insert(link.other_name);
    for(auto &t : topics)
        std::cout << t << std::endl;
    resolv_cli.cleanup();
    node.cleanup();
    return 0;
}

