#include <set>
#include <iostream>
#include <b0/node.h>
#include <b0/resolver/client.h>

int main(int argc, char **argv)
{
    b0::init(argc, argv);
    b0::Node node("service_list");
    b0::resolver::Client resolv_cli(&node);
    node.init();
    resolv_cli.init();
    b0::message::graph::Graph graph;
    resolv_cli.getGraph(graph);
    std::set<std::string> services;
    for(auto &link : graph.node_service)
        services.insert(link.other_name);
    for(auto &s : services)
        std::cout << s << std::endl;
    resolv_cli.cleanup();
    node.cleanup();
    return 0;
}

