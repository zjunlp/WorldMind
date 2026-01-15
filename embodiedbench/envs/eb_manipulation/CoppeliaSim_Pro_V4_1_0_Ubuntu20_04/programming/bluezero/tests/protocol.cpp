#include <iostream>

#include <b0/message/resolv/announce_node_request.h>
#include <b0/message/log/log_entry.h>
#include <b0/message/graph/graph.h>
#include <b0/message/graph/graph_link.h>

bool verbose = true;

template<typename T>
void test(const T &msg)
{
    std::string serialized;
    serialize(msg, serialized);
    if(verbose)
        std::cout << msg.type() << ": serialized: " << serialized << std::endl;
    T msg2;
    parse(msg2, serialized);
    std::string reserialized;
    serialize(msg2, reserialized);
    if(verbose)
        std::cout << msg.type() << ": re-serialized: " << reserialized << std::endl;
    if(serialized != reserialized)
    {
        std::cerr << "Test for " << msg.type() << " failed" << std::endl;
        exit(1);
    }
}

int main(int argc, char **argv)
{
    b0::init(argc, argv);

    {
        b0::message::resolv::AnnounceNodeRequest msg;
        msg.node_name = "foo";
        test(msg);
    }

    {
        b0::message::log::LogEntry msg;
        msg.node_name = "foo";
        msg.level = b0::logger::levelInfo(b0::logger::Level::warn).str;
        msg.message = "Hello \x01\xff world";
        msg.time_usec = (uint64_t(1) << 60) + 5978629785;
        test(msg);
    }

    {
        b0::message::graph::Graph g1;
        b0::message::graph::GraphNode n1;
        n1.node_name = "a";
        g1.nodes.push_back(n1);
        b0::message::graph::GraphNode n2;
        n2.node_name = "b";
        g1.nodes.push_back(n2);
        b0::message::graph::GraphNode n3;
        n3.node_name = "c";
        g1.nodes.push_back(n3);
        b0::message::graph::GraphLink l1;
        l1.node_name = "a";
        l1.other_name = "t";
        l1.reversed = false;
        g1.node_topic.push_back(l1);
        b0::message::graph::GraphLink l2;
        l2.node_name = "b";
        l2.other_name = "t";
        l2.reversed = true;
        g1.node_topic.push_back(l2);
        b0::message::graph::GraphLink l3;
        l3.node_name = "c";
        l3.other_name = "t";
        l3.reversed = true;
        g1.node_topic.push_back(l3);
        test(g1);
    }

    return 0;
}

