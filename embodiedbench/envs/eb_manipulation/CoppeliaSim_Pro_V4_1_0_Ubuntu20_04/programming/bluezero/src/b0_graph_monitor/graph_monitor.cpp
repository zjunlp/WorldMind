#include <cstdlib>
#include <iostream>
#include <b0/node.h>
#include <b0/subscriber.h>
#include <b0/resolver/client.h>
#include <b0/utils/graphviz.h>
#include <b0/utils/env.h>
#ifdef HAVE_BOOST_PROCESS
#include <boost/process.hpp>
#endif

namespace b0
{

namespace graph
{

class Console : public b0::Node
{
public:
    Console()
        : Node("graph_monitor"),
          resolv_cli_(this),
          sub_(this, "graph", &Console::onGraphChanged, this)
    {
    }

    ~Console()
    {
    }

    void init() override
    {
        Node::init();

        init_time_ = hardwareTimeUSec();

        resolv_cli_.init();
    }

    void spinOnce() override
    {
        Node::spinOnce();

        if(!graph_received_ && (hardwareTimeUSec() - init_time_) > 2000000)
        {
            // if a graph has not been received within first few seconds
            // manually request it via resolv service:
            info("Requesting graph");

            b0::message::graph::Graph graph;
            resolv_cli_.getGraph(graph);
            printOrDisplayGraph("Current graph", graph);

            graph_received_ = true;
        }
    }

    void onGraphChanged(const b0::message::graph::Graph &graph)
    {
        printOrDisplayGraph("Graph has changed", graph);
    }

    void printOrDisplayGraph(std::string message, const b0::message::graph::Graph &graph)
    {
        graph_received_ = true;

        if(termHasImageCapability())
        {
            info(message);
            renderAndDisplayGraph(graph);
        }
        else
        {
            info("%s: %d nodes", message, graph.nodes.size());
        }
    }

    void renderAndDisplayGraph(const b0::message::graph::Graph &graph)
    {
        GraphvizOutputOptions outputOpts;
        outputOpts.setOutlineColor("white");
        outputOpts.setTopicColor("cyan");
        outputOpts.setServiceColor("red");
        outputOpts.setClusterHosts(b0::hasOption("cluster"));
        toGraphviz(graph, "graph.gv", outputOpts);

        GraphvizRenderOptions renderOpts;
        if(renderGraphviz("graph.gv", "graph.png", renderOpts) == 0)
        {
            displayInlineImage("graph.png");
        }
        else
        {
            std::cerr << "failed to execute " << renderOpts.program << std::endl;
            return;
        }
    }

    bool termHasImageCapability()
    {
        std::string TERM_PROGRAM = b0::env::get("TERM_PROGRAM");
        return TERM_PROGRAM == "iTerm.app";
    }

    int displayInlineImage(std::string filename)
    {
#ifdef HAVE_BOOST_PROCESS
        boost::process::child c(boost::process::search_path("imgcat"), "graph.png");
        c.wait();
        return c.exit_code();
#else
        std::cerr << "boost/process.hpp is needed for inline image display" << std::endl;
        return 1;
#endif
    }

protected:
    b0::resolver::Client resolv_cli_;
    b0::Subscriber sub_;
    int64_t init_time_;
    bool graph_received_{false};
};

} // namespace graph

} // namespace b0

int main(int argc, char **argv)
{
    b0::addOption("cluster,c", "Group (cluster) nodes by host");
    b0::init(argc, argv);
    b0::graph::Console console;
    console.init();
    console.spin();
    console.cleanup();
    return 0;
}

