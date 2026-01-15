#include <iostream>

#include <b0/node.h>
#include <b0/subscriber.h>

void callback(const std::string &payload)
{
    std::cout << payload << std::flush;
}

int main(int argc, char **argv)
{
    std::string node_name = "b0_topic_echo", topic_name = "";
    b0::addOptionString("node-name,n", "name of node", &node_name);
    b0::addOptionString("topic-name,t", "name of topic", &topic_name);
    b0::setPositionalOption("topic-name");
    b0::init(argc, argv);

    b0::Node node(node_name);
    b0::Subscriber sub(&node, topic_name, callback);
    node.init();
    node.spin();
    node.cleanup();
    return 0;
}

