#include <string>
#include <iostream>
#include <iterator>

#include <b0/node.h>
#include <b0/service_client.h>

int main(int argc, char **argv)
{
    std::string node_name = "b0_service_call", service_name = "", content_type = "";
    b0::addOptionString("node-name,n", "name of node", &node_name);;
    b0::addOptionString("service-name,s", "name of service", &service_name);
    b0::addOptionString("content-type,c", "content type", &content_type);
    b0::setPositionalOption("service-name");
    b0::init(argc, argv);

    std::cin >> std::noskipws;
    std::istream_iterator<char> it(std::cin);
    std::istream_iterator<char> end;
    std::string request(it, end), response, response_type;

    b0::Node node(node_name);
    b0::ServiceClient cli(&node, service_name);
    node.init();
    cli.call(request, content_type, response, response_type);
    if(content_type != "")
        std::cout << "Content-type: " << response_type << std::endl;
    std::cout << response << std::flush;
    node.cleanup();
    return 0;
}

