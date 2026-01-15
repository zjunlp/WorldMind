#include <boost/python.hpp>

#include <b0/node.h>
#include <b0/publisher.h>
#include <b0/subscriber.h>
#include <b0/service_client.h>
#include <b0/service_server.h>

using namespace boost::python;

void Node_spin(b0::Node *node)
{
    node->spin();
}

b0::Subscriber * Subscriber_new(b0::Node *node, std::string topic_name, object const &callback)
{
    return new b0::Subscriber(node, topic_name,
            [=](const std::string &payload)
            {
                callback(payload);
            }
        );
}

std::string ServiceClient_call(b0::ServiceClient *cli, const std::string &req)
{
    std::string rep;
    cli->call(req, rep);
    return rep;
}

b0::ServiceServer * ServiceServer_new(b0::Node *node, std::string service_name, object const &callback)
{
    return new b0::ServiceServer(node, service_name,
            [=](const std::string &req, std::string &rep)
            {
                object rep_obj = callback(req);
                rep = extract<std::string>(str(rep_obj))();
            }
        );
}

BOOST_PYTHON_MODULE(pyb0)
{
    class_<b0::Node, boost::noncopyable>
        ("Node", init<std::string>())
        .def("init", &b0::Node::init)
        .def("cleanup", &b0::Node::cleanup)
        .def("spin_once", &b0::Node::spinOnce)
        .def("spin", &Node_spin)
        .def("time_usec", &b0::Node::timeUSec)
        .def("hardware_time_usec", &b0::Node::hardwareTimeUSec)
    ;
    class_<b0::Publisher, boost::noncopyable>
        ("Publisher", init<b0::Node*, std::string>())
        .def("init", &b0::Publisher::init)
        .def("cleanup", &b0::Publisher::cleanup)
        .def("get_topic_name", &b0::Publisher::getTopicName)
        .def("publish", &b0::Publisher::publish)
    ;
    class_<b0::Subscriber, boost::noncopyable>
        ("Subscriber", no_init)
        .def("__init__", make_constructor(&Subscriber_new))
        .def("init", &b0::Subscriber::init)
        .def("cleanup", &b0::Subscriber::cleanup)
        .def("get_topic_name", &b0::Subscriber::getTopicName)
    ;
    class_<b0::ServiceClient, boost::noncopyable>
        ("ServiceClient", init<b0::Node*, std::string>())
        .def("init", &b0::ServiceClient::init)
        .def("cleanup", &b0::ServiceClient::cleanup)
        .def("get_service_name", &b0::ServiceClient::getServiceName)
        .def("call", &ServiceClient_call)
    ;
    class_<b0::ServiceServer, boost::noncopyable>
        ("ServiceServer", no_init)
        .def("__init__", make_constructor(&ServiceServer_new))
        .def("init", &b0::ServiceServer::init)
        .def("cleanup", &b0::ServiceServer::cleanup)
        .def("get_service_name", &b0::ServiceServer::getServiceName)
    ;
}

