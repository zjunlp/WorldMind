#ifndef B0__B0_H__INCLUDED
#define B0__B0_H__INCLUDED

#include <b0/config.h>
#include <b0/logger/level.h>

//! \file

/*!
 * \mainpage BlueZero
 *
 * \section Introduction
 *
 * BlueZero (in short, B0) is an open-source library for developing
 * distributed applications.
 *
 * The building blocks of B0 are the \b nodes, which can talk to each other by
 * sending \b messages over \b sockets. Multiple nodes can exist in the same thread,
 * or in multiple threads, or in multiple processes, or distributed across
 * multiple machines.
 *
 * A message can be any sequence of bytes, and it is guaranteed to be delivered atomically.
 *
 * A socket is an abstraction for connecting nodes to each other, and sending messages.
 *
 * The two principal paradigms of communication are the \b client-server, and the \b publish-subscribe.
 *
 * In the client-server pattern, a client node sends one request to a server node, addressed by
 * name (i.e. the service name), and receives back one reply from the server node. The
 * communication in this pattern is synchronous.
 *
 * In the publish-subscribe pattern, a publisher node sends data to a channel addressed by name
 * (i.e. the topic name), and any node which is interested in that data, can subscribe
 * to that topic in order to receive it. The communication in this pattern is asynchronous.
 *
 * \section nodes Nodes
 *
 * The main entity in the BlueZero is the node. A node can represent a process running in a
 * machine. Anyway, there is no such restriction as one-node-per-process, or one-node-per-thread,
 * so it is possible to have many nodes running in the same process. See also \ref threading.
 *
 * The main class used to create a node is b0::Node.
 * Node uses two-phase initialization, so you must call b0::Node::init() after the constructor,
 * and b0::Node::cleanup() before the destructor.
 * <b>Do not call b0::Node::init() from your node class constructor!</b>
 *
 * The methods of b0::Node class communicate with the resolver node (see below) to perform some
 * handshaking and naming assignment and resolution.
 *
 * b0::Node::spinOnce() must be called periodically to process incoming messages (the method
 * b0::Node::spin() does exactly this, until node shutdown is requested).
 *
 * \image html node-state-machine.png Node state transtion diagram.
 *
 * b0::Node::init() will initialize the node and announce its name to the
 * resolver node, and it will initialize each of its publishers, subscribers,
 * clients and servers.
 *
 * Any publishers, subscribers, service client and servers must be constructed prior to calling
 * b0::Node::init(), although there is a way to dynamically create such sockets afterwards (by
 * setting the `managed` flag to false).
 *
 * \section topics Topics
 *
 * When using topics, messages are routed via a transport system using publish / subscribe semantics.
 * The topic is a name used to identify the content of the messages.
 * A node that is interested in a certain kind of data will subscribe to the appropriate topic.
 * There may be multiple concurrent publishers and subscribers for a single topic.
 * and a single node may publish and/or subscribe to multiple topics. In general, publishers and
 * subscribers are not aware of each others' existence. The idea is to decouple the production of
 * information from its consumption. Logically, one can think of a topic as a strongly typed
 * message bus. Each bus has a name, and anyone can connect to the bus to send or receive
 * messages as long as they are the right type.
 *
 * Note: topic names can be changed at runtime; see \ref remapping.
 *
 * \section services Services
 *
 * The publish / subscribe model is a very flexible communication paradigm, but its many-to-many,
 * one-way transport is not appropriate for request / reply interactions, which are often required
 * in a distributed system. Request / reply is done via services, which are defined by a pair of
 * message structures: one for the request and one for the reply. A providing node offers a service
 * under a name and a client uses the service by sending the request message and awaiting the reply.
 *
 * Note: service names can be changed at runtime; see \ref remapping.
 *
 * \section threading Threading and thread safety
 *
 * The functions of the library are generally not thread-safe (unless stated otherwise).
 *
 * Thus, every node's non-thread-safe methods must be accessed always from the same thread which
 * created the node.
 *
 * \section resolver_intro Resolver
 *
 * The most important part of the network is the resolver node.
 *
 * The resolver node implements a part of the \ref protocol "protocol", with the other counterpart of the protocol implemented by the \ref b0::Node "node".
 *
 * The resolver node is implemented in b0::resolver::Resolver and will provide following services to other nodes:
 *
 * - node name resolution
 * - socket name resolution (for service client / server sockets)
 * - messages routing (for publisher / subscriber sockets)
 * - liveness monitoring
 * - tracking of connected nodes
 * - topics / services introspection (see \ref graph)
 * - clock synchronization (see \ref timesync)
 *
 * \b Important: you must have the resolver node running prior to running any node. See \ref remote_nodes for more information about running distributed nodes.
 *
 * See also \ref remote_nodes for information about connecting remote nodes.
 *
 * \section examples Examples
 *
 * \subsection example_pubsub Topics (Publisher/Subscriber)
 *
 * Example of how to create a simple node with one publisher and sending
 * some messages to some topic:
 *
 * \include examples/publisher_subscriber/publisher_node.cpp
 *
 * And the corresponding example of a simple node with one subscriber:
 *
 * \include examples/publisher_subscriber/subscriber_node.cpp
 *
 * You can have multiple publishers and subscribers as well:
 *
 * \ref publisher_subscriber_multi/multi_publisher.cpp "Node with multiple publishers"
 *
 * \ref publisher_subscriber_multi/multi_subscriber.cpp "Node with multiple subscribers"
 *
 * And following is an example of using it in a more object-oriented way:
 *
 * \ref publisher_subscriber_oop/publisher_node_object.cpp "OOP publisher node"
 *
 * \ref publisher_subscriber_oop/subscriber_node_object.cpp "OOP subscriber node"
 *
 * \subsection example_clisrv Services (Client/Server)
 *
 * Example of how to create a simple node with a service client:
 *
 * \include examples/client_server/client_node.cpp
 *
 * And the corresponding example of a simple node with a service server:
 *
 * \include examples/client_server/server_node.cpp
 *
 * And the same thing, object-oriented:
 *
 * \ref client_server_oop/client_node_object.cpp "OOP client node"
 *
 * \ref client_server_oop/server_node_object.cpp "OOP server node"
 *
 *
 * \page remote_nodes Connecting remote nodes
 *
 * When distributing nodes across multiple machines, you must make sure that:
 * - nodes know how to reach the resolver node;
 * - all nodes are able to reach each other, using IP and TCP protocol.
 *
 * \section reaching_resolver Reaching the resolver node
 *
 * Nodes by default will try to reach resolver at tcp://localhost:22000, which only works
 * when testing all nodes on the same machine. When the resolver node is on another machine,
 * (for example the machine hostname is resolver-node-host.local)
 * the environment variable B0_RESOLVER must be set to the correct address, e.g.:
 *
 * ~~~
 * export B0_RESOLVER="tcp://resolver-node-host.local:22000"
 * ~~~
 *
 * prior to launching every node, or in .bashrc or similar.
 *
 * When B0_RESOLVER is not specified it defaults to "tcp://localhost:22000".
 *
 * \section reaching_nodes Reaching every other node
 *
 * Nodes also need to be able to reach each other node.
 * When a node creates a directly addressed socket, such as b0::ServiceClient or b0::ServiceServer, it
 * will advertise that socket name and address to resolver.
 *
 * Since there is no reliable way of automatically determining the correct IP address of the node
 * (as there may be more than one), by default the node will use its hostname when specifying the
 * socket TCP address.
 *
 * This requires that all machines are able to reach each other by their hostnames.
 * This is the case when there is a name resolution service behind, such as a DNS, or
 * Avahi/ZeroConf/Bonjour.
 *
 * Suppose we have a network with two machines: A and B.
 * Machine A hostname is alice.local, and machine B hostname is bob.local.
 *
 * If from machine A we are able to reach (ping) machine B by using its hostname bob.local,
 * and vice-versa, from machine B we are able to reach machine A by using its hostname alice.local,
 * there is no additional configuration to set. Otherwise, we need to explicitly tell how a machine
 * is reached from outside (i.e. what's the correct IP or hostname), by setting the B0_HOST_ID
 * environment variable, e.g.:
 *
 * ~~~
 * export B0_HOST_ID="192.168.1.3"
 * ~~~
 *
 * When B0_HOST_ID is not specified it defaults to the machine hostname.
 *
 * \section remote_nodes_example Example
 *
 * Suppose we have a network with two machines: A and B.
 *
 * - Machine A (192.168.1.5) will run the resolver node and a subscriber node.
 * - Machine B (192.168.1.6) will run a publisher node.
 *
 * By default, nodes will use their hostnames when announcing socket addresses.
 * We override this behavior, by setting B0_HOST_ID to the machine IP address.
 *
 * \subsection remote_nodes_example_a_resolver Machine A - starting resolver
 *
 * On machine A we run
 *
 * ~~~
 * export B0_HOST_ID="192.168.1.5"
 *
 * ./resolver
 * ~~~
 *
 * to run the resolver node.
 *
 * \subsection remote_nodes_example_a_subscriber Machine A - starting subscriber
 *
 * On machine A we run
 *
 * ~~~
 * export B0_HOST_ID="192.168.1.5"
 *
 * ./examples/publisher_subscriber/subscriber_node
 * ~~~
 *
 * to run the subscriber node.
 *
 * Note that for this machine we don't need to specify B0_RESOLVER, because the default value
 * (tcp://localhost:22000) is good to reach the resolver socket.
 *
 * \subsection remote_nodes_example_b_publisher Machine B - starting publisher
 *
 * On machine B we run
 *
 * ~~~
 * export B0_HOST_ID="192.168.1.6"
 *
 * export B0_RESOLVER="tcp://192.168.1.5:22000"
 *
 * ./examples/publisher_subscriber/publisher_node
 * ~~~
 *
 * to run the publisher node.
 *
 *
 * \page protocol Protocol
 *
 * This page describes the BlueZero protocol. Knowledge of the communication protocol is
 * not required to use the BlueZero library, but serves as a specification for
 * re-implementing BlueZero.
 *
 * \section protocol_intro Introduction
 *
 * The transport of messages exchanged between BlueZero nodes is implemented with ZeroMQ.
 * REQ/REP sockets are used for b0::ServiceClient and b0::ServiceServer sockets, and PUB/SUB
 * sockets are run through an XSUB/XPUB proxy and used for b0::Publisher and b0::Subscriber sockets.
 *
 * \subsection protocol_serialization Serialization
 *
 * Messages are wrapped in a b0::message::MessageEnvelope message, which contains the
 * target socket name, followed by a series of headers (HTTP-like), followed by a blank
 * line, followed by a payload of arbitrary length and format.
 *
 * The headers specify the number of payloads, the length of each payload, and optionally
 * the content type of each payload, and the compression algorithm used.
 *
 * BlueZero is agnostic to message payload. Payloads can be plain text strings with
 * arbitrary encodings such as ASCII or UTF8, or some higher level serialization
 * mechanism can be used, such as JSON, BSON, MsgPack, or Protocol Buffers.
 *
 * BlueZero uses JSON (via the spotify-json library) to serialize messages used for
 * core functionality, such as node announcement, name resolution, and heartbeat sending.
 *
 * The messages exchanged via the 'resolv' service are instance of the class
 * b0::message::resolv::Request and b0::message::resolv::Response. These classes have
 * several optional fields of which only one is set, for the corresponding request/response type.
 *
 * See \ref node_startup for an actual example of how a BlueZero core message is serialized.
 *
 * \subsection protocol_topology Topology
 *
 * The network architecture is mostly centralized: every node will talk to the resolver node, except
 * for services which use dedicated sockets, and topics which use a XPUB/XSUB proxy.
 *
 * The resolver node offers one service ('resolv'), and also runs the XPUB/XSUB proxy.
 *
 * There are three main phases of the lifetime of a node:
 * - startup
 * - normal lifetime
 * - shutdown
 *
 * \section node_startup Node startup
 *
 * In the startup phase a node must announce its presence to the resolver node via the
 * b0::message::resolv::AnnounceNodeRequest message.
 * The resolver will reply with the b0::message::resolv::AnnounceNodeResponse message,
 * containing the final node name (as it may be changed in case of a
 * name clash) and important info for node communication, such as the XPUB/XSUB addresses.
 *
 * \mscfile node-startup.msc
 *
 * Example request:
 *
 * ```
 * resolv
 * Content-length: 91
 * Part-count: 1
 * Content-length-0: 91
 * Content-type-0: b0.message.resolv.Request
 *
 * {
 *     "announce_node":
 *     {
 *         "host_id": "10.0.25.67",
 *         "process_id": 27489,
 *         "node_name": "robot_status_publisher"
 *     }
 * }
 * ```
 *
 * Example response:
 *
 * ```
 * resolv
 * Content-length: 190
 * Part-count: 1
 * Content-length-0: 190
 * Content-type-0: b0.message.resolv.Response
 *
 * {
 *     "announce_node":
 *     {
 *         "ok": true,
 *         "node_name": "robot_status_publisher",
 *         "xsub_sock_addr": "tcp://10.0.25.38:61060",
 *         "xpub_sock_addr": "tcp://10.0.25.38:61061",
 *         "minimum_heartbeat_interval": 30000000
 *     }
 * }
 * ```
 *
 * \subsection node_startup_topics Topics
 *
 * As part of the \ref graph "node graph protocol", if the node subscribes or publishes on some topic,
 * it will inform the resolver node via the b0::message::graph::NodeTopicRequest message.
 *
 * \mscfile graph-topic.msc
 *
 * \subsection node_startup_services Services
 *
 * If the node offers some service, it will announce each service name and address
 * via the b0::message::resolv::AnnounceServiceRequest message.
 *
 * \mscfile node-startup-service.msc
 *
 * Additionally, as part of the \ref graph "node graph protocol", if the node offers or uses some service,
 * it will inform the resolver node via the b0::message::graph::NodeServiceRequest message.
 *
 * \mscfile graph-service.msc
 *
 * \section node_lifetime Normal node lifetime
 *
 * During node lifetime, a node will periodically send a heartbeat to allow the resolver node
 * to track dead nodes.
 *
 * \mscfile node-lifetime.msc
 *
 * \subsection node_lifetime_topics Topics
 *
 * When a node wants to publish to some topic, it has to use the XPUB address given by resolver
 * in the b0::message::resolv::AnnounceNodeResponse message.
 * The payload to write to the socket is a b0::message::MessageEnvelope message.
 *
 * \mscfile topic-write.msc
 *
 * Similarly, when it wants to subscribe to some topic, the messages are read from the XSUB
 * socket.
 *
 * \mscfile topic-read.msc
 *
 * \subsection node_lifetime_services Services
 *
 * When a node wants to use a service, it has to resolve the service name to an address,
 * via the b0::message::resolv::ResolveServiceRequest message.
 *
 * \mscfile service-resolve.msc
 *
 * The request payload to write to the socket, as well as the response payload to be read
 * from the socket, are a b0::message::MessageEnvelope message.
 *
 * \mscfile service-call.msc
 *
 * \section node_shutdown Node shutdown
 *
 * When a node is shutdown, it will send a b0::message::resolv::ShutdownNodeRequest message to inform the resolver node about that.
 *
 * \mscfile node-shutdown.msc
 *
 * Additionally, it will send b0::message::graph::NodeTopicRequest and b0::message::graph::NodeServiceRequest to inform about not using or offering the
 * topics or services anymore.
 *
 * \mscfile graph-topic.msc
 * \mscfile graph-service.msc
 *
 *
 * \page graph Graph protocol
 *
 * The graph protocol is a subset of the \ref protocol "protocol", consisting of a series of messages used to allow introspection of node, topics, and services connections.
 *
 * The messages sent by sockets to inform resolver about these connections are b0::message::graph::NodeTopicRequest and b0::message::graph::NodeServiceRequest (see \ref protocol).
 *
 * Additionally, the b0::message::graph::GetGraphRequest message can be used to retrieve the graph:
 *
 * \mscfile graph-get.msc
 *
 * The program b0_graph_console (and also gui/b0_graph_console_gui) included in BlueZero is
 * an example of displaying such graph, while whatching for changes to it in realtime.
 *
 * Here is a rendering of the graph (b0::message::graph::Graph) during a BlueZero session
 * with several nodes running:
 *
 * \dotfile graph-example.gv
 *
 * Black rectangles are nodes, red diamonds are services, and blue ovals are topics.
 *
 * An arrow from node to topic means a node is publishing to a topic. Vice-versa, an arrow from topic to node means a node is subscribing to a topic.
 *
 * An arrow from node to service means a node is offering a service. Vice-versa, an arrow from service to node means a node is using a service.
 *
 * Nodes have an implicit connection to the 'resolv' service, however it is not shown in the graph.
 *
 *
 * \page remapping Remapping: dynamically changing names of nodes and sockets
 *
 * Names for nodes, topics and services, are usually specified statically, i.e. as constant
 * values in the C++ application.
 *
 * This works fine for most applications. For example the built-in topic `log` used by
 * BlueZero for implementing remote logging, is an example of that.
 *
 * There are cases instead where we need multiple instances of the same node.
 * Since BlueZero doesn't allow two nodes in the same network to have the same name, when
 * starting a second instance of the same node, the name  will be automatically changed by
 * resolver during the announceNode handshake phase.
 * This is not a big deal, as the node name is not usually needed for addressing other sockets.
 *
 * However, if the node offers a service, it will not be possible to startup the same node
 * twice, as the second instance of the node will fail to register its own service socket address
 * with the specified name.
 *
 * Another usecase is with topics: topics create a named network connection between two nodes.
 * If we have a pair of nodes, one with a publisher, and one with a subscriber, those are going to
 * be connected always via the same named channel (i.e. the topic).
 * Starting multiple instances of the publisher and subscriber node will keep using the existing
 * channel, making it impossible to reuse nodes for doing the same task in different contexts
 * (e.g. an image processing node that can re-used multiple times).
 *
 * The common pattern that is used in the above mentioned scenarios, is to make the node name,
 * topic name, or service name, to be based on a command line argument, or on an environment variable,
 * so that is possible to have a dynamic node name, topic name or service name.
 *
 * This usage pattern is so common that BlueZero already provides a way to do it via command
 * line options.
 * The command line options (passed when launching a node) used to do this are:
 *
 *  - `--remap-node oldName=newName` or in short `-NoldName=newName`
 *  - `--remap-topic oldName=newName` or in short `-ToldName=newName`
 *  - `--remap-service oldName=newName` or in short `-SoldName=newName`
 *
 * There exists also a fourth option, to remap any name:
 *
 *  - `--remap oldName=newName` or in short `-RoldName=newName`
 *
 * which is a shorthand for using all of the three options with the same remapping.
 *
 * \section remapping_placeholders Special placeholders
 *
 * Another common pattern used for creating unique node names, topic names, or service names,
 * is to include the hostname in the node name, or to include the node name in the topic or
 * service name.
 *
 * BlueZero allows to easily perform these substitutions as well:
 *  - the placeholder `%%h` will be replaced with the hostname (`B0_HOST_ID`);
 *  - the placeholder `%%n` will be replaced with the final node name (i.e. after any remapping and after being re-assigned by resolver in case of name clash).
 *
 * So, by assigning a node the name `thenode@%%h`, it will finally get assigned the name
 * `thenode@thehostname.local` (if `thehostname.local` is the machine's hostname, or has been
 * set in `B0_HOST_ID`).
 *
 * \section remapping_example Example
 *
 * Here we consider a simple example with 3 nodes:
 *
 *  - \ref remapping/const.cpp "const" is a node publishing a constant value, specified via the `-v` option, on the `out` topic;
 *  - \ref remapping/print.cpp "print" is a node printing what it receives on the `in` topic;
 *  - \ref remapping/operation.cpp "operation" will perform a mathematical operation, specified via the `-o` option, on the values received on the `a` and `b` topics, and will publish the result on the `out` topic.
 *
 * In order to connect these nodes, we have to use remapping. For example we can connect an instance of `const` to an instance of `print`, by remapping `in` to `out` in `print` node (or by remapping `out` to `in` in `const`). Or we can create a more complex scenario, for example for computing the expression `(1+2)*3`:
 *
 * \dotfile remapping-example.gv
 *
 * The above node graph has been created by starting the nodes:
 *
 *  - `./const -v1 -Tout=x -Nconst=const-1`
 *  - `./const -v2 -Tout=y -Nconst=const-2`
 *  - `./operation -o+ -Ta=x -Tb=y -Tout=sum -Noperation=op-sum`
 *  - `./const -v3 -Tout=z -Nconst=const-3`
 *  - `./operation -o\* -Ta=sum -Tb=z -Tout=result -Noperation=op-mul`
 *  - `./print -Tin=result`
 *
 *
 * \page console_logging Console logging control
 *
 * When nodes use the logging methods (b0::Node::log(), or any of the shortcuts b0::Node::trace(),
 * b0::Node::debug(), b0::Node::info(), b0::Node::warn(), b0::Node::error(), b0::Node::fatal()), a
 * log entry is sent to the `log` topic, but it is also displayed in the console if the console
 * logging level is low enough to let the message pass.
 *
 * The console logging level can be controlled by the `B0_CONSOLE_LOGLEVEL` environment variable
 * (e.g. set `B0_CONSOLE_LEVEL=trace`), and can be overridden by the `--console-loglevel` command
 * line switch (e.g. by passing `--console-loglevel=trace` or in short `-Ltrace` as command line
 * option).
 *
 * Additionally, it can be changed via the b0::setConsoleLogLevel() function. Depending if this is
 * called before b0::init() or after b0::init(), it can provide a different default (recommended),
 * or completely override any env var or command line switch setting.
 *
 *
 * \page cmdline_args Parsing command line arguments
 *
 * The BlueZero library handles the parsing of command line options.
 *
 * Nodes can specify additional options to parse, calling b0::addOption() before b0::init().
 *
 * The user-specified options will be listed in the help page, among with standard BlueZero
 * command-line options.
 *
 * An option can be marked as positional using b0::setPositionalOption().
 *
 * Option values can be read, after calling b0::init(), using the function b0::hasOption() and
 * b0::getOption().
 *
 * Example:
 *
 * \include examples/cmdline_args/args.cpp
 *
 * Running the program with the `-h` option produces the help page, which includes both
 * standard BlueZero node options, and the custom options:
 *
 * ```
 * Allowed options:
 *   -h [ --help ]                         display help message
 *   -R [ --remap ] oldName=newName        remap any name
 *   -N [ --remap-node ] oldName=newName   remap a node name
 *   -T [ --remap-topic ] oldName=newName  remap a topic name
 *   -S [ --remap-service ] oldName=newName
 *                                         remap a service name
 *   -L [ --console-loglevel ] arg (=info) specify the console loglevel
 *   -F [ --spin-rate ] arg (=10)          specify the default spin rate
 *   -n [ --fancy-name ] arg               a string arg
 *   -x [ --lucky-number ] arg (=23)       an int arg with default
 *   -f [ --file ] arg                     file arg
 * ```
 */

#include <memory>
#include <string>
#include <vector>

namespace b0
{

//! \cond HIDDEN_SYMBOLS

class Node;

class Global final
{
private:
    Global();

    Global(const Global &) = delete;

    struct Private;

public:
    static Global & getInstance();

    void addRemapings(const std::vector<std::string> &raw_arg);

    void addNodeRemapings(const std::vector<std::string> &raw_arg);

    void addTopicRemapings(const std::vector<std::string> &raw_arg);

    void addServiceRemapings(const std::vector<std::string> &raw_arg);

    void addRemaping(const std::string &orig_name, const std::string &new_name);

    void addNodeRemaping(const std::string &orig_name, const std::string &new_name);

    void addTopicRemaping(const std::string &orig_name, const std::string &new_name);

    void addServiceRemaping(const std::string &orig_name, const std::string &new_name);

    void printUsage(const std::string &argv0, bool toStdErr = false);

    void addOption(const std::string &name, const std::string &description);

    void addOptionString(const std::string &name, const std::string &description, std::string *ptr, bool required, const std::string &default_value);

    void addOptionInt(const std::string &name, const std::string &description, int *ptr, bool required, int default_value);

    void addOptionInt64(const std::string &name, const std::string &description, int64_t *ptr, bool required, int64_t default_value);

    void addOptionDouble(const std::string &name, const std::string &description, double *ptr, bool required, double default_value);

    void addOptionStringVector(const std::string &name, const std::string &description, std::vector<std::string> *ptr, bool required, const std::vector<std::string> &default_value);

    void addOptionIntVector(const std::string &name, const std::string &description, std::vector<int> *ptr, bool required, const std::vector<int> &default_value);

    void addOptionInt64Vector(const std::string &name, const std::string &description, std::vector<int64_t> *ptr, bool required, const std::vector<int64_t> &default_value);

    void addOptionDoubleVector(const std::string &name, const std::string &description, std::vector<double> *ptr, bool required, const std::vector<double> &default_value);

    void setPositionalOption(const std::string &option, int max_count);

    int hasOption(const std::string &option);

    std::string getOptionString(const std::string &option);

    int getOptionInt(const std::string &option);

    int64_t getOptionInt64(const std::string &option);

    double getOptionDouble(const std::string &option);

    std::vector<std::string> getOptionStringVector(const std::string &option);

    std::vector<int> getOptionIntVector(const std::string &option);

    std::vector<int64_t> getOptionInt64Vector(const std::string &option);

    std::vector<double> getOptionDoubleVector(const std::string &option);

    void init(const std::vector<std::string> &argv);

    void init(int &argc, char **argv);

    bool isInitialized() const;

    std::string getRemappedNodeName(const b0::Node &node, const std::string &node_name);

    std::string getRemappedTopicName(const b0::Node &node, const std::string &topic_name);

    std::string getRemappedServiceName(const b0::Node &node, const std::string &service_name);

    bool remapNodeName(const b0::Node &node, const std::string &node_name, std::string &remapped_node_name);

    bool remapTopicName(const b0::Node &node, const std::string &topic_name, std::string &remapped_topic_name);

    bool remapServiceName(const b0::Node &node, const std::string &service_name, std::string &remapped_service_name);

    logger::Level getConsoleLogLevel();

    void setConsoleLogLevel(logger::Level level);

    double getSpinRate();

    void setSpinRate(double rate);

    bool quitRequested();

    void quit();

private:
    std::unique_ptr<Private> private_;

    friend Private;
};

//! \endcond

/*!
 * Initialize the B0 library.
 *
 * This function will read any relevant environment variables and command line arguments,
 * and change settings in the global data structure.
 *
 * Depending if you want to override those settings or just provide a different default,
 * you need to alter such settings before or after calling b0::init() respectively.
 *
 * This function will validate command line arguments and terminate the process in case of error.
 */
void init(const std::vector<std::string> &argv = {});

/*!
 * Initialize the B0 library.
 *
 * This function will read any relevant environment variables and command line arguments,
 * and change settings in the global data structure.
 *
 * Depending if you want to override those settings or just provide a different default,
 * you need to alter such settings before or after calling b0::init() respectively.
 *
 * This function will validate command line arguments and terminate the process in case of error.
 */
void init(int &argc, char **argv);

/*!
 * Check wether BlueZero is already initialized
 */
bool isInitialized();

/*!
 * Print a description of command line options
 */
void printUsage(const std::string &argv0, bool toStdErr = false);

/*!
 * Declare a named command line option.
 *
 * Use the form "long,l" to declare a long option name (--long) together with
 * a short option name (-l) for the same option.
 *
 * See also \ref cmdline_args.
 */
void addOption(const std::string &name, const std::string &description);

/*!
 * Declare a named command line string option.
 *
 * Use the form "long,l" to declare a long option name (--long) together with
 * a short option name (-l) for the same option.
 *
 * See also \ref cmdline_args.
 */
void addOptionString(const std::string &name, const std::string &description, std::string *ptr = nullptr, bool required = false, const std::string &default_value = "");

/*!
 * Declare a named command line int option.
 *
 * Use the form "long,l" to declare a long option name (--long) together with
 * a short option name (-l) for the same option.
 *
 * See also \ref cmdline_args.
 */
void addOptionInt(const std::string &name, const std::string &description, int *ptr = nullptr, bool required = false, int default_value = 0);

/*!
 * Declare a named command line int64_t option.
 *
 * Use the form "long,l" to declare a long option name (--long) together with
 * a short option name (-l) for the same option.
 *
 * See also \ref cmdline_args.
 */
void addOptionInt64(const std::string &name, const std::string &description, int64_t *ptr = nullptr, bool required = false, int64_t default_value = 0);

/*!
 * Declare a named command line double option.
 *
 * Use the form "long,l" to declare a long option name (--long) together with
 * a short option name (-l) for the same option.
 *
 * See also \ref cmdline_args.
 */
void addOptionDouble(const std::string &name, const std::string &description, double *ptr = nullptr, bool required = false, double default_value = 0.0);

/*!
 * Declare a named command line string vector option.
 *
 * Use the form "long,l" to declare a long option name (--long) together with
 * a short option name (-l) for the same option.
 *
 * See also \ref cmdline_args.
 */
void addOptionStringVector(const std::string &name, const std::string &description, std::vector<std::string> *ptr = nullptr, bool required = false, const std::vector<std::string> &default_value = {});

/*!
 * Declare a named command line int vector option.
 *
 * Use the form "long,l" to declare a long option name (--long) together with
 * a short option name (-l) for the same option.
 *
 * See also \ref cmdline_args.
 */
void addOptionIntVector(const std::string &name, const std::string &description, std::vector<int> *ptr = nullptr, bool required = false, const std::vector<int> &default_value = {});

/*!
 * Declare a named command line int64_t vector option.
 *
 * Use the form "long,l" to declare a long option name (--long) together with
 * a short option name (-l) for the same option.
 *
 * See also \ref cmdline_args.
 */
void addOptionInt64Vector(const std::string &name, const std::string &description, std::vector<int64_t> *ptr = nullptr, bool required = false, const std::vector<int64_t> &default_value = {});

/*!
 * Declare a named command line double vector option.
 *
 * Use the form "long,l" to declare a long option name (--long) together with
 * a short option name (-l) for the same option.
 *
 * See also \ref cmdline_args.
 */
void addOptionDoubleVector(const std::string &name, const std::string &description, std::vector<double> *ptr = nullptr, bool required = false, const std::vector<double> &default_value = {});

/*!
 * Convenience method for adding a positional command line option (the option must have been previously added with b0::addOption())
 *
 * See also \ref cmdline_args.
 */
void setPositionalOption(const std::string &option, int max_count = 1);

/*!
 * Check if the specified command line option is present, and how many times
 *
 * See also \ref cmdline_args.
 */
int hasOption(const std::string &option);

/*!
 * Retrieve the value of a string option
 *
 * See also \ref cmdline_args.
 */
std::string getOptionString(const std::string &option);

/*!
 * Retrieve the value of an int option
 *
 * See also \ref cmdline_args.
 */
int getOptionInt(const std::string &option);

/*!
 * Retrieve the value of an int64_t option
 *
 * See also \ref cmdline_args.
 */
int64_t getOptionInt64(const std::string &option);

/*!
 * Retrieve the value of a double option
 *
 * See also \ref cmdline_args.
 */
double getOptionDouble(const std::string &option);

/*!
 * Retrieve the value of a string vector option
 *
 * See also \ref cmdline_args.
 */
std::vector<std::string> getOptionStringVector(const std::string &option);

/*!
 * Retrieve the value of a int vector option
 *
 * See also \ref cmdline_args.
 */
std::vector<int> getOptionIntVector(const std::string &option);

/*!
 * Retrieve the value of a int64_t vector option
 *
 * See also \ref cmdline_args.
 */
std::vector<int64_t> getOptionInt64Vector(const std::string &option);

/*!
 * Retrieve the value of a double vector option
 *
 * See also \ref cmdline_args.
 */
std::vector<double> getOptionDoubleVector(const std::string &option);

/*!
 * Get the console logging level. This can be changed also by the B0_CONSOLE_LOGLEVEL env var,
 * by the --console-loglevel= command line option.
 */
logger::Level getConsoleLogLevel();

/*!
 * Set the console logging level. This can be changed also by the B0_CONSOLE_LOGLEVEL env var,
 * by the --console-loglevel= command line option.
 */
void setConsoleLogLevel(logger::Level level);

/*!
 * Get the default spin rate (can be changed by the --spin-rate= command line option)
 */
double getSpinRate();

/*!
 * Set the default spin rate (can be changed by the --spin-rate= command line option)
 */
void setSpinRate(double rate);

/*!
 * Return wether quit has requested (by b0::quit() method or by pressing CTRL-C)
 *
 * This function is thread safe
 */
bool quitRequested();

/*!
 * Shutdown all nodes.
 *
 * This will cause all nodes to exit their spin loop cleanly, equivalent to calling
 * b0::Node::shutdown() on all nodes.
 *
 * This function is thread safe
 */
void quit();

} // namespace b0

#endif // B0__B0_H__INCLUDED
