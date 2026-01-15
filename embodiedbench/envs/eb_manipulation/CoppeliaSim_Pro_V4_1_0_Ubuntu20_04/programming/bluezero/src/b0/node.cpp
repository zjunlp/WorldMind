#include <b0/node.h>
#include <b0/publisher.h>
#include <b0/subscriber.h>
#include <b0/service_client.h>
#include <b0/service_server.h>
#include <b0/exceptions.h>
#include <b0/logger/logger.h>
#include <b0/utils/thread_name.h>
#include <b0/utils/env.h>
#include <b0/resolver/client.h>

#include <cstdlib>
#include <boost/chrono.hpp>
#include <boost/thread.hpp>
#include <boost/asio.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

#include <zmq.hpp>

namespace b0
{

struct Node::Private
{
    Private(Node *node, int io_threads)
        : context_(io_threads)
    {
    }

    zmq::context_t context_;
};

struct Node::Private2
{
    Private2(Node *node)
        : resolv_cli_(node)
    {
    }

    resolver::Client resolv_cli_;
};

Node::Node(const std::string &nodeName)
    : private_(new Private(this, 1)),
      private2_(new Private2(this)),
      name_(Global::getInstance().getRemappedNodeName(*this, nodeName)),
      orig_name_(nodeName),
      state_(NodeState::Created),
      thread_id_(boost::this_thread::get_id()),
      p_logger_(new logger::Logger(this)),
      shutdown_flag_(false),
      minimum_heartbeat_interval_(0),
      spin_rate_(-1)
{
    set_thread_name("main");

    if(!Global::getInstance().isInitialized())
        throw std::runtime_error("b0::init() must be called first");
}

Node::~Node()
{
    if(logger::Logger *p_logger = dynamic_cast<logger::Logger*>(p_logger_))
        delete p_logger;
}

void Node::setResolverAddress(const std::string &addr)
{
    // This is mostly needed for implementing the resolver node;
    // but can be used to connect to a different resolver as well.
    // Use this before Node::init().
    resolv_addr_ = addr;
    if(!resolv_addr_.empty()) private2_->resolv_cli_.setRemoteAddress(resolv_addr_);
}

void Node::init()
{
    NodeState state = state_.load();
    if(state != NodeState::Created)
        throw exception::InvalidStateTransition("init", state);

    if(Global::getInstance().remapNodeName(*this, orig_name_, name_))
        info("Node name '%s' remapped to '%s'", orig_name_, name_);

    debug("Initialization...");

    private2_->resolv_cli_.init(); // resolv_cli_ is not managed

    announceNode();

    if(minimum_heartbeat_interval_ > 0)
        startHeartbeatThread();

    debug("Initializing sockets...");
    for(auto socket : sockets_)
        socket->init();

    state_.store(NodeState::Ready);

    debug("Initialization complete.");
}

void Node::shutdown()
{
    NodeState state = state_.load();
    if(state != NodeState::Ready)
        throw exception::InvalidStateTransition("shutdown", state);

    debug("Shutting down...");

    shutdown_flag_.store(true);

    debug("Shutting complete.");
}

bool Node::shutdownRequested() const
{
    return shutdown_flag_.load() || quitRequested();
}

void Node::spinOnce()
{
    NodeState state = state_.load();
    if(state != NodeState::Ready)
        throw exception::InvalidStateTransition("spinOnce", state);

    // spin sockets:
    for(auto socket : sockets_)
        socket->spinOnce();
}

void Node::spin(boost::function<void(void)> callback, double spinRate)
{
    NodeState state = state_.load();
    if(state != NodeState::Ready)
        throw exception::InvalidStateTransition("spin", state);

    if(spinRate <= 0)
        spinRate = getSpinRate();

    info("Node spinning...");

    while(!shutdownRequested())
    {
        int64_t sleep_period = 1000000. / spinRate;

        auto checkSpinRate = [=](const char *what, int64_t last_time, int64_t current_time = -1)
        {
            if(current_time == -1)
                current_time = hardwareTimeUSec();
            if(current_time - last_time > sleep_period)
                warn("%s took more than %ldusec. Failing to achieve desired "
                        "spin rate of %fHz!", what, sleep_period, spinRate);
            return current_time;
        };

        int64_t t0 = hardwareTimeUSec();

        spinOnce();

        int64_t t1 = checkSpinRate("spinOnce()", t0);

        if(!callback.empty())
            callback();

        int64_t t2 = checkSpinRate("spin()'s callback", t1);

        // compensate for time spent in spin and callbacks:
        sleep_period -= t2 - t0;
        checkSpinRate("spinOnce() together with spin()'s callback", t0, t2);

        responsiveSleepUSec(sleep_period);
    }

    info("spin() finished");
}

void Node::cleanup()
{
    NodeState state = state_.load();
    if(state != NodeState::Ready)
        throw exception::InvalidStateTransition("cleanup", state);

    shutdown_flag_.store(true);

    // stop the heartbeat_thread so that the last zmq socket will be destroyed
    // and we avoid an unclean exit (zmq::error_t: Context was terminated)
    if(minimum_heartbeat_interval_ > 0)
        stopHeartbeatThread();

    debug("Cleanup sockets...");
    for(auto socket : sockets_)
        socket->cleanup();

    // inform resolver that we are shutting down
    notifyShutdown();

    private2_->resolv_cli_.cleanup(); // resolv_cli_ is not managed

    state_.store(NodeState::Terminated);
}

void Node::log(logger::Level level, const std::string &message) const
{
    if(boost::this_thread::get_id() != thread_id_)
        throw exception::Exception("cannot call Node::log() from another thread");

    p_logger_->log(level, message);
}

void Node::startHeartbeatThread()
{
    trace("Starting heartbeat thread...");
    heartbeat_thread_ = boost::thread(&Node::heartbeatLoop, this);
}

void Node::stopHeartbeatThread()
{
    trace("Stopping heartbeat thread...");
    heartbeat_thread_.interrupt();
    heartbeat_thread_.join();
}

std::string Node::getName() const
{
    // return this node's name, used to address sockets (together with the socket name).
    // we get this value from the resolver node, during the announceNode() phase.
    return name_;
}

NodeState Node::getState() const
{
    return state_.load();
}

void * Node::getContext()
{
    return &private_->context_;
}

std::string Node::getXPUBSocketAddress() const
{
    return xpub_sock_addr_;
}

std::string Node::getXSUBSocketAddress() const
{
    return xsub_sock_addr_;
}

void Node::addSocket(Socket *socket)
{
    NodeState state = state_.load();
    if(state != NodeState::Created)
        throw exception::Exception("Cannot create a socket with an already initialized node");

    sockets_.insert(socket);
}

void Node::removeSocket(Socket *socket)
{
    sockets_.erase(socket);
}

std::string Node::hostname() const
{
    return b0::env::get("B0_HOST_ID", boost::asio::ip::host_name());
}

int Node::pid()
{
    return ::getpid();
}

std::string Node::threadID()
{
    return boost::lexical_cast<std::string>(thread_id_);
}

int Node::freeTCPPort()
{
    // by binding the OS socket to port 0, an available port number will be used
    boost::asio::ip::tcp::endpoint ep(boost::asio::ip::tcp::v4(), 0);
    boost::asio::io_service io_service;
    boost::asio::ip::tcp::socket socket(io_service, ep);
    return socket.local_endpoint().port();
}

void Node::notifyTopic(const std::string &topic_name, bool reverse, bool active)
{
    resolver::Client &resolv_cli_ = private2_->resolv_cli_;
    resolv_cli_.notifyTopic(topic_name, reverse, active);
}

void Node::notifyService(const std::string &service_name, bool reverse, bool active)
{
    resolver::Client &resolv_cli_ = private2_->resolv_cli_;
    resolv_cli_.notifyService(service_name, reverse, active);
}

void Node::announceService(const std::string &service_name, const std::string &addr)
{
    resolver::Client &resolv_cli_ = private2_->resolv_cli_;
    resolv_cli_.announceService(service_name, addr);
}

void Node::resolveService(const std::string &service_name, std::string &addr)
{
    resolver::Client &resolv_cli_ = private2_->resolv_cli_;
    resolv_cli_.resolveService(service_name, addr);
}

void Node::setAnnounceTimeout(int timeout)
{
    resolver::Client &resolv_cli_ = private2_->resolv_cli_;
    resolv_cli_.setAnnounceTimeout(timeout);
}

std::string Node::freeTCPAddress()
{
    boost::format fmt("tcp://%s:%d");
    return (fmt % hostname() % freeTCPPort()).str();
}

void Node::announceNode()
{
    private2_->resolv_cli_.announceNode(hostname(), pid(), name_, xpub_sock_addr_, xsub_sock_addr_, minimum_heartbeat_interval_);

    if(logger::Logger *p_logger = dynamic_cast<logger::Logger*>(p_logger_))
        p_logger->connect(xsub_sock_addr_);
}

void Node::notifyShutdown()
{
    private2_->resolv_cli_.notifyShutdown();
}

void Node::heartbeatLoop()
{
    set_thread_name("HB");
    b0::logger::LocalLogger logger(this);
    logger.trace("HB: started");

    while(!shutdownRequested())
    {
        try
        {
            resolver::Client resolv_cli(this);
            resolv_cli.setReadTimeout(minimum_heartbeat_interval_ / 3000);
            resolv_cli.init();

            while(!shutdownRequested())
            {
                int64_t time_usec;
                resolv_cli.sendHeartbeat(&time_usec);
                time_sync_.updateTime(time_usec);
                sleepUSec(minimum_heartbeat_interval_ / 3);
            }

            resolv_cli.cleanup();
        }
        catch(std::exception &ex)
        {
            logger.error("HB: %s", ex.what());
            sleepUSec(minimum_heartbeat_interval_ / 3);
        }
    }

    logger.trace("HB: finished");
}

int64_t Node::hardwareTimeUSec() const
{
    return time_sync_.hardwareTimeUSec();
}

int64_t Node::timeUSec()
{
    return time_sync_.timeUSec();
}

void Node::setTimesyncMaxSlope(double max_slope)
{
    time_sync_.setMaxSlope(max_slope);
}

void Node::sleepUSec(int64_t usec)
{
    boost::this_thread::sleep_for(boost::chrono::microseconds{usec});
}

void Node::responsiveSleepUSec(int64_t usec)
{
    int64_t until = hardwareTimeUSec() + usec;
    int64_t max_sleep = 100000; // 100ms
    while(1)
    {
        if(shutdownRequested()) return;
        int64_t remaining = until - hardwareTimeUSec();
        int64_t sleep = std::min(max_sleep, remaining);
        boost::this_thread::sleep_for(boost::chrono::microseconds{sleep});
        if(remaining <= max_sleep) break;
    }
}

void Node::setSpinRate(double rate)
{
    spin_rate_ = rate;
}

double Node::getSpinRate()
{
    return spin_rate_ > 0 ? spin_rate_ : b0::getSpinRate();
}

} // namespace b0

