#ifndef B0__NODE_H__INCLUDED
#define B0__NODE_H__INCLUDED

#include <b0/b0.h>
#include <b0/user_data.h>
#include <b0/node_state.h>
#include <b0/socket.h>
#include <b0/logger/interface.h>
#include <b0/utils/time_sync.h>

#include <atomic>
#include <set>
#include <string>

#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>

namespace b0
{

namespace logger
{

class Logger;

} // namespace logger

/*!
 * \brief The abstraction for a node in the network.
 *
 * You must create at most one node per thread.
 * You can have many nodes in one process by creating several threads.
 */
class Node : public logger::LogInterface, public UserData
{
private:
    //! \cond HIDDEN_SYMBOLS

    struct Private;
    struct Private2;

    //! \endcond

public:
    using logger::LogInterface::log;

    /*!
     * \brief Create a node with a given name.
     * \param nodeName the name of the node
     * \sa getName()
     *
     * Create a node with a given name.
     *
     * A message will be send to resolver to announce this node presence on the network.
     * If a node with the same name already exists in the network, this node will get
     * a different name.
     *
     * The heartbeat thread will be started to let resolver track the online status of this node.
     */
    Node(const std::string &nodeName = "");

    /*!
     * \brief Destruct this node
     *
     * Any threads, such as the heartbeat thread, will be stopped, and the sockets
     * will be freeed.
     */
    virtual ~Node();

    /*!
     * \brief Specify a different value for the resolver address. Otherwise B0_RESOLVER env var is used.
     */
    void setResolverAddress(const std::string &addr);

    /*!
     * \brief Initialize the node (connect to resolve, start heartbeat, announce node name)
     *
     * If you need to extend the init phase, when overriding it in your
     * subclass, remember to first call this Node::init() (unless you know what
     * you are doing).
     */
    virtual void init();

    /*!
     * \brief Shutdown the node (stop all running threads, send shutdown notification)
     *
     * If you need to perform additional cleanup, when overriding this method
     * in your subclass, remember to first call this Node::shutdown() (unless you know
     * what you are doing).
     *
     * This method is thread-safe.
     */
    virtual void shutdown();

    /*!
     * \brief Return wether shutdown has requested (by Node::shutdown() method or by pressing CTRL-C)
     *
     * This method is thread-safe.
     */
    bool shutdownRequested() const;

    /*!
     * \brief Read all available messages from the various ZeroMQ sockets, and
     * dispatch them to callbacks.
     *
     * This method will call b0::Subscriber::spinOnce() and b0::ServiceServer::spinOnce()
     * on the subscribers and service servers that belong to this node.
     *
     * Warning: every message sent on a topic which has no registered callback will be discarded.
     */
    virtual void spinOnce();

    /*!
     * \brief Run the spin loop
     *
     * This will continuously call spinOnce() and the specified callback, at the specified rate.
     *
     * If the spinRate parameter is not specified, the value returned by b0::Node::getSpinRate()
     * will be used.
     *
     * \param callback a callback to be called each time after spinOnce()
     * \param spinRate the approximate frequency (in Hz) at which spinOnce() will be called
     */
    virtual void spin(boost::function<void(void)> callback = {}, double spinRate = -1);

    /*!
     * \brief Node cleanup: stop all threads, send a shutdown notification to resolver, and so on...
     */
    virtual void cleanup();

protected:
    /*!
     * \brief Start the heartbeat thread
     *
     * The heartbeat thread will periodically send a heartbeat message to inform the
     * resolver node that this node is alive.
     */
    virtual void startHeartbeatThread();

    /*!
     * \brief Stop the heartbeat thread
     */
    virtual void stopHeartbeatThread();

public:
    /*!
     * \brief Log a message to the default logger of this node
     */
    void log(logger::Level level, const std::string &message) const override;

    /*!
     * \brief Get the name assigned by resolver to this node
     */
    std::string getName() const;

    /*!
     * \brief Get the state of this node
     *
     * This method is thread-safe.
     */
    NodeState getState() const;

    /*!
     * \brief Get the ZeroMQ Context
     */
    void * getContext();

    /*!
     * \brief Retrieve address of the proxy's XPUB socket
     */
    virtual std::string getXPUBSocketAddress() const;

    /*!
     * \brief Retrieve address of the proxy's XSUB socket
     */
    virtual std::string getXSUBSocketAddress() const;

private:
    /*!
     * Register a socket for this node. Do not call this directly. Called by Socket class.
     */
    void addSocket(Socket *socket);

    /*!
     * Register a socket for this node. Do not call this directly. Called by Socket class.
     */
    void removeSocket(Socket *socket);

public:
    /*!
     * \brief Return the public address (IP or hostname) to reach this node on the network
     */
    virtual std::string hostname() const;

    /*!
     * \brief Return the process id of this node
     */
    virtual int pid();

    /*!
     * \brief Return the thread identifier of this node.
     */
    virtual std::string threadID();

    /*!
     * \brief Find and return an available TCP port
     */
    virtual int freeTCPPort();

    /*!
     * \brief Notify topic publishing/subscription start or end
     */
    virtual void notifyTopic(const std::string &topic_name, bool reverse, bool active);

    /*!
     * \brief Notify service advertising/use start or end
     */
    virtual void notifyService(const std::string &service_name, bool reverse, bool active);

    /*!
     * \brief Announce service address
     */
    virtual void announceService(const std::string &service_name, const std::string &addr);

    /*!
     * \brief Resolve service address by name
     */
    virtual void resolveService(const std::string &service_name, std::string &addr);

    /*!
     * \brief Set the timeout for the announce phase. See b0::resolver::Client::setAnnounceTimeout()
     */
    virtual void setAnnounceTimeout(int timeout = -1);

protected:
    /*!
     * \brief Find and return an available tcp address, e.g. tcp://hostname:portnumber
     */
    virtual std::string freeTCPAddress();

    /*!
     * \brief Announce this node to resolver
     */
    virtual void announceNode();

    /*!
     * \brief Notify resolver of this node shutdown
     */
    virtual void notifyShutdown();

    /*!
     * \brief The heartbeat message loop (run in its own thread)
     */
    virtual void heartbeatLoop();

public:
    /*!
     * \brief Return this computer's clock time in microseconds
     *
     * This method is thread-safe.
     */
    int64_t hardwareTimeUSec() const;

    /*!
     * \brief Return the adjusted time in microseconds. See \ref timesync for details.
     *
     * This method is thread-safe.
     */
    int64_t timeUSec();

    /*!
     * \brief Sleep for the specified amount of microseconds
     *
     * This method is thread-safe.
     */
    void sleepUSec(int64_t usec);

    /*!
     * \brief Sleep for the specified amount of microseconds, but be responsive of shutdown event
     *
     * This method is thread-safe.
     */
    void responsiveSleepUSec(int64_t usec);

    /*!
     * \brief Set the default spin rate
     */
    void setSpinRate(double rate);

    /*!
     * \brief Get the default spin rate
     *
     * If a spin rate has been previously specified with b0::Node::setSpinRate(), that
     * will be returned, otherwise it will return the value from b0::getSpinRate().
     */
    double getSpinRate();

    /*!
     * \brief Set time synchronization maximum slope
     */
    void setTimesyncMaxSlope(double max_slope);

private:
    std::unique_ptr<Private> private_;
    std::unique_ptr<Private2> private2_;

protected:
    //! Target address of resolver client
    std::string resolv_addr_;

    //! The logger of this node
    logger::LogInterface *p_logger_;

private:
    //! Name of this node as it has been assigned by resolver
    std::string name_;

    //! The requested name of this node prior to any remapping
    std::string orig_name_;

    //! State of this node
    std::atomic<NodeState> state_;

    //! Id of the thread in which this node has been created
    boost::thread::id thread_id_;

    //! Heartbeat thread
    boost::thread heartbeat_thread_;

    //! List of sockets
    std::set<Socket*> sockets_;

    //! Address of the proxy's XSUB socket
    std::string xsub_sock_addr_;

    //! Address of the proxy's XPUB socket
    std::string xpub_sock_addr_;

    //! Flag set by Node::shutdown()
    std::atomic<bool> shutdown_flag_;

    //! Minimum heartbeat send interval (0 means not required to send)
    int64_t minimum_heartbeat_interval_;

    //! Time synchronization object
    TimeSync time_sync_;

    //! Node's default spin rate
    double spin_rate_;

public:
    friend class Socket;
};

} // namespace b0

#endif // B0__NODE_H__INCLUDED
