#ifndef B0__SOCKET_H__INCLUDED
#define B0__SOCKET_H__INCLUDED

#include <string>

#include <b0/b0.h>
#include <b0/user_data.h>
#include <b0/logger/interface.h>
#include <b0/message/message_part.h>
#include <b0/message/message_envelope.h>
#include <b0/exception/message_unpack_error.h>

#include <boost/function.hpp>
#include <boost/bind.hpp>

namespace b0
{

class Node;

/*!
 * \brief The Socket class
 *
 * This class wraps a ZeroMQ socket. It provides wrappers for reading and writing
 * raw payloads, as well as b0::message::Message messages.
 *
 * \sa b0::Publisher, b0::Subscriber, b0::ServiceClient, b0::ServiceServer
 */
class Socket : public logger::LogInterface, public UserData
{
private:
    //! \cond HIDDEN_SYMBOLS

    struct Private;

    //! \endcond

public:
    /*!
     * \brief Construct a Socket
     *
     * If managed is false, this socket will be unmanaged, meaning that the node
     * will not call init() and cleanup(), and those methods must be called manually
     * at the appropriate time.
     */
    Socket(Node *node, int type, const std::string &name, bool managed = true);

    /*!
     * \brief Socket destructor
     */
    virtual ~Socket();

    /*!
     * \brief Log a message to the default logger of this node
     */
    void log(logger::Level level, const std::string &message) const override;

    /*!
     * \brief Perform initialization (resolve name, connect socket, set subscription)
     */
    virtual void init() = 0;

    /*!
     * \brief Perform cleanup (clear subscription, disconnect socket)
     */
    virtual void cleanup() = 0;

    /*!
     * \brief Process incoming messages and call callbacks
     */
    virtual void spinOnce();

    /*!
     * \brief Set the remote address the socket will connect to
     */
    void setRemoteAddress(const std::string &addr);

    /*!
     * \brief Return the name of the socket bus
     */
    std::string getName() const;

    /*!
     * \brief Return the node owning this socket
     */
    Node & getNode() const;

    /*!
     * \brief Check if this socket name matches the specified pattern.
     *
     *  Example of patterns:
     *  - A `*` pattern always matches.
     *  - A `*.sockName` pattern always matches if sockName matches.
     *  - A `nodeName.*` pattern always matches if nodeName matches.
     *  - A `nodeName.sockName` pattern matches if both the node name and the socket name match.
     */
    bool matchesPattern(const std::string &pattern) const;

private:
    std::unique_ptr<Private> private_;

protected:
    //! The Node owning this Socket
    Node &node_;

    //! This socket bus name
    std::string name_;

    //! This socket bus name prior to any remapping
    std::string orig_name_;

    //! True if this socket is managed (init(), cleanup() are called by the owner Node)
    const bool managed_;

    //! The address of the ZeroMQ socket to connect to (will skip name resolution if given)
    std::string remote_addr_;

public:
    /*!
     * \brief Read a MessageEnvelope from the underlying ZeroMQ socket
     */
    virtual void readRaw(b0::message::MessageEnvelope &env);

    /*!
     * \brief Read a raw multipart payload from the underlying ZeroMQ socket
     */
    virtual void readRaw(std::vector<b0::message::MessagePart> &parts);

    /*!
     * \brief Read a raw payload from the underlying ZeroMQ socket
     */
    virtual void readRaw(std::string &msg);

    /*!
     * \brief Read a raw payload with type from the underlying ZeroMQ socket
     */
    virtual void readRaw(std::string &msg, std::string &type);

    /*!
     * \brief Read a Message from the underlying ZeroMQ socket
     */
    template<class TMsg>
    void readMsg(TMsg &msg)
    {
        std::string str, type;
        readRaw(str, type);
        parse(msg, str, type);
    }

    /*!
     * \brief Read a (multipart) Message from the underlying ZeroMQ socket
     */
    template<class TMsg>
    void readMsg(TMsg &msg, std::vector<b0::message::MessagePart> &parts)
    {
        readRaw(parts);
        parse(msg, parts[0].payload, parts[0].content_type);
        parts.erase(parts.begin());
    }

    /*!
     * \brief Poll for messages. If timeout is 0 return immediately, otherwise wait
     *        for the specified amount of milliseconds.
     */
    virtual bool poll(long timeout = 0);

    /*!
     * \brief Write a MessageEnvelope to the underlying ZeroMQ socket
     */
    virtual void writeRaw(const b0::message::MessageEnvelope &env);

    /*!
     * \brief Write a raw multipart payload to the underlying ZeroMQ socket
     */
    virtual void writeRaw(const std::vector<b0::message::MessagePart> &parts);

    /*!
     * \brief Write a raw payload to the underlying ZeroMQ socket
     */
    virtual void writeRaw(const std::string &msg, const std::string &type = "");

    /*!
     * \brief Write a Message to the underlying ZeroMQ socket
     */
    template<class TMsg>
    void writeMsg(const TMsg &msg)
    {
        std::string str, type;
        serialize(msg, str, type);
        writeRaw(str, type);
    }

    /*!
     * \brief Write a (multipart) Message to the underlying ZeroMQ socket
     */
    template<class TMsg>
    void writeMsg(const TMsg &msg, const std::vector<b0::message::MessagePart> &parts)
    {
        std::vector<b0::message::MessagePart> parts1(parts);
        b0::message::MessagePart part0;
        serialize(msg, part0.payload, part0.content_type);
        part0.compression_algorithm = compression_algorithm_;
        part0.compression_level = compression_level_;
        parts1.insert(parts1.begin(), part0);
        writeRaw(parts1);
    }


public:
    /*!
     * \brief Set compression algorithm and level
     *
     * The messages sent with this socket will be compressed using the specified algorithm.
     * This has no effect on received messages, which will be automatically decompressed
     * using the algorithm specified in the message envelope.
     */
    void setCompression(const std::string &algorithm, int level = -1);

private:
    //! If set, payloads will be encoded using the specified compression algorithm
    //! \sa WriteSocket::setCompression()
    std::string compression_algorithm_;

    //! If a compression algorithm is set, payloads will be encoded using the specified compression level
    //! \sa WriteSocket::setCompression()
    int compression_level_;

public:
    //! (low-level socket option) Get read timeout (in milliseconds, -1 for no timeout)
    int getReadTimeout() const;

    //! (low-level socket option) Set read timeout (in milliseconds, -1 for no timeout)
    void setReadTimeout(int timeout);

    //! (low-level socket option) Get write timeout (in milliseconds, -1 for no timeout)
    int getWriteTimeout() const;

    //! (low-level socket option) Set write timeout (in milliseconds, -1 for no timeout)
    void setWriteTimeout(int timeout);

    //! (low-level socket option) Get linger period (in milliseconds, -1 for no timeout)
    int getLingerPeriod() const;

    //! (low-level socket option) Set linger period (in milliseconds, -1 for no timeout)
    void setLingerPeriod(int period);

    //! (low-level socket option) Get backlog
    int getBacklog() const;

    //! (low-level socket option) Set backlog
    void setBacklog(int backlog);

    //! (low-level socket option) Get immediate flag
    bool getImmediate() const;

    //! (low-level socket option) Set immediate flag
    void setImmediate(bool immediate);

    //! (low-level socket option) Get conflate flag
    bool getConflate() const;

    //! (low-level socket option) Set conflate flag
    void setConflate(bool conflate);

    //! (low-level socket option) Get read high-water-mark
    int getReadHWM() const;

    //! (low-level socket option) Set read high-water-mark
    void setReadHWM(int n);

    //! (low-level socket option) Get write high-water-mark
    int getWriteHWM() const;

    //! (low-level socket option) Set write high-water-mark
    void setWriteHWM(int n);

protected:
    //! Wrapper to zmq::socket_t::connect
    void connect(const std::string &addr);

    //! Wrapper to zmq::socket_t::disconnect
    void disconnect(const std::string &addr);

    //! Wrapper to zmq::socket_t::bind
    void bind(const std::string &addr);

    //! Wrapper to zmq::socket_t::unbind
    void unbind(const std::string &addr);

    //! Wrapper to zmq::socket_t::setsockopt
    void setsockopt(int option, const void *optval, size_t optvallen);

    //! Wrapper to zmq::socket_t::getsockopt
    void getsockopt(int option, void *optval, size_t *optvallen) const;

    //! High level wrapper for setsockopt
    void setIntOption(int option, int value);

    //! High level wrapper for getsockopt
    int getIntOption(int option) const;
};

} // namespace b0

#endif // B0__SOCKET_H__INCLUDED
