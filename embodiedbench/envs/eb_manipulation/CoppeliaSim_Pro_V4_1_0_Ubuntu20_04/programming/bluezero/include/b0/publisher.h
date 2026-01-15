#ifndef B0__PUBLISHER_H__INCLUDED
#define B0__PUBLISHER_H__INCLUDED

#include <string>

#include <b0/b0.h>
#include <b0/socket.h>
#include <b0/message/message_part.h>

namespace b0
{

class Node;

/*!
 * \brief The publisher class
 *
 * This class wraps a PUB socket. It will automatically connect to the
 * XSUB socket of the proxy (note: the proxy is started by the resolver node).
 *
 * \sa b0::Publisher, b0::Subscriber
 */
class Publisher : public Socket
{
public:
    using logger::LogInterface::log;

    /*!
     * \brief Construct an Publisher child of the specified Node
     */
    Publisher(Node *node, const std::string &topic_name, bool managed = true, bool notify_graph = true);

    /*!
     * \brief Publisher destructor
     */
    virtual ~Publisher();

    /*!
     * \brief Log a message using node's logger, prepending this publisher informations
     */
    void log(logger::Level level, const std::string &message) const override;

    /*!
     * \brief Perform initialization and optionally send graph notify
     */
    virtual void init() override;

    /*!
     * \brief Perform cleanup and optionally send graph notify
     */
    virtual void cleanup() override;

    /*!
     * \brief Return the name of this publisher's topic
     */
    std::string getTopicName();

    /*!
     * \brief Publish a raw multipart message
     */
    virtual void publish(const std::vector<b0::message::MessagePart> &parts);

    /*!
     * \brief Publish a raw message
     */
    virtual void publish(const std::string &msg, const std::string &type = "");

    /*!
     * \brief Publish a message
     */
    template<class TMsg>
    void publish(const TMsg &msg)
    {
        writeMsg(msg);
    }

    /*!
     * \brief Publish a message and any additional raw parts
     */
    template<class TMsg>
    void publish(const TMsg &msg, const std::vector<b0::message::MessagePart> &parts)
    {
        writeMsg(msg, parts);
    }

protected:
    /*!
     * \brief Connect to the remote address
     */
    virtual void connect();

    /*!
     * \brief Disconnect from the remote address
     */
    virtual void disconnect();

    //! If false this socket will not send announcement to resolv (i.e. it will be "invisible")
    const bool notify_graph_;
};

} // namespace b0

#endif // B0__PUBLISHER_H__INCLUDED
