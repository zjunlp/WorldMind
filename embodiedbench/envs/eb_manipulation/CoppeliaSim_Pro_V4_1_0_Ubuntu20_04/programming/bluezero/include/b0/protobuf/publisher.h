#ifndef B0__PROTOBUF__PUBLISHER_H__INCLUDED
#define B0__PROTOBUF__PUBLISHER_H__INCLUDED

#include <string>

#include <b0/publisher.h>
#include <b0/protobuf/socket.h>

namespace b0
{

namespace protobuf
{

template<typename TMsg>
class Publisher : public b0::Publisher, public SocketProtobuf
{
public:
    /*!
     * \brief Construct a Publisher child of the specified Node
     */
    Publisher(Node *node, std::string topic_name, bool managed = true, bool notify_graph = true)
        : b0::Publisher(node, topic_name, managed, notify_graph)
    {
    }

    /*!
     * \brief Publish the message on the publisher's topic
     */
    virtual void publish(const TMsg &msg)
    {
        write(this, msg);
    }
};

} // namespace protobuf

} // namespace b0

#endif // B0__PROTOBUF__PUBLISHER_H__INCLUDED
