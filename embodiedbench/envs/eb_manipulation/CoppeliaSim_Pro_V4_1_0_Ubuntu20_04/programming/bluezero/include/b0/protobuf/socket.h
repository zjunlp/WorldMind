#ifndef B0__PROTOBUF__SOCKET_H__INCLUDED
#define B0__PROTOBUF__SOCKET_H__INCLUDED

#include <b0/socket.h>
#include <b0/exception/message_pack_error.h>

#include <google/protobuf/message.h>

namespace b0
{

namespace protobuf
{

class SocketProtobuf
{
public:
    /*!
     * \brief Read a google::protobuf::Message from the underlying ZeroMQ socket
     */
    virtual void read(Socket *socket, google::protobuf::Message &msg);

    /*!
     * \brief Write a google::protobuf::Message
     */
    virtual void write(Socket *socket, const google::protobuf::Message &msg);
};

} // namespace protobuf

namespace exception
{

class ProtobufSerializeError : public MessagePackError
{
public:
    ProtobufSerializeError();
};

class ProtobufParseError : public MessageUnpackError
{
public:
    ProtobufParseError();
};

} // namespace exception

} // namespace b0

#endif // B0__PROTOBUF__SOCKET_H__INCLUDED
