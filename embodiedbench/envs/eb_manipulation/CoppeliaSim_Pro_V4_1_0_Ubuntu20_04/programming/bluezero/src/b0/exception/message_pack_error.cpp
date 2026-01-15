#include <b0/exception/message_pack_error.h>

namespace b0
{

namespace exception
{

MessagePackError::MessagePackError(std::string message)
    : Exception(message)
{
}

EnvelopeEncodeError::EnvelopeEncodeError()
    : MessagePackError("Failed to encode message envelope (serialize error)")
{
}

SocketWriteError::SocketWriteError()
    : MessagePackError("Socket write error (send() failed)")
{
}

} // namespace exception

} // namespace b0

