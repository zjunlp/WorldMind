#include <b0/exception/message_unpack_error.h>

#include <boost/format.hpp>

namespace b0
{

namespace exception
{

MessageUnpackError::MessageUnpackError(std::string message)
    : Exception(message)
{
}

MessageMissingHeaderError::MessageMissingHeaderError()
    : MessageUnpackError("Message header (topic) is missing (multipart message expected)")
{
}

MessageTooManyPartsError::MessageTooManyPartsError()
    : MessageUnpackError("Too many message parts")
{
}

HeaderMismatch::HeaderMismatch(std::string header, std::string expected_header)
    : MessageUnpackError((boost::format("Message header (topic) does not match (got '%s', expected '%s')") % header % expected_header).str())
{
}

EnvelopeDecodeError::EnvelopeDecodeError()
    : MessageUnpackError("Failed to decode message envelope (parse error)")
{
}

SocketReadError::SocketReadError()
    : MessageUnpackError("Socket read error (recv() failed)")
{
}

MessageTypeMismatch::MessageTypeMismatch(std::string type, std::string expected_type)
    : MessageUnpackError((boost::format("Message type mismatch (got '%s', expected '%s')") % type % expected_type).str())
{
}

} // namespace exception

} // namespace b0

