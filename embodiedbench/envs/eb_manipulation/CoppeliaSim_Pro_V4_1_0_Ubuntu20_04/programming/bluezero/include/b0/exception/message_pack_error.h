#ifndef B0__EXCEPTION__MESSAGE_PACK_ERROR_H__INCLUDED
#define B0__EXCEPTION__MESSAGE_PACK_ERROR_H__INCLUDED

#include <b0/b0.h>
#include <b0/exception/exception.h>

namespace b0
{

namespace exception
{

/*!
 * \brief An exception thrown when encoding/packing a message fails
 */
class MessagePackError : public Exception
{
public:
    /*!
     * \brief Construct a MessagePackError exception
     */
    MessagePackError(std::string message = "");
};

/*!
 * \brief An exception thrown when an error encoding the message headers occurs
 */
class EnvelopeEncodeError : public MessagePackError
{
public:
    /*!
     * \brief Construct a EnvelopeEncodeError exception
     */
    EnvelopeEncodeError();
};

/*!
 * \brief An exception thrown when writing to socket fails
 */
class SocketWriteError : public MessagePackError
{
public:
    /*!
     * \brief Construct a SocketWriteError exception
     */
    SocketWriteError();
};

} // namespace exception

} // namespace b0

#endif // B0__EXCEPTION__MESSAGE_PACK_ERROR_H__INCLUDED
