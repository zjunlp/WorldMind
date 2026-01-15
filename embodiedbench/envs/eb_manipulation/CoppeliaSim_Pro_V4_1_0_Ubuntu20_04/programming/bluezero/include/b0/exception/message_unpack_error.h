#ifndef B0__EXCEPTION__MESSAGE_UNPACK_ERROR_H__INCLUDED
#define B0__EXCEPTION__MESSAGE_UNPACK_ERROR_H__INCLUDED

#include <b0/b0.h>
#include <b0/exception/exception.h>

namespace b0
{

namespace exception
{

/*!
 * \brief An exception thrown when unpacking/decoding a message fails
 */
class MessageUnpackError : public Exception
{
public:
    /*!
     * \brief Construct a MessageUnpackError exception
     */
    MessageUnpackError(std::string message = "");
};

/*!
 * \brief An exception thrown when a single part message is received, but a multipart message was expected
 */
class MessageMissingHeaderError : public MessageUnpackError
{
public:
    /*!
     * \brief Construct a MessageMissingHeaderError exception
     */
    MessageMissingHeaderError();
};

/*!
 * \brief An exception thrown when a message has too many parts (zmq multipart)
 */
class MessageTooManyPartsError : public MessageUnpackError
{
public:
    /*!
     * \brief Construct a MessageTooManyPartsError exception
     */
    MessageTooManyPartsError();
};

/*!
 * \brief An exception thrown when the zmq header (first part of a multipart message) is not of the expected value
 */
class HeaderMismatch : public MessageUnpackError
{
public:
    /*!
     * \brief Construct a HeaderMismatch exception
     */
    HeaderMismatch(std::string header, std::string expected_header);
};

/*!
 * \brief An exception thrown when an error decoding the message headers occurs
 */
class EnvelopeDecodeError : public MessageUnpackError
{
public:
    /*!
     * \brief Construct a EnvelopeDecodeError exception
     */
    EnvelopeDecodeError();
};

/*!
 * \brief An exception thrown when reading from socket fails
 */
class SocketReadError : public MessageUnpackError
{
public:
    /*!
     * \brief Construct a SocketReadError exception
     */
    SocketReadError();
};

/*!
 * \brief An exception thrown when a message of an unexpected type is received
 */
class MessageTypeMismatch : public MessageUnpackError
{
public:
    /*!
     * \brief Construct a MessageTypeMismatch exception
     */
    MessageTypeMismatch(std::string type, std::string expected_type);
};

} // namespace exception

} // namespace b0

#endif // B0__EXCEPTION__MESSAGE_UNPACK_ERROR_H__INCLUDED
