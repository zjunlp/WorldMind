#ifndef B0__MESSAGE__MESSAGE_H__INCLUDED
#define B0__MESSAGE__MESSAGE_H__INCLUDED

#include <string>
#include <boost/format.hpp>
#include <spotify/json.hpp>
#include <spotify/json/codec/boost.hpp>

#include <b0/b0.h>
#include <b0/exception/message_unpack_error.h>

namespace b0
{

namespace message
{

/*!
 * \brief The base class for all BlueZero's protocol messages
 *
 * It contains some utility methods for string serialization/parsing.
 */
class Message
{
public:
    //! \brief destructor
    virtual ~Message();

    //! \brief Returns a string with the type (typically the name of the class)
    virtual std::string type() const = 0;
};

/*!
 * \brief Parse a message from a string
 */
template<class TMsg>
void parse(TMsg &msg, const std::string &s)
{
    if(!spotify::json::try_decode(msg, s))
        throw exception::MessageUnpackError("json parse error");
}

/*!
 * \brief Parse a message from a string
 */
template<class TMsg>
void parse(TMsg &msg, const std::string &s, const std::string &type)
{
    if(type != msg.type())
        throw exception::MessageUnpackError((boost::format("bad content type: got %s, expected %s") % type % msg.type()).str());
    if(!spotify::json::try_decode(msg, s))
        throw exception::MessageUnpackError("json parse error");
}

/*!
 * \brief Serialize a message to a string
 */
template<class TMsg>
void serialize(const TMsg &msg, std::string &s)
{
    s = spotify::json::encode(msg);
}

/*!
 * \brief Serialize a message to a string
 */
template<class TMsg>
void serialize(const TMsg &msg, std::string &s, std::string &type)
{
    s = spotify::json::encode(msg);
    type = msg.type();
}

} // namespace message

} // namespace b0

#endif // B0__MESSAGE__MESSAGE_H__INCLUDED
