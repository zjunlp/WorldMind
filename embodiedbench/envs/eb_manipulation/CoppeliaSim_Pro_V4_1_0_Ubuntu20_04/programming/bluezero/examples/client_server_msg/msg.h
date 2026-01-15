#ifndef B0__EXAMPLES__CLIENT_SERVER_MSG__MSG_H__INCLUDED
#define B0__EXAMPLES__CLIENT_SERVER_MSG__MSG_H__INCLUDED

#include <b0/message/message.h>

/*!
 * \example client_server_msg/msg.h
 */

//! \cond HIDDEN_SYMBOLS

class AddRequest : public b0::message::Message
{
public:
    int a, b;

    std::string type() const override {return "AddRequest";}
};

class AddReply : public b0::message::Message
{
public:
    int c;

    std::string type() const override {return "AddReply";}
};

namespace spotify
{

namespace json
{

template <>
struct default_codec_t<AddRequest> {
    static codec::object_t<AddRequest> codec() {
        auto codec = codec::object<AddRequest>();
        codec.required("a", &AddRequest::a);
        codec.required("b", &AddRequest::b);
        return codec;
    }
};

template <>
struct default_codec_t<AddReply> {
    static codec::object_t<AddReply> codec() {
        auto codec = codec::object<AddReply>();
        codec.required("c", &AddReply::c);
        return codec;
    }
};

} // namespace json

} // namespace spotify

//! \endcond

#endif // B0__EXAMPLES__CLIENT_SERVER_MSG__MSG_H__INCLUDED
