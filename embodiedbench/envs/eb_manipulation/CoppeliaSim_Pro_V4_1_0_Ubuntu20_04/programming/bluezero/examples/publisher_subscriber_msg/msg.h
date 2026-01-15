#ifndef B0__EXAMPLES__PUBLISHER_SUBSCRIBER_MSG__MSG_H__INCLUDED
#define B0__EXAMPLES__PUBLISHER_SUBSCRIBER_MSG__MSG_H__INCLUDED

#include <string>
#include <b0/message/message.h>

/*!
 * \example publisher_subscriber_msg/msg.h
 */

//! \cond HIDDEN_SYMBOLS

class MyMsg : public b0::message::Message
{
public:
    std::string greeting;
    int n;

    std::string type() const override {return "MyMsg";}
};

namespace spotify
{

namespace json
{

template <>
struct default_codec_t<MyMsg> {
    static codec::object_t<MyMsg> codec() {
        auto codec = codec::object<MyMsg>();
        codec.required("greeting", &MyMsg::greeting);
        codec.required("n", &MyMsg::n);
        return codec;
    }
};

} // namespace json

} // namespace spotify

//! \endcond

#endif // B0__EXAMPLES__PUBLISHER_SUBSCRIBER_MSG__MSG_H__INCLUDED
