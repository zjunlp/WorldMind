#ifndef B0__MESSAGE__RESOLV__HEARTBEAT_RESPONSE_H__INCLUDED
#define B0__MESSAGE__RESOLV__HEARTBEAT_RESPONSE_H__INCLUDED

#include <b0/b0.h>
#include <b0/message/message.h>

namespace b0
{

namespace message
{

namespace resolv
{

/*!
 * \brief Response to HeartBeatRequest message
 *
 * \mscfile node-lifetime.msc
 *
 * \sa HeartBeatRequest, \ref protocol
 */
class HeartbeatResponse : public Message
{
public:
    //! True if successful, false if error
    bool ok;

    //! Time stamp of the message
    int64_t time_usec;

public:
    std::string type() const override {return "b0.message.resolv.HeartbeatResponse";}
};

} // namespace resolv

} // namespace message

} // namespace b0

//! \cond HIDDEN_SYMBOLS

namespace spotify
{

namespace json
{

using b0::message::resolv::HeartbeatResponse;

template <>
struct default_codec_t<HeartbeatResponse>
{
    static codec::object_t<HeartbeatResponse> codec()
    {
        auto codec = codec::object<HeartbeatResponse>();
        codec.required("ok", &HeartbeatResponse::ok);
        codec.required("time_usec", &HeartbeatResponse::time_usec);
        return codec;
    }
};

} // namespace json

} // namespace spotify

//! \endcond

#endif // B0__MESSAGE__RESOLV__HEARTBEAT_RESPONSE_H__INCLUDED
