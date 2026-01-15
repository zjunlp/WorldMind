#ifndef B0__MESSAGE__RESOLV__HEARTBEAT_REQUEST_H__INCLUDED
#define B0__MESSAGE__RESOLV__HEARTBEAT_REQUEST_H__INCLUDED

#include <boost/serialization/string.hpp>

#include <b0/b0.h>
#include <b0/message/message.h>

namespace b0
{

namespace message
{

namespace resolv
{

/*!
 * \brief Heartbeat periodically sent by node to resolver
 *
 * \mscfile node-lifetime.msc
 *
 * \sa HeartBeatResponse, \ref protocol
 */
class HeartbeatRequest : public Message
{
public:
    //! The name of the node
    std::string node_name;

public:
    std::string type() const override {return "b0.message.resolv.HeartbeatRequest";}
};

} // namespace resolv

} // namespace message

} // namespace b0

//! \cond HIDDEN_SYMBOLS

namespace spotify
{

namespace json
{

using b0::message::resolv::HeartbeatRequest;

template <>
struct default_codec_t<HeartbeatRequest>
{
    static codec::object_t<HeartbeatRequest> codec()
    {
        auto codec = codec::object<HeartbeatRequest>();
        codec.required("node_name", &HeartbeatRequest::node_name);
        return codec;
    }
};

} // namespace json

} // namespace spotify

//! \endcond

#endif // B0__MESSAGE__RESOLV__HEARTBEAT_REQUEST_H__INCLUDED
