#ifndef B0__MESSAGE__RESOLV__SHUTDOWN_NODE_REQUEST_H__INCLUDED
#define B0__MESSAGE__RESOLV__SHUTDOWN_NODE_REQUEST_H__INCLUDED

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
 * \brief Sent by node to resolver when shutting down
 *
 * (not really a request, just a notification)
 *
 * \mscfile node-shutdown.msc
 *
 * \sa ShutdownNodeResponse, \ref protocol
 */
class ShutdownNodeRequest : public Message
{
public:
    //! The name of the node
    std::string node_name;

public:
    std::string type() const override {return "b0.message.resolv.ShutdownNodeRequest";}
};

} // namespace resolv

} // namespace message

} // namespace b0

//! \cond HIDDEN_SYMBOLS

namespace spotify
{

namespace json
{

using b0::message::resolv::ShutdownNodeRequest;

template <>
struct default_codec_t<ShutdownNodeRequest>
{
    static codec::object_t<ShutdownNodeRequest> codec()
    {
        auto codec = codec::object<ShutdownNodeRequest>();
        codec.required("node_name", &ShutdownNodeRequest::node_name);
        return codec;
    }
};

} // namespace json

} // namespace spotify

//! \endcond

#endif // B0__MESSAGE__RESOLV__SHUTDOWN_NODE_REQUEST_H__INCLUDED
