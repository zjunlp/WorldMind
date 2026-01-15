#ifndef B0__MESSAGE__RESOLV__RESOLVE_SERVICE_RESPONSE_H__INCLUDED
#define B0__MESSAGE__RESOLV__RESOLVE_SERVICE_RESPONSE_H__INCLUDED

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
 * \brief Response to ResolveServiceRequest message
 *
 * \mscfile service-resolve.msc
 *
 * \sa ResolveServiceRequest, \ref protocol
 */
class ResolveServiceResponse : public Message
{
public:
    //! True if successful, false if error (i.e. does not exist)
    bool ok;

    //! The name of the zmq socket
    std::string sock_addr;

public:
    std::string type() const override {return "ResolveServiceResponse";}
};

} // namespace resolv

} // namespace message

} // namespace b0

//! \cond HIDDEN_SYMBOLS

namespace spotify
{

namespace json
{

using b0::message::resolv::ResolveServiceResponse;

template <>
struct default_codec_t<ResolveServiceResponse>
{
    static codec::object_t<ResolveServiceResponse> codec()
    {
        auto codec = codec::object<ResolveServiceResponse>();
        codec.required("ok", &ResolveServiceResponse::ok);
        codec.required("sock_addr", &ResolveServiceResponse::sock_addr);
        return codec;
    }
};

} // namespace json

} // namespace spotify

//! \endcond

#endif // B0__MESSAGE__RESOLV__RESOLVE_SERVICE_RESPONSE_H__INCLUDED
