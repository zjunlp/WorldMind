#ifndef B0__MESSAGE__RESOLV__ANNOUNCE_SERVICE_RESPONSE_H__INCLUDED
#define B0__MESSAGE__RESOLV__ANNOUNCE_SERVICE_RESPONSE_H__INCLUDED

#include <b0/b0.h>
#include <b0/message/message.h>

namespace b0
{

namespace message
{

namespace resolv
{

/*!
 * \brief Response to AnnounceServiceRequest message
 *
 * \mscfile node-startup-service.msc
 *
 * \sa AnnounceServiceRequest, \ref protocol
 */
class AnnounceServiceResponse : public Message
{
public:
    //! True if successful, false if error
    bool ok;

public:
    std::string type() const override {return "b0.message.resolv.AnnounceServiceResponse";}
};

} // namespace resolv

} // namespace message

} // namespace b0

//! \cond HIDDEN_SYMBOLS

namespace spotify
{

namespace json
{

using b0::message::resolv::AnnounceServiceResponse;

template <>
struct default_codec_t<AnnounceServiceResponse>
{
    static codec::object_t<AnnounceServiceResponse> codec()
    {
        auto codec = codec::object<AnnounceServiceResponse>();
        codec.required("ok", &AnnounceServiceResponse::ok);
        return codec;
    }
};

} // namespace json

} // namespace spotify

//! \endcond

#endif // B0__MESSAGE__RESOLV__ANNOUNCE_SERVICE_RESPONSE_H__INCLUDED
