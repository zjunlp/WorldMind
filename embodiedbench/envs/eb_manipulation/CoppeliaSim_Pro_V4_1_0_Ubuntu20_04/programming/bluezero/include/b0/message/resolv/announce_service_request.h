#ifndef B0__MESSAGE__RESOLV__ANNOUNCE_SERVICE_REQUEST_H__INCLUDED
#define B0__MESSAGE__RESOLV__ANNOUNCE_SERVICE_REQUEST_H__INCLUDED

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
 * \brief Sent by ServiceServer to announce a service by some name
 *
 * \mscfile node-startup-service.msc
 *
 * The service name must be unique.
 *
 * \sa AnnounceServiceResponse, \ref protocol
 */
class AnnounceServiceRequest : public Message
{
public:
    //! The name of the node
    std::string node_name;

    //! The name of the service
    std::string service_name;

    //! The address of the zmq socket
    std::string sock_addr;

public:
    std::string type() const override {return "b0.message.resolv.AnnounceServiceRequest";}
};

} // namespace resolv

} // namespace message

} // namespace b0

//! \cond HIDDEN_SYMBOLS

namespace spotify
{

namespace json
{

using b0::message::resolv::AnnounceServiceRequest;

template <>
struct default_codec_t<AnnounceServiceRequest>
{
    static codec::object_t<AnnounceServiceRequest> codec()
    {
        auto codec = codec::object<AnnounceServiceRequest>();
        codec.required("node_name", &AnnounceServiceRequest::node_name);
        codec.required("service_name", &AnnounceServiceRequest::service_name);
        codec.required("sock_addr", &AnnounceServiceRequest::sock_addr);
        return codec;
    }
};

} // namespace json

} // namespace spotify

//! \endcond

#endif // B0__MESSAGE__RESOLV__ANNOUNCE_SERVICE_REQUEST_H__INCLUDED
