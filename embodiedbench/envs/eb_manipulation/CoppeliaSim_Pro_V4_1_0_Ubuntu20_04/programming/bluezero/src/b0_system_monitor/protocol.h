#ifndef B0__PROCESS_MANAGER__PROTOCOL_H__INCLUDED
#define B0__PROCESS_MANAGER__PROTOCOL_H__INCLUDED

#include <vector>
#include <b0/message/message.h>

namespace b0
{

namespace system_monitor
{

class Load : public b0::message::Message
{
public:
    std::vector<float> load_averages;
    int free_memory;

    std::string type() const override {return "b0::system_monitor::Load";}
};

} // namespace system_monitor

} // namespace b0

namespace spotify
{

namespace json
{

template <>
struct default_codec_t<b0::system_monitor::Load> {
    static codec::object_t<b0::system_monitor::Load> codec() {
        auto codec = codec::object<b0::system_monitor::Load>();
        codec.required("load_averages", &b0::system_monitor::Load::load_averages);
        codec.required("free_memory", &b0::system_monitor::Load::free_memory);
        return codec;
    }
};

} // namespace json

} // namespace spotify

#endif // B0__PROCESS_MANAGER__PROTOCOL_H__INCLUDED
