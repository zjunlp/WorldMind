#include <string>
#include <iostream>
#include <spotify/json.hpp>
#include <spotify/json/codec/boost.hpp>

struct MyMessage
{
    std::string required;
    boost::optional<std::string> optional;
};

namespace spotify {
namespace json {
template <>
struct default_codec_t<MyMessage> {
  static codec::object_t<MyMessage> codec() {
    auto codec = codec::object<MyMessage>();
    codec.required("required", &MyMessage::required);
    codec.optional("optional", &MyMessage::optional);
    return codec;
  }
};
} // namespace json
} // namespace spotify

int main(int argc, char **argv)
{
    const auto msg = spotify::json::decode<MyMessage>(R"({ "required": "1", "zoptional": "foo" })");
    MyMessage msg2;
    msg2.optional = "bar";
    const auto json = spotify::json::encode(msg);
    std::cout << "Re-encoded:" << std::endl << json << std::endl;
    return 0;
}

