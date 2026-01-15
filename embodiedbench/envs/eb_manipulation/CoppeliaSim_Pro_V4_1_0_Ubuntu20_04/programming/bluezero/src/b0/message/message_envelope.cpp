#include <b0/message/message_envelope.h>
#include <b0/exception/message_unpack_error.h>
#include <b0/compress/compress.h>

#include <vector>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

namespace b0
{

namespace message
{

void parse(MessageEnvelope &env, const std::string &s)
{
    size_t content_begin = s.find("\n\n");
    if(content_begin == std::string::npos)
        throw exception::EnvelopeDecodeError();
    std::string message_headers = s.substr(0, content_begin);
    std::string payload = s.substr(content_begin + 2);
    std::vector<std::string> headers_split;
    boost::split(headers_split, message_headers, boost::is_any_of("\n"));
    env.header0 = headers_split.at(0);
    headers_split.erase(headers_split.begin());
    for(auto &header_line : headers_split)
    {
        size_t delim_pos = header_line.find(": ");
        if(delim_pos == std::string::npos)
            throw exception::EnvelopeDecodeError();
        std::string key = header_line.substr(0, delim_pos),
            value = header_line.substr(delim_pos + 2);
        env.headers[key] = value;
    }
    try
    {
        auto part_count_it = env.headers.find("Part-count");
        if(part_count_it == env.headers.end()) throw;
        int part_count = boost::lexical_cast<int>(part_count_it->second);
        env.headers.erase(part_count_it);
        env.parts.resize(part_count);
    }
    catch(...) {throw exception::EnvelopeDecodeError();}
    int part_start = 0;
    for(int i = 0; i < env.parts.size(); i++)
    {
        auto content_type_it = env.headers.find((boost::format("Content-type-%d") % i).str());
        if(content_type_it != env.headers.end())
        {
            env.parts[i].content_type = content_type_it->second;
            env.headers.erase(content_type_it);
        }

        auto compression_algorithm_it = env.headers.find((boost::format("Compression-algorithm-%d") % i).str());
        if(compression_algorithm_it != env.headers.end())
        {
            env.parts[i].compression_algorithm = compression_algorithm_it->second;
            env.headers.erase(compression_algorithm_it);
        }

        auto compression_level_it = env.headers.find((boost::format("Compression-level-%d") % i).str());
        if(compression_level_it != env.headers.end())
        {
            env.parts[i].compression_level = boost::lexical_cast<int>(compression_level_it->second);
            env.headers.erase(compression_level_it);
        }

        auto uncompressed_content_length_it = env.headers.find((boost::format("Uncompressed-content-length-%d") % i).str());
        int uncompressed_content_length = -1;
        if(uncompressed_content_length_it != env.headers.end())
        {
            uncompressed_content_length = boost::lexical_cast<int>(uncompressed_content_length_it->second);
            env.headers.erase(uncompressed_content_length_it);
        }

        auto content_length_it = env.headers.find((boost::format("Content-length-%d") % i).str());
        int content_length = -1;
        if(content_length_it != env.headers.end())
        {
            content_length = boost::lexical_cast<int>(content_length_it->second);
            env.headers.erase(content_length_it);
        }

        if(content_length == -1)
            throw exception::EnvelopeDecodeError();

        env.parts[i].payload = b0::compress::decompress(env.parts[i].compression_algorithm, payload.substr(part_start, content_length), uncompressed_content_length);
        part_start += content_length;
    }
}

void serialize(const MessageEnvelope &env, std::string &s)
{
    std::stringstream ss;

    ss << env.header0 << std::endl;
    ss << "Part-count: " << env.parts.size() << std::endl;
    std::vector<std::string> compressed_payloads;
    int total_length = 0;
    for(size_t i = 0; i < env.parts.size(); i++)
    {
        std::string compressed_payload = b0::compress::compress(env.parts[i].compression_algorithm, env.parts[i].payload, env.parts[i].compression_level);
        compressed_payloads.push_back(compressed_payload);
        total_length += compressed_payload.size();

        ss << "Content-length-" << i << ": " << compressed_payload.size() << std::endl;
        if(env.parts[i].content_type != "")
            ss << "Content-type-" << i << ": " << env.parts[i].content_type << std::endl;
        if(env.parts[i].compression_algorithm != "")
        {
            ss << "Compression-algorithm-" << i << ": " << env.parts[i].compression_algorithm << std::endl;
            ss << "Uncompressed-content-length-" << i << ": " << env.parts[i].payload.size() << std::endl;
            if(env.parts[i].compression_level > 0)
                ss << "Compression-level-" << i << ": " << env.parts[i].compression_level << std::endl;
        }
    }
    ss << "Content-length: " << total_length << std::endl;

    for(auto &pair : env.headers)
        ss << pair.first << ": " << pair.second << std::endl;

    ss << std::endl;

    for(auto &payload : compressed_payloads)
        ss << payload;

    s = ss.str();
}

} // namespace message

} // namespace b0

