#include <stdexcept>
#include <cstring>

#include <boost/format.hpp>

#include <b0/exceptions.h>
#include <b0/compress/lz4.h>

#ifdef LZ4_FOUND

#include <lz4.h>

namespace b0
{

namespace compress
{

std::string lz4_compress(const std::string &str, int level)
{
    std::string ret;
    ret.reserve(LZ4_compressBound(str.size()));
    int bytesWritten = LZ4_compress_default(str.data(), (char*)ret.data(), str.size(), ret.capacity());
    if(!bytesWritten)
        throw exception::Exception("lz4 compress failed");
    ret.assign(ret.data(), bytesWritten);
    return ret;
}

std::string lz4_decompress(const std::string &str, size_t size)
{
    std::string ret;
    ret.reserve(size ? size : str.size() * 10);
    int bytesWritten = LZ4_decompress_safe(str.data(), (char*)ret.data(), str.size(), ret.capacity());
    if(bytesWritten < 0)
        throw exception::Exception("lz4 decompress failed");
    ret.assign(ret.data(), bytesWritten);
    return ret;
}

} // namespace compress

} // namespace b0

#endif // LZ4_FOUND

