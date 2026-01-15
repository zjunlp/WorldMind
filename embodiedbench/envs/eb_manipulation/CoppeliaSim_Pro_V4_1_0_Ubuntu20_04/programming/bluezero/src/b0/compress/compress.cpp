
#include <b0/exceptions.h>
#include <b0/compress/zlib.h>
#include <b0/compress/lz4.h>

namespace b0
{

namespace compress
{

std::string compress(const std::string &algorithm, const std::string &str, int level = -1)
{
    if(algorithm == "")
    {
        return str;
    }
#ifdef ZLIB_FOUND
    else if(algorithm == "zlib")
    {
        return zlib_compress(str, level);
    }
#endif
#ifdef LZ4_FOUND
    else if(algorithm == "lz4")
    {
        return lz4_compress(str, level);
    }
#endif
    else throw exception::UnsupportedCompressionAlgorithm(algorithm);
}

std::string decompress(const std::string &algorithm, const std::string &str, size_t size = 0)
{
    if(algorithm == "")
    {
        return str;
    }
#ifdef ZLIB_FOUND
    else if(algorithm == "zlib")
    {
        return zlib_decompress(str, size);
    }
#endif
#ifdef LZ4_FOUND
    else if(algorithm == "lz4")
    {
        return lz4_decompress(str, size);
    }
#endif
    else throw exception::UnsupportedCompressionAlgorithm(algorithm);
}

} // namespace compress

} // namespace b0

