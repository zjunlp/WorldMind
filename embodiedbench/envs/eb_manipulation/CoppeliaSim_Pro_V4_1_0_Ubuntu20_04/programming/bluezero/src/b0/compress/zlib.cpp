#include <stdexcept>
#include <cstring>

#include <boost/format.hpp>

#include <b0/exceptions.h>
#include <b0/compress/zlib.h>

#ifdef ZLIB_FOUND

#include <zlib.h>

namespace b0
{

namespace compress
{

std::string zlib_wrapper(const std::string &str, bool compress, int level, size_t size)
{
    if(level == -1) level = Z_BEST_COMPRESSION;
    const char *method = compress ? "deflate" : "inflate";
    int ret;
    z_stream zs;
    memset(&zs, 0, sizeof(zs));
    if(compress) ret = deflateInit(&zs, level); else ret = inflateInit(&zs);
    if(ret != Z_OK)
        throw exception::Exception((boost::format("%sInit failed") % method).str());;
    zs.next_in = (Bytef*)(str.data());
    zs.avail_in = str.size();
    if(size == 0) size = str.size() * (compress ? 2 : 10);
    char *outbuf = new char[size];
    std::string outstr;
    do
    {
        zs.next_out = reinterpret_cast<Bytef*>(outbuf);
        zs.avail_out = sizeof(outbuf);
        ret = compress ? deflate(&zs, Z_FINISH) : inflate(&zs, 0);
        if(outstr.size() < zs.total_out)
            outstr.append(outbuf, zs.total_out - outstr.size());
    }
    while(ret == Z_OK);
    if(compress) deflateEnd(&zs); else inflateEnd(&zs);
    delete[] outbuf;
    if(ret != Z_STREAM_END)
        throw exception::Exception((boost::format("zlib %s error %d%s%s") % method % ret % (zs.msg ? ": " : "") % (zs.msg ? zs.msg : "")).str());
    return outstr;
}

std::string zlib_compress(const std::string &str, int level)
{
    return zlib_wrapper(str, true, level, 0);
}

std::string zlib_decompress(const std::string &str, size_t size)
{
    return zlib_wrapper(str, false, 0, size);
}

} // namespace compress

} // namespace b0

#endif // ZLIB_FOUND

