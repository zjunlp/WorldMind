#ifndef B0__COMPRESS__LZ4_H__INCLUDED
#define B0__COMPRESS__LZ4_H__INCLUDED

#include <string>

#include <b0/b0.h>

namespace b0
{

namespace compress
{

#ifdef LZ4_FOUND

std::string lz4_compress(const std::string &str, int level = -1);
std::string lz4_decompress(const std::string &str, size_t size = 0);

#endif

} // namespace compress

} // namespace b0

#endif // B0__COMPRESS__LZ4_H__INCLUDED
