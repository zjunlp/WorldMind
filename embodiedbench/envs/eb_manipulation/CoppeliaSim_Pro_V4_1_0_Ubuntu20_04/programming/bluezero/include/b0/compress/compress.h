#ifndef B0__COMPRESS__COMPRESS_H__INCLUDED
#define B0__COMPRESS__COMPRESS_H__INCLUDED

#include <string>

#include <b0/b0.h>

namespace b0
{

namespace compress
{

std::string compress(const std::string &algorithm, const std::string &str, int level = -1);
std::string decompress(const std::string &algorithm, const std::string &str, size_t size = 0);

} // namespace compress

} // namespace b0

#endif // B0__COMPRESS__COMPRESS_H__INCLUDED
