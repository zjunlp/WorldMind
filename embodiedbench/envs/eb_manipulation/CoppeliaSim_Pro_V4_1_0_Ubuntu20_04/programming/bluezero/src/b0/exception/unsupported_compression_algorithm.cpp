#include <b0/exception/unsupported_compression_algorithm.h>

#include <boost/format.hpp>

namespace b0
{

namespace exception
{

UnsupportedCompressionAlgorithm::UnsupportedCompressionAlgorithm(std::string algorithm)
    : Exception((boost::format("Unsupported compression algorithm: %s") % algorithm).str())
{
}

} // namespace exception

} // namespace b0

