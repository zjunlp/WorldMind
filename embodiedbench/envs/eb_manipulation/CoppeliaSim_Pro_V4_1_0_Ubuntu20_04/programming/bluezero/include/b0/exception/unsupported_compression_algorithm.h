#ifndef B0__EXCEPTION__UNSUPPORTED_COMPRESSION_ALGORITHM_H__INCLUDED
#define B0__EXCEPTION__UNSUPPORTED_COMPRESSION_ALGORITHM_H__INCLUDED

#include <b0/b0.h>
#include <b0/exception/exception.h>

namespace b0
{

namespace exception
{

/*!
 * \brief An exception thrown when an unsupported compression algorithm is specified or used
 */
class UnsupportedCompressionAlgorithm : public Exception
{
public:
    /*!
     * \brief Construct an UnsupportedCompressionAlgorithm exception
     */
    UnsupportedCompressionAlgorithm(std::string algorithm = "");
};

} // namespace exception

} // namespace b0

#endif // B0__EXCEPTION__UNSUPPORTED_COMPRESSION_ALGORITHM_H__INCLUDED
