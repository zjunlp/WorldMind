#include <b0/exception/argument_error.h>

#include <boost/format.hpp>

namespace b0
{

namespace exception
{

ArgumentError::ArgumentError(std::string argValue, std::string argName)
    : Exception((boost::format("Invalid %s: %s") % argName % argValue).str())
{
}

} // namespace exception

} // namespace b0

