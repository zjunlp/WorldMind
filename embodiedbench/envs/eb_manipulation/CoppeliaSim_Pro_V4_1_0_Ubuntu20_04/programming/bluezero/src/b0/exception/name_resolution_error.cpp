#include <b0/exception/name_resolution_error.h>

#include <boost/format.hpp>

namespace b0
{

namespace exception
{

NameResolutionError::NameResolutionError(std::string name)
    : Exception((boost::format("Failed to resolve '%s'") % name).str())
{
}

} // namespace exception

} // namespace b0

