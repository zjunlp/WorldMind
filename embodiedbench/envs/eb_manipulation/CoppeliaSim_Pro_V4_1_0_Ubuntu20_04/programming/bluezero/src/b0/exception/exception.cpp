#include <b0/exception/exception.h>

namespace b0
{

namespace exception
{

Exception::Exception(std::string message)
    : message_(message)
{
}

const char * Exception::what() const noexcept
{
    return message_.c_str();
}

} // namespace exception

} // namespace b0

