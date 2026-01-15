#ifndef B0__EXCEPTION__ARGUMENT_ERROR_H__INCLUDED
#define B0__EXCEPTION__ARGUMENT_ERROR_H__INCLUDED

#include <b0/b0.h>
#include <b0/exception/exception.h>

namespace b0
{

namespace exception
{

/*!
 * \brief An exception thrown when an invalid argument or parameter is supplied
 */
class ArgumentError : public Exception
{
public:
    /*!
     * \brief Construct an ArgumentError exception
     */
    ArgumentError(std::string argValue, std::string argName = "argument");
};

} // namespace exception

} // namespace b0

#endif // B0__EXCEPTION__ARGUMENT_ERROR_H__INCLUDED
