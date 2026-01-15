#ifndef B0__EXCEPTION__NAME_RESOLUTION_ERROR_H__INCLUDED
#define B0__EXCEPTION__NAME_RESOLUTION_ERROR_H__INCLUDED

#include <b0/b0.h>
#include <b0/exception/exception.h>

namespace b0
{

namespace exception
{

/*!
 * \brief An exception thrown when a socket name (service) fails to resolve (on the client side)
 */
class NameResolutionError : public Exception
{
public:
    /*!
     * \brief Construct a NameResolutionError exception
     */
    NameResolutionError(std::string name = "");
};

} // namespace exception

} // namespace b0

#endif // B0__EXCEPTION__NAME_RESOLUTION_ERROR_H__INCLUDED
