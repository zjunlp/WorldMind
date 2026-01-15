#ifndef B0__EXCEPTION__EXCEPTION_H__INCLUDED
#define B0__EXCEPTION__EXCEPTION_H__INCLUDED

#include <b0/b0.h>

#include <string>
#include <stdexcept>

namespace b0
{

namespace exception
{

/*!
 * \brief The base exception class
 */
class Exception : public std::exception
{
public:
    /*!
     * \brief Construct an Exception
     */
    Exception(std::string message = "");

    /*!
     * \brief Return error message
     */
    virtual const char * what() const noexcept;

protected:
    //! The exception message
    std::string message_;
};

} // namespace exception

} // namespace b0

#endif // B0__EXCEPTION__EXCEPTION_H__INCLUDED
