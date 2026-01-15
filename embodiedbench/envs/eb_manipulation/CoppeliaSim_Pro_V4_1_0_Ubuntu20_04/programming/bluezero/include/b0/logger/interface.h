#ifndef B0__LOGGER__INTERFACE_H__INCLUDED
#define B0__LOGGER__INTERFACE_H__INCLUDED

#include <string>

#include <boost/format.hpp>

#include <b0/b0.h>
#include <b0/logger/level.h>

namespace b0
{

namespace logger
{

/*!
 * \brief Base class to add logging functionalities to nodes
 */
class LogInterface
{
public:
    /*!
     * \brief Log a message to the remote logger, with a specified level
     */
    virtual void log(Level level, const std::string &message) const = 0;

    /*!
     * \brief Log a message using a format string
     */
    template<typename... Arguments>
    void log(Level level, std::string const &fmt, Arguments&&... args) const
    {
        try
        {
            boost::format format(fmt);
            log_helper(level, format, std::forward<Arguments>(args)...);
        }
        catch(boost::io::too_many_args &ex)
        {
            std::string s = fmt;
            s += " (error during formatting)";
            log(level, s);
        }
    }

    /*!
     * \brief Log a message at trace level using a format string
     */
    template<typename... Arguments>
    void trace(std::string const &fmt, Arguments&&... args) const
    {
        log(Level::trace, fmt, std::forward<Arguments>(args)...);
    }

    /*!
     * \brief Log a message at debug level using a format string
     */
    template<typename... Arguments>
    void debug(std::string const &fmt, Arguments&&... args) const
    {
        log(Level::debug, fmt, std::forward<Arguments>(args)...);
    }

    /*!
     * \brief Log a message at info level using a format string
     */
    template<typename... Arguments>
    void info(std::string const &fmt, Arguments&&... args) const
    {
        log(Level::info, fmt, std::forward<Arguments>(args)...);
    }

    /*!
     * \brief Log a message at warn level using a format string
     */
    template<typename... Arguments>
    void warn(std::string const &fmt, Arguments&&... args) const
    {
        log(Level::warn, fmt, std::forward<Arguments>(args)...);
    }

    /*!
     * \brief Log a message at error level using a format string
     */
    template<typename... Arguments>
    void error(std::string const &fmt, Arguments&&... args) const
    {
        log(Level::error, fmt, std::forward<Arguments>(args)...);
    }

    /*!
     * \brief Log a message at fatal level using a format string
     */
    template<typename... Arguments>
    void fatal(std::string const &fmt, Arguments&&... args) const
    {
        log(Level::fatal, fmt, std::forward<Arguments>(args)...);
    }

protected:
    //! \cond HIDDEN_SYMBOLS

    virtual void log_helper(Level level, boost::format &format) const;

    template<class T, class... Args>
    void log_helper(Level level, boost::format &format, T &&t, Args&&... args) const
    {
        return log_helper(level, format % std::forward<T>(t), std::forward<Args>(args)...);
    }

    //! \endcond
};

} // namespace logger

} // namespace b0

#endif // B0__LOGGER__INTERFACE_H__INCLUDED
