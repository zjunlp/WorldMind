#ifndef B0__LOGGER__LOGGER_H__INCLUDED
#define B0__LOGGER__LOGGER_H__INCLUDED

#include <b0/b0.h>
#include <b0/logger/interface.h>

#include <string>
#include <sstream>
#include <boost/thread.hpp>
#include <boost/format.hpp>

namespace b0
{

class Node;

namespace logger
{

/*!
 * \brief A logger which prints messages to local console.
 */
class LocalLogger : public LogInterface
{
public:
    using LogInterface::log;

    //! Constructor
    LocalLogger(b0::Node *node = nullptr);

    //! Destructor
    virtual ~LocalLogger();

    /*!
     * Log a message to the local console logger (i.e. using std::cout)
     */
    virtual void log(Level level, const std::string &message) const override;

protected:
    //! The node
    b0::Node *node_;

    //! The output log level.
    Level outputLevel_;

    //! Flag to indicate if terminal supports ANSI escapes sequences for text color
    bool color_;

    friend class Console;
};

/*!
 * \brief A subclass of LocalLogger which also sends log messages remotely, via a ZeroMQ PUB socket
 */
class Logger : public LocalLogger
{
private:
    //! \cond HIDDEN_SYMBOLS

    struct Private;

    //! \endcond

public:
    using LocalLogger::log;

    /*!
     * Construct a Logger for the given named object
     */
    Logger(b0::Node *node);

    /*!
     * Logger destructor
     */
    virtual ~Logger();

    /*!
     * Connect the underlying ZeroMQ PUB socket to the given address
     */
    void connect(const std::string &addr);

    void log(Level level, const std::string &message) const override;

protected:
    /*!
     * Log a message to the remote logger (i.e. using the log publisher)
     */
    virtual void remoteLog(Level level, const std::string &message) const;

private:
    mutable std::unique_ptr<Private> private_;
};

} // namespace logger

} // namespace b0

#endif // B0__LOGGER__LOGGER_H__INCLUDED
