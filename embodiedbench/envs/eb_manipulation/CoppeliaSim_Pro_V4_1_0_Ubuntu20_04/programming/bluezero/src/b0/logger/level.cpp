#include <b0/logger/level.h>
#include <b0/exception/argument_error.h>

#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

namespace b0
{

namespace logger
{

std::string LevelInfo::ansiEscape() const
{
    boost::format fmt("\x1b[%d;%dm");
    return (fmt % attr % fg).str();
}

std::string LevelInfo::ansiReset() const
{
    return "\x1b[0m";
}

static const LevelInfo levelInfo_[] = {
    {"trace", Level::trace, 10, 1, 0x1e, 0},
    {"debug", Level::debug, 10, 1, 0x1e, 0},
    {"info",  Level::info,  20, 1, 0x25, 0},
    {"warn",  Level::warn,  30, 0, 0x21, 0},
    {"error", Level::error, 40, 0, 0x1f, 0},
    {"fatal", Level::fatal, 50, 7, 0x1f, 0},
};

const LevelInfo & levelInfo(const Level level)
{
    if     (level == Level::trace) return levelInfo_[0];
    else if(level == Level::debug) return levelInfo_[1];
    else if(level == Level::info)  return levelInfo_[2];
    else if(level == Level::warn)  return levelInfo_[3];
    else if(level == Level::error) return levelInfo_[4];
    else if(level == Level::fatal) return levelInfo_[5];
    else throw exception::ArgumentError("?", "log level");
}

const LevelInfo & levelInfo(const std::string &str)
{
    if     (str == "trace") return levelInfo_[0];
    else if(str == "debug") return levelInfo_[1];
    else if(str == "info")  return levelInfo_[2];
    else if(str == "warn")  return levelInfo_[3];
    else if(str == "error") return levelInfo_[4];
    else if(str == "fatal") return levelInfo_[5];
    else throw exception::ArgumentError(str, "log level");
}

} // namespace logger

} // namespace b0

