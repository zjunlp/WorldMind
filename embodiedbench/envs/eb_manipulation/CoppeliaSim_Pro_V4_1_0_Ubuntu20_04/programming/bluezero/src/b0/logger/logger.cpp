#include <b0/logger/logger.h>
#include <b0/publisher.h>
#include <b0/message/log/log_entry.h>
#include <b0/node.h>
#include <b0/exception/argument_error.h>
#include <b0/utils/thread_name.h>
#include <b0/utils/env.h>

#include <iostream>
#include <iomanip>
#include <chrono>

#include <boost/lexical_cast.hpp>

namespace b0
{

namespace logger
{

void LogInterface::log_helper(Level level, boost::format &format) const
{
    return log(level, format.str());
}

LocalLogger::LocalLogger(b0::Node *node)
    : node_(node),
      color_(false)
{
    outputLevel_ = getConsoleLogLevel();

    std::string term = b0::env::get("TERM");
    if(term == "xterm-color" || term == "xterm-256color") color_ = true;
}

LocalLogger::~LocalLogger()
{
}

void LocalLogger::log(Level level, const std::string &message) const
{
    if(level < outputLevel_) return;

    LevelInfo info = levelInfo(level);
    std::stringstream ss;
    if(color_) ss << info.ansiEscape();

    auto now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S ");

    if(node_)
    {
        std::string name = node_->getName();
        if(!name.empty()) ss << "[" << name << "] ";
    }

    ss << info.str << ": ";

    ss << message;

    if(color_) ss << info.ansiReset();

    std::cout << ss.str() << std::endl;
}

struct Logger::Private
{
    Private(Node *node)
        : pub_(node, "log", false, false)
    {
    }

    Publisher pub_;
};

Logger::Logger(b0::Node *node)
    : LocalLogger(node),
      private_(new Private(node))
{
}

Logger::~Logger()
{
}

void Logger::connect(const std::string &addr)
{
    private_->pub_.setRemoteAddress(addr);
    private_->pub_.init();
}

void Logger::log(Level level, const std::string &message) const
{
    LocalLogger::log(level, message);

    remoteLog(level, message);
}

void Logger::remoteLog(Level level, const std::string &message) const
{
    b0::message::log::LogEntry e;

    if(node_)
    {
        e.node_name = node_->getName();
        e.time_usec = node_->timeUSec();
    }

    e.level = levelInfo(level).str;
    e.message = message;

    private_->pub_.publish(e);
}

} // namespace logger

} // namespace b0

