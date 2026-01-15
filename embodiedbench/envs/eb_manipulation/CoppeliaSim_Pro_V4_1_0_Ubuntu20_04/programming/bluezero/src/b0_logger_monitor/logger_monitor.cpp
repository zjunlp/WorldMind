#include <iostream>
#include <string>

#include <b0/node.h>
#include <b0/subscriber.h>
#include <b0/logger/logger.h>
#include <b0/message/log/log_entry.h>

namespace b0
{

namespace logger
{

class Console : public Node
{
public:
    Console()
        : Node("logger_monitor"),
          sub_(this, "log", &Console::onLogMessage, this)
    {
    }

    ~Console()
    {
    }

    void onLogMessage(const b0::message::log::LogEntry &entry)
    {
        LevelInfo info = levelInfo(entry.level);
        std::cout << info.ansiEscape() << "[" << entry.node_name << "] " << info.str << ": " << entry.message << info.ansiReset() << std::endl;
    }

protected:
    b0::Subscriber sub_;
};

} // namespace logger

} // namespace b0

int main(int argc, char **argv)
{
    b0::init(argc, argv);
    b0::logger::Console console;
    console.init();
    console.spin();
    console.cleanup();
    return 0;
}

