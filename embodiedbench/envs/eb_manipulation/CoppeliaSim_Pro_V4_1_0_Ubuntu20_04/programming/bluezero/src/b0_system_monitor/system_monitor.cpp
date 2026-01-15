#include <memory>
#include <string>
#include <vector>
#include <map>
#include <b0/node.h>
#include <b0/publisher.h>
#include "protocol.h"
#ifdef HAVE_BOOST_PROCESS
#include <boost/process.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/regex.hpp>
#include <boost/regex.hpp>

namespace bp = boost::process;

std::vector<std::string> readSubprocessOutput(const std::string &exe, const std::vector<std::string> &args = {})
{
    std::vector<std::string> lines;
    bp::ipstream output;
    bp::child c(bp::search_path(exe), args, bp::std_out > output);
    c.wait();
    std::string line;
    while(std::getline(output, line)) lines.push_back(line);
    output.pipe().close();
    return lines;
}

/*
 * Platform-specific implementation of system monitor has to provide
 * the following functions:
 *
 *  - std::vector<float> getLoadAverages();
 *  - int getFreeMemory();
 */
#ifdef _WIN32
#include "impl/win32.cpp"
#elif __APPLE__
#include "impl/macos.cpp"
#elif __linux__
#include "impl/linux.cpp"
#elif __unix__
#include "impl/unix.cpp"
#elif defined(_POSIX_VERSION)
#include "impl/posix.cpp"
#endif

namespace b0
{

namespace system_monitor
{

class SystemMonitor : public b0::Node
{
public:
    SystemMonitor()
        : Node("system_monitor"),
          pub_(this, "system_monitor")
    {
    }

    ~SystemMonitor()
    {
    }

    void spinOnce()
    {
        Node::spinOnce();

        try
        {
            Load load_msg;
            load_msg.load_averages = getLoadAverages();
            load_msg.free_memory = getFreeMemory();
            pub_.publish(load_msg);
        }
        catch(std::exception &ex)
        {
            error(ex.what());
        }
    }

protected:
    b0::Publisher pub_;
};

} // namespace system_monitor

} // namespace b0

int main(int argc, char **argv)
{
    b0::init(argc, argv);
    b0::system_monitor::SystemMonitor node;
    node.init();
    node.spin();
    node.cleanup();
    return 0;
}

#else // HAVE_BOOST_PROCESS

#include <iostream>

int main()
{
    std::cerr << "boost/process.hpp is needed for system_monitor" << std::endl;
    return 1;
}

#endif // HAVE_BOOST_PROCESS
