#include <stdexcept>
#include <string>
#include <vector>
#include <map>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/regex.hpp>
#include <boost/regex.hpp>

std::vector<float> getLoadAverages()
{
    std::vector<std::string> lines = readSubprocessOutput("uptime");
    if(lines.size() == 0)
        throw std::runtime_error("Could not read output of 'uptime' program");
    const int n = 3;
    std::vector<float> avgs;
    boost::match_results<std::string::const_iterator> matches;
    boost::regex e(".*load average: ([^ ]+), ([^ ]+), ([^ ]+)");
    if(boost::regex_match(lines[0], matches, e, boost::match_default | boost::match_partial) && matches.size() == n + 1)
    {
        avgs.resize(n);
        for(int i = 0; i < n; i++)
            avgs[i] = boost::lexical_cast<float>(matches[i + 1]);
    }
    else throw std::runtime_error("Could not parse output of 'uptime' program");
    return avgs;
}

int getFreeMemory()
{
    std::vector<std::string> lines = readSubprocessOutput("free", {"-b"});
    if(lines.size() == 0)
        throw std::runtime_error("Could not read output of 'free' program");
    int r = -1, pgsz = 1;
    boost::match_results<std::string::const_iterator> matches;
    for(auto &line : lines)
    {
        boost::regex e("^Mem: +([^ ]+) +([^ ]+) +([^ ]+) +.*");
        if(boost::regex_match(line, matches, e) && matches.size() == 3)
        {
            return boost::lexical_cast<int>(matches[3]);
        }
    }
    return -1;
}
