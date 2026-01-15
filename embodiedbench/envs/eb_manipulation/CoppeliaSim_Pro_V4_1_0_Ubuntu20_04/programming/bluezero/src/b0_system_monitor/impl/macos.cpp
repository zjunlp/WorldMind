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
        throw std::runtime_error("Could not read output of 'vm_stat' program");
    const int n = 3;
    std::vector<float> avgs;
    boost::match_results<std::string::const_iterator> matches;
    boost::regex e(".*load averages: ([^ ]+) ([^ ]+) ([^ ]+)");
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
    std::vector<std::string> lines = readSubprocessOutput("vm_stat");
    if(lines.size() == 0)
        throw std::runtime_error("Could not read output of 'vm_stat' program");
    int r = -1, pgsz = 1;
    boost::match_results<std::string::const_iterator> matches;
    boost::regex e("Mach Virtual Memory Statistics: \\(page size of ([0-9]+) bytes\\)");
    if(!boost::regex_match(lines[0], matches, e, boost::match_default | boost::match_partial) || matches.size() != 2)
        throw std::runtime_error("Could not parse output of 'vm_stat' program");
    pgsz = boost::lexical_cast<int>(matches[1]);
    std::map<std::string, int> stats;
    for(int i = 1; i < lines.size(); i++)
    {
        std::vector<std::string> pair;
        boost::algorithm::split_regex(pair, lines[i], boost::regex(": *"));
        stats[pair[0]] = int(boost::lexical_cast<float>(pair[1]));
    }
    auto it = stats.find("Pages free");
    if(it != stats.end())
        r = it->second * pgsz;
    return r;
}
