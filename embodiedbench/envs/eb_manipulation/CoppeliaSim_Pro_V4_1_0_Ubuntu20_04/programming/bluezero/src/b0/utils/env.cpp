#include <b0/utils/env.h>
#include <b0/exception/argument_error.h>

#include <boost/algorithm/string.hpp>

namespace b0
{

namespace env
{

std::string get(const std::string &var, const std::string &def)
{
    std::string ret = def;
    const char *v = std::getenv(var.c_str());
    if(v) ret = v;
    return ret;
}

bool getBool(const std::string &var, bool def)
{
    bool ret = def;
    std::string val = get(var), vlc = boost::algorithm::to_lower_copy(val);
    if(vlc == "");
    else if(vlc == "true"  || vlc == "yes" || vlc == "on"  || vlc == "1") ret = true;
    else if(vlc == "false" || vlc == "no"  || vlc == "off" || vlc == "0") ret = false;
    else throw exception::ArgumentError(val, var);
    return ret;
}

int getInt(const std::string &var, int def)
{
    int ret = def;
    std::string val = get(var);
    if(val != "") ret = std::atoi(val.c_str());
    return ret;
}

double getDouble(const std::string &var, double def)
{
    double ret = def;
    std::string val = get(var);
    if(val != "") ret = std::atof(val.c_str());
    return ret;
}

} // namespace env

} // namespace b0
