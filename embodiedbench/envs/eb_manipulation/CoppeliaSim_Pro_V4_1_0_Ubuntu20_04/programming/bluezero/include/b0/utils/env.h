#ifndef B0__UTILS__ENV_H__INCLUDED
#define B0__UTILS__ENV_H__INCLUDED

#include <b0/b0.h>

#include <string>

//! \file

namespace b0
{

namespace env
{

std::string get(const std::string &var, const std::string &def = "");

bool getBool(const std::string &var, bool def = false);

int getInt(const std::string &var, int def = 0);

double getDouble(const std::string &var, double def = 0);

} // namespace env

} // namespace b0

#endif // B0__UTILS__ENV_H__INCLUDED
