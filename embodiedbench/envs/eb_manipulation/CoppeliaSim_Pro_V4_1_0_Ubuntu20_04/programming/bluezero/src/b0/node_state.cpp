#include <b0/node_state.h>

#include <boost/format.hpp>

namespace b0
{

std::string NodeState_str(NodeState s)
{
#define STATE_TO_STRING(s) case NodeState::s: return #s;
    switch(s)
    {
    STATE_TO_STRING(Created)
    STATE_TO_STRING(Ready)
    STATE_TO_STRING(Terminated)
    }
#undef STATE_TO_STRING
    return (boost::format("NodeState#%d") % static_cast<int>(s)).str();
}

} // namespace b0

