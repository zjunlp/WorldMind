#include <b0/exception/invalid_state_transition.h>

#include <boost/format.hpp>

namespace b0
{

namespace exception
{

InvalidStateTransition::InvalidStateTransition(std::string function, NodeState state)
    : Exception((boost::format("Cannot call %s() in current state (%s)") % function % NodeState_str(state)).str()),
      function_(function),
      state_(state)
{
}

std::string InvalidStateTransition::getFunction() const
{
    return function_;
}

NodeState InvalidStateTransition::getState() const
{
    return state_;
}

} // namespace exception

} // namespace b0

