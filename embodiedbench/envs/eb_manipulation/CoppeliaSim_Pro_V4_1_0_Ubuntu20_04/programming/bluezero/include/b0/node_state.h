#ifndef B0__NODE_STATE_H__INCLUDED
#define B0__NODE_STATE_H__INCLUDED

#include <string>

namespace b0
{

/*!
 * \brief The state of a Node
 */
enum NodeState
{
    //! \brief Just after creation, before initialization.
    Created,
    //! \brief After initialization.
    Ready,
    //! \brief Just after cleanup.
    Terminated
};

/*!
 * \brief Convert a state to string
 */
std::string NodeState_str(NodeState s);

} // namespace b0

#endif // B0__NODE_STATE_H__INCLUDED
