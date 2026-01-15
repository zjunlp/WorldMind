#ifndef B0__EXCEPTION__INVALID_STATE_TRANSITION_H__INCLUDED
#define B0__EXCEPTION__INVALID_STATE_TRANSITION_H__INCLUDED

#include <b0/b0.h>
#include <b0/exception/exception.h>
#include <b0/node_state.h>

namespace b0
{

namespace exception
{

/*!
 * \brief An exception thrown when an invalid method for the current object state is called
 */
class InvalidStateTransition : public Exception
{
public:
    /*!
     * \brief Construct an InvalidStateTransition exception
     */
    InvalidStateTransition(std::string function, NodeState state);

    /*!
     * \brief Get the function that has been called
     */
    std::string getFunction() const;

    /*!
     * \brief Get the state in which the functiomn has been called
     */
    NodeState getState() const;

private:
    std::string function_;
    NodeState state_;
};

} // namespace exception

} // namespace b0

#endif // B0__EXCEPTION__INVALID_STATE_TRANSITION_H__INCLUDED
