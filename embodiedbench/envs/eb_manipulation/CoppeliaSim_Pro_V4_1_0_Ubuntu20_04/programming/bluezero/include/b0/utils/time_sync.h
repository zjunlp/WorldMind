#ifndef B0__UTILS__TIMESYNC_H__INCLUDED
#define B0__UTILS__TIMESYNC_H__INCLUDED

#include <cstdint>

#include <boost/thread/mutex.hpp>

#include <b0/b0.h>

/*!
 * \page timesync Time Synchronization
 *
 * This page describes how time synchronization works.
 *
 * The objective of time synchronization is to coordinate otherwise independent clocks. Even when initially set accurately, real clocks will differ after some amount of time due to clock drift, caused by clocks counting time at slightly different rates.
 *
 * \image html timesync_plot1.png "Example of two drifting clocks" width=500pt
 *
 * There is one master clock node, which usualy coincides with the resolver node,
 * and every other node instance will try to synchronize its clock to the master clock,
 * while maintaining a guarrantee on some properties:
 *
 *  - time must not do arbitrarily big jumps
 *  - time must always increase monotonically, i.e. if we read time into variable T1, and after some time we read again time into variable T2, it must always be that T2 >= T1
 *  - locally, adjusted time must change at a constant speed, that is, the adjustment must happen at a constant rate
 *  - the adjustment must be a continuous function of time, such that even if the time is adjusted at a low rate (typically 1Hz) we get a consistent behavior for sub-second reads
 *
 *  Time synchronization never changes the computer's hardware clock.
 *  It rather computes an offset to add to the hardware clock.
 *
 *  The method Node::timeUSec() returns the value of the hardware clock corrected by the required offset, while the method Node::hardwareTimeUSec() will return the hardware clock actual value.
 *
 *  Each time a new time is received from master clock (tipically in the heartbeat message) the method Node::updateTime() is called, and a new offset is computed.
 *
 * \image html timesync_plot2.png "Example time series of the offset, which is computed as the difference between local time and remote time. Note that is not required that the offset is received at fixed intervals, and in fact in this example it is not the case." width=500pt
 *
 *  If we look at the offset as a function of time we see that is discontinuous.
 *  This is bad because just adding the offset to the hardware clock would cause
 *  arbitrarily big jumps and even jump backwards in time, thus violating the
 *  two properties stated before.
 *
 * \image html timesync_plot3.png "The adjusted time obtained by adding the offset to local time" width=500pt
 *
 *  To fix this, the offset function is smoothed so that it is continuous, and
 *  limited in its rate of change (max slope).
 *  It is important that the max slope is always greater than zero, so as to produce an actual change, and strictly less than 1, so as to not cause time to stop or go backwards.
 *
 * \image html timesync_plot4.png "The smoothed offset. In this example we used a max slope of 0.5, such that the time adjustment is at most half second per second." width=500pt
 *
 * \image html timesync_plot5.png "The resulting adjusted time" width=500pt
 *
 */

namespace b0
{

/*!
 * \brief The TimeSync class
 *
 * Handles the time synchronization aspects of the Node class.
 */
class TimeSync
{
public:
    /*!
     * \brief TimeSync constructor
     */
    TimeSync();

    /*!
     * \brief TimeSync destructor
     */
    virtual ~TimeSync();

    /*!
     * Set the maximum time synchronization slope
     * \param max_slope Indicates the maximum correction speed (in adjusted seconds per real second). Always keep this strictly greater than 0 and strictly less than 1.
     */
    void setMaxSlope(double max_slope);

    /*!
     * \brief Return this computer's clock time in microseconds
     *
     * This method is thread-safe.
     */
    virtual int64_t hardwareTimeUSec() const;

    /*!
     * \brief Return the adjusted time in microseconds. See \ref timesync for details.
     *
     * This method is thread-safe.
     */
    virtual int64_t timeUSec();

    /*!
     * Compute a smoothed offset with a linear velocity profile
     * with a slope never greater (in absolute value) than max_slope
     *
     * This method is thread-safe.
     */
    virtual int64_t constantRateAdjustedOffset();

    /*!
     * Update the time offset with a time from remote server (in microseconds)
     *
     * This method is thread-safe.
     */
    virtual void updateTime(int64_t remoteTime);

private:
    /*
     * State variables related to time synchronization
     */
    int64_t target_offset_;
    int64_t max_acceptable_offset_;
    int64_t last_offset_time_;
    int64_t last_offset_value_;
    double max_slope_;
    boost::mutex mutex_;
};

} // namespace b0

#endif // B0__UTILS__TIMESYNC_H__INCLUDED
