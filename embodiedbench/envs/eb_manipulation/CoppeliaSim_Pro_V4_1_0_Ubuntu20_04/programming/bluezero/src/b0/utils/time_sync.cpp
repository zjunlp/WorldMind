#include <b0/utils/time_sync.h>
#include <b0/utils/env.h>

#include <boost/format.hpp>
#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

namespace b0
{

TimeSync::TimeSync()
{
    target_offset_ = 0;
    max_acceptable_offset_ = b0::env::getInt("B0_TIMESYNC_MAX_OFFSET", 5 * 1000 * 1000);
    last_offset_time_ = hardwareTimeUSec();
    last_offset_value_ = 0;
    setMaxSlope(b0::env::getDouble("B0_TIMESYNC_MAX_SLOPE", 0.05));
}

TimeSync::~TimeSync()
{
}

void TimeSync::setMaxSlope(double max_slope)
{
    if(max_slope <= 0)
        throw std::runtime_error("max_slope must be strictly positive");
    if(max_slope > 1)
        throw std::runtime_error("max_slope must not be greater than one");

    max_slope_ = max_slope;
}

int64_t TimeSync::hardwareTimeUSec() const
{
    static boost::posix_time::ptime epoch(boost::gregorian::date(1970, 1, 1));
    boost::posix_time::ptime t = boost::posix_time::microsec_clock::universal_time();
    return (t - epoch).total_microseconds();
}

int64_t TimeSync::timeUSec()
{
    return hardwareTimeUSec() + constantRateAdjustedOffset();
}

int64_t TimeSync::constantRateAdjustedOffset()
{
    boost::mutex::scoped_lock lock(mutex_);

    int64_t offset_delta = target_offset_ - last_offset_value_;
    int64_t slope_time = abs(offset_delta) / max_slope_;
    int64_t t = hardwareTimeUSec() - last_offset_time_;
    if(t >= slope_time)
        return target_offset_;
    else
        return last_offset_value_ + offset_delta * t / slope_time;
}

void TimeSync::updateTime(int64_t remoteTime)
{
    int64_t last_offset_value = constantRateAdjustedOffset();
    int64_t local_time = hardwareTimeUSec();

    {
        boost::mutex::scoped_lock lock(mutex_);

        last_offset_value_ = last_offset_value;
        last_offset_time_ = local_time;
        target_offset_ = remoteTime - local_time;

        if(max_acceptable_offset_ > 0 && abs(target_offset_) > max_acceptable_offset_)
            throw std::runtime_error((boost::format("Clock offset (%ld usec) is larger in absolute value than B0_TIMESYNC_MAX_OFFSET (%ld usec)") % target_offset_ % max_acceptable_offset_).str());
    }
}

} // namespace b0

