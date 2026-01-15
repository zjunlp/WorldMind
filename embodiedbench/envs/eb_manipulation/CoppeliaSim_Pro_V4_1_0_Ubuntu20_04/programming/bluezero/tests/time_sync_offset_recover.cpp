#include <b0/utils/time_sync.h>

#include <boost/thread.hpp>
#include <boost/lexical_cast.hpp>

#include <iostream>
#include <string>

// unit-test for the time sync algorithm
// (does not cover time synchronization between nodes)

namespace b0
{

// adds an offset to hardware clock

class TimeSync_TEST : public TimeSync
{
public:
    TimeSync_TEST(int64_t offset, double speed) : t0_(TimeSync::hardwareTimeUSec()), offset_(offset), speed_(speed) {}
    int64_t hardwareTimeUSec() const {return offset_ + speed_ * (TimeSync::hardwareTimeUSec() - t0_) + t0_;}
protected:
    int64_t t0_;
    int64_t offset_;
    double speed_;
};

}

inline int64_t sec(double s) { return s * 1000 * 1000; }

int main(int argc, char **argv)
{
    b0::init(argc, argv);

    double max_slope = 0.5;
    int64_t offset = sec(5);
    double speed = 1.0;
    b0::TimeSync c;
    c.setMaxSlope(max_slope);
    b0::TimeSync_TEST s(offset, speed);
    s.setMaxSlope(max_slope);

    int64_t test_end_time = offset * 1.2 / max_slope + c.hardwareTimeUSec(); // run for the required time to adjust clock + 20%
    int64_t error = 0;

    while(c.hardwareTimeUSec() < test_end_time)
    {
        int64_t server_time = s.hardwareTimeUSec();
        c.updateTime(server_time);
        error = c.timeUSec() - server_time;
        std::cout << "server_time=" << server_time << ", error=" << error << std::endl;
        boost::this_thread::sleep_for(boost::chrono::seconds{1});
    }

    return !(std::abs(error) < sec(0.05));
}

