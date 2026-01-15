#include <iostream>
#include <boost/thread.hpp>

#include <b0/resolver/resolver.h>
#include <b0/node.h>
#include <b0/publisher.h>
#include <b0/subscriber.h>

void resolver_thread()
{
    b0::resolver::Resolver node;
    node.init();
    node.spin();
}

const double spinRate = 2.5;
const int count = 4;

void pub_thread()
{
    b0::Node node("pub");
    node.setSpinRate(spinRate);
    b0::Publisher pub(&node, "topic1");
    node.init();
    int i = 0;
    node.spin([&](){
        int64_t t = node.hardwareTimeUSec();
        std::cerr << t << ": send msg" << std::endl;
        std::string msg;
        pub.publish(msg);
        node.sleepUSec(1000 * 1000 / 2 / spinRate);
        if(++i > count) node.shutdown();
    });
    node.cleanup();
}

void sub_thread()
{
    std::vector<int64_t> recv_time;
    b0::Node node("sub");
    b0::Subscriber sub(&node, "topic1");
    node.init();
    int i = 0;
    for(;;)
    {
        std::string m;
        sub.readRaw(m);
        int64_t t = node.hardwareTimeUSec();
        std::cerr << t << ": recv msg" << std::endl;
        recv_time.push_back(t);
        if(++i > count) break;
    }
    node.cleanup();
    double avg_freq = 0;
    for(int i = 1; i < recv_time.size(); i++)
    {
        avg_freq += (1000000. / (recv_time[i] - recv_time[i - 1]));
    }
    avg_freq /= recv_time.size() - 1;
    if(fabs(avg_freq - spinRate) / spinRate  >  0.05)
    {
        std::cerr << "failure: avg_freq =  " << avg_freq << " (should be " << spinRate << ")" << std::endl;
        exit(1);
    }
    else
    {
        std::cerr << "success" << std::endl;
        exit(0);
    }
}

void timeout_thread()
{
    boost::this_thread::sleep_for(boost::chrono::seconds{3} +
            boost::chrono::microseconds{2 * long(count * 1000000 / spinRate)});
    std::cerr << "failure: timeout" << std::endl;
    exit(1);
}

int main(int argc, char **argv)
{
    b0::init(argc, argv);

    boost::thread t0(&timeout_thread);
    boost::thread t1(&resolver_thread);
    boost::this_thread::sleep_for(boost::chrono::seconds{1});
    boost::thread t2(&sub_thread);
    boost::this_thread::sleep_for(boost::chrono::seconds{1});
    boost::thread t3(&pub_thread);
    t0.join();
}

