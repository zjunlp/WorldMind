#include <atomic>
#include <iostream>
#include <boost/thread.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>

#include <b0/resolver/resolver.h>
#include <b0/node.h>
#include <b0/publisher.h>
#include <b0/subscriber.h>

namespace po = boost::program_options;

void resolver_thread()
{
    b0::resolver::Resolver node;
    node.init();
    node.spin();
}

int expect_failure;
int enable_conflate;

std::atomic<bool> stop{false};

std::atomic<long> pub_max{0};

void pub_thread()
{
    b0::Node node("pub");
    b0::Publisher pub(&node, "topic1");
    //pub.setCompression("zlib", 9);
    node.init();
    long i = 0;
    for(;;)
    {
        if(stop) break;
        std::string m = boost::lexical_cast<std::string>(i);
        std::cout << "send: " << m << std::endl;
        pub.publish(m);
        pub_max = i;
        i++;
        boost::this_thread::sleep_for(boost::chrono::milliseconds{20});
    }
    node.cleanup();
}

std::atomic<long> sub_num_recv{0};
std::atomic<long> sub_sum{0};
std::atomic<long> exp_sum{0};

void sub_thread()
{
    b0::Node node("sub");
    b0::Subscriber sub(&node, "topic1");
    sub.setConflate(enable_conflate);
    node.init();
    for(;;)
    {
        if(stop) break;
        std::string m;
        sub.readRaw(m);
        std::cout << "                recv: " << m << std::endl;
        long i = boost::lexical_cast<long>(m);
        sub_num_recv++;
        sub_sum += i;
        exp_sum += pub_max;
        boost::this_thread::sleep_for(boost::chrono::milliseconds{250});
    }
    node.cleanup();
}

void timeout_thread()
{
    boost::this_thread::sleep_for(boost::chrono::seconds{4});
    stop = true;
    boost::this_thread::sleep_for(boost::chrono::seconds{1});
    bool passed = false;
    static const char *bool2str[] = {"NO", "YES"};
    std::cout << "Finish:" << std::endl;
    std::cout << "    use_conflate = " << bool2str[enable_conflate] << std::endl;
    std::cout << "    pub_max      = " << pub_max << std::endl;
    std::cout << "    sub_num_recv = " << sub_num_recv << std::endl;
    if(sub_num_recv)
    {
        std::cout << "    sub_sum      = " << sub_sum << std::endl;
        std::cout << "    exp_sum      = " << exp_sum << std::endl;
        double error = fabs(exp_sum - sub_sum) / (exp_sum + 0.001);
        passed = expect_failure ^ (error < 0.1);
        std::cout << "    error        = " << int(error * 100) << "%" << std::endl;
    }
    std::cout << "    passed       = " << bool2str[passed] << std::endl;
    /*
     * Without conflate, messages will queue up regularly.
     * Numbers that get accumulated in sub_sum are 0, 1, 2, 3, 4, 5, ...
     *
     * With conflate, new messages will replace queued messages.
     * Numbers that get accumulated in sub_sum are 0, 10, 20, ...
     * (more or less, YMMV)
     * At the same time of receiving a message, we store into exp_sum
     * the expected sum of received numbers, by directly reading the
     * variable pub_max (value of last number published).
     *
     * This should give us a rough estimate of what sub_sum should be
     * in order to say that the test is passed.
     *
     * So if sub_sum is approximately equal (up to 10% of error) to
     * exp_sum, the test is passed.
     */
    exit(passed ? 0 : 1);
}

int main(int argc, char **argv)
{
    b0::addOptionInt("enable-conflate,c", "enable conflate", &enable_conflate, true);
    b0::setPositionalOption("enable-conflate");
    b0::init(argc, argv);

    expect_failure = !enable_conflate;

    boost::thread t0(&timeout_thread);
    boost::thread t1(&resolver_thread);
    boost::this_thread::sleep_for(boost::chrono::seconds{1});
    boost::thread t2(&sub_thread);
    boost::this_thread::sleep_for(boost::chrono::seconds{1});
    boost::thread t3(&pub_thread);
    t0.join();
}

