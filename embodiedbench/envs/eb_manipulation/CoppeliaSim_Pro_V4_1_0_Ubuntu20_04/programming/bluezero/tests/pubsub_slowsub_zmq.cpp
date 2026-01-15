#include <atomic>
#include <iostream>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/thread.hpp>
#include <zmq.hpp>

//#define USE_PROXY

#ifdef USE_PROXY
boost::format xpub_proxy_addr("tcp://%s:38921");
boost::format xsub_proxy_addr("tcp://%s:38922");
#else
boost::format pub_addr("tcp://%s:38922");
#endif

bool expect_failure;
bool enable_conflate;

std::atomic<bool> stop{false};

std::atomic<long> pub_max{0};

void pub_thread(zmq::context_t &context)
{
    zmq::socket_t pub(context, zmq::socket_type::pub);
#ifdef USE_PROXY
    pub.connect((xsub_proxy_addr % "localhost").str());
#else
    pub.bind((pub_addr % "*").str());
#endif
    long i = 0;
    for(;;)
    {
        if(stop) break;
        std::string m = boost::lexical_cast<std::string>(i);
        zmq::message_t msg(6 + m.size());
        char *data = static_cast<char *>(msg.data());
        memcpy(data, "topic1", 6);
        memcpy(data + 6, m.data(), m.size());
        std::cout << "send: " << m << std::endl;
        if(!pub.send(msg))
            std::cout << "send error" << std::endl;
        pub_max = i;
        i++;
        if(i>80)break;
        boost::this_thread::sleep_for(boost::chrono::milliseconds{20});
    }
}

std::atomic<long> sub_num_recv{0};
std::atomic<long> sub_sum{0};
std::atomic<long> exp_sum{0};

void sub_thread(zmq::context_t &context)
{
    zmq::socket_t sub(context, zmq::socket_type::sub);
    if(enable_conflate)
    {
        const int v_true = 1;
        sub.setsockopt(ZMQ_CONFLATE, &v_true, sizeof(v_true));
    }
#ifdef USE_PROXY
    sub.connect((xpub_proxy_addr % "localhost").str());
#else
    sub.connect((pub_addr % "localhost").str());
#endif
    sub.setsockopt(ZMQ_SUBSCRIBE, "topic1", 6);
    for(;;)
    {
        if(stop) break;
        zmq::message_t msg;
        if(!sub.recv(&msg))
            std::cout << "recv error" << std::endl;
        std::string m(reinterpret_cast<const char*>(msg.data()) + 6, msg.size() - 6);
        std::cout << "                recv: " << m << std::endl;
        long i = boost::lexical_cast<long>(m);
        sub_num_recv++;
        sub_sum += i;
        exp_sum += pub_max;
        boost::this_thread::sleep_for(boost::chrono::milliseconds{250});
    }
}

void proxy_thread(zmq::context_t &context)
{
#ifdef USE_PROXY
    zmq::socket_t xpub(context, zmq::socket_type::xpub);
    xpub.bind((xpub_proxy_addr % "*").str());
    zmq::socket_t xsub(context, zmq::socket_type::xsub);
    xsub.bind((xsub_proxy_addr % "*").str());
    std::cout << "starting xpub/xsub proxy" << std::endl;
    zmq::proxy(xpub, xsub, nullptr);
    std::cout << "xpub/xsub proxy terminated" << std::endl;
#endif
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
    if(argc != 2)
    {
        std::cerr << "usage: " << argv[0] << " <0 or 1>" << std::endl;
        exit(2);
    }

    enable_conflate = argv[1][0] == '1';
    expect_failure = !enable_conflate;

    const int io_threads = 1;
    zmq::context_t context(io_threads);

    boost::thread t0(&timeout_thread);
    boost::thread t1(&proxy_thread, boost::ref(context));
    boost::this_thread::sleep_for(boost::chrono::seconds{1});
    boost::thread t2(&sub_thread, boost::ref(context));
    boost::this_thread::sleep_for(boost::chrono::seconds{1});
    boost::thread t3(&pub_thread, boost::ref(context));
    t0.join();
}

