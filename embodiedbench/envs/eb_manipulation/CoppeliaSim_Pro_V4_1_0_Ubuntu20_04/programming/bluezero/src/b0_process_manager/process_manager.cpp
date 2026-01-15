#include <map>
#include <b0/node.h>
#include <b0/service_server.h>
#include <b0/publisher.h>
#include "protocol.h"
#ifdef HAVE_BOOST_PROCESS
#ifdef HAVE_POSIX_SIGNALS
#include <signal.h>
#endif // HAVE_POSIX_SIGNALS
#include <boost/process.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/dll.hpp>

namespace bp = boost::process;

namespace b0
{

namespace process_manager
{

class ProcessManager;

ProcessManager *instance = nullptr;

class ProcessManager : public b0::Node
{
public:
    ProcessManager()
        : Node("process_manager@%h"),
          beacon_pub_(this, "process_manager/beacon"),
          srv_(this, "%n/control", &ProcessManager::handleRequest, this)
    {
        if(instance)
            throw std::runtime_error("ProcessManager constructed multiple times");

        instance = this;
    }

    ~ProcessManager()
    {
        instance = nullptr;
    }

    void handleRequest(const Request &req, Response &rep)
    {
        if(0) {}
#define HANDLER(N) else if(req.N) { rep.N.emplace(); handle_##N(*req.N, *rep.N); }
        HANDLER(start_process)
        HANDLER(stop_process)
        HANDLER(query_process_status)
        HANDLER(list_active_processes)
#undef HANDLER
        else
        {
            error("bad request");
        }
    }

    void handle_start_process(const StartProcessRequest &req, StartProcessResponse &rep)
    {
        if(!canLaunchProgram(req.path))
        {
            error("Failed to launch %s: permission denied.", req.path);
            rep.success = false;
            rep.error_message = "permission denied";
            return;
        }
        auto c = new bp::child(req.path, req.args);
        children_[c->id()].child_ = std::shared_ptr<bp::child>(c);
        info("Process %d (%s) started.", c->id(), req.path);
        rep.success = true;
        rep.pid = c->id();
    }

    void handle_stop_process(const StopProcessRequest &req, StopProcessResponse &rep)
    {
        auto it = children_.find(req.pid);
        if(it == children_.end())
        {
            error("Failed to stop PID %d which is not managed by this process manager.", req.pid);
            rep.success = false;
            rep.error_message = (boost::format("PID %d is not managed by this process manager (%s)") % req.pid % getName()).str();
            return;
        }
        auto c = it->second.child_;
#ifdef HAVE_POSIX_SIGNALS
        info("Sending SIGINT to process %d...", c->id());
        kill(c->id(), SIGINT);
        it->second.int_requested_ = timeUSec();
#else
        info("Terminating process %d...", c->id());
        c->terminate();
#endif
        rep.success = true;
    }

    void handle_query_process_status(const QueryProcessStatusRequest &req, QueryProcessStatusResponse &rep)
    {
        auto it = children_.find(req.pid);
        if(it == children_.end())
        {
            rep.success = false;
            rep.error_message = "no such pid";
            return;
        }
        auto c = it->second.child_;
        rep.running = c->running();
        if(!rep.running)
            rep.exit_code = c->exit_code();
        rep.success = true;
    }

    void handle_list_active_processes(const ListActiveProcessesRequest &req, ListActiveProcessesResponse &rep)
    {
        for(auto &p : children_)
        {
            auto c = p.second.child_;
            rep.pids.push_back(c->id());
        }
    }

    bool canLaunchProgram(std::string p)
    {
        // security measure: only allow certain programs to be launched
        // current implementation: allow programs contained in current directory

        boost::filesystem::path self_path = boost::dll::program_location().parent_path();
        std::string sp = self_path.string();

        // XXX: if this is launched with ./b0_process_manager, sp ends with .
        if(sp[sp.size() - 1] == '.')
            sp = sp.substr(0, sp.size() - 1);

        // filter attempts to bypass this security measure:
        if(p.find("/../") != std::string::npos) return false;

        if(!boost::starts_with(p, sp))
        {
            error("Permission denied: '%s' does not start with '%s'.", p, sp);
            return false;
        }

        return true;
    }

    void cleanup()
    {
        for(auto it = children_.begin(); it != children_.end(); )
        {
            auto c = it->second.child_;
            if(c->running())
            {
#ifdef HAVE_POSIX_SIGNALS
                // after 5s from SIGINT, try with SIGTERM
                if(it->second.int_requested_ && timeUSec() - it->second.int_requested_ > 5000000)
                {
                    warn("Escalating to SIGTERM for process %d...", c->id());
                    kill(c->id(), SIGTERM);
                    it->second.term_requested_ = timeUSec();
                }

                // after 5s from SIGTERM, try with SIGKILL
                if(it->second.term_requested_ && timeUSec() - it->second.term_requested_ > 5000000)
                {
                    warn("Escalating to SIGKILL for process %d...", c->id());
                    c->terminate();
                }
#endif

                ++it;
            }
            else
            {
                c->wait();
                info("Process %d finished with exit code %d.", c->id(), c->exit_code());
                it = children_.erase(it);
            }
        }
    }

    void sendBeacon()
    {
        Beacon beacon;
        beacon.host_name = hostname();
        beacon.node_name = getName();
        beacon.service_name = srv_.getName();
        beacon_pub_.publish(beacon);
    }

    void spinOnce()
    {
        Node::spinOnce();
        cleanup();
        sendBeacon();
    }

    void killChildren(int signal)
    {
        for(auto it = children_.begin(); it != children_.end(); )
        {
            auto c = it->second.child_;
#ifdef HAVE_POSIX_SIGNALS
            kill(c->id(), signal);
#endif
        }
    }

protected:
    struct Child
    {
        std::shared_ptr<bp::child> child_;
        int64_t int_requested_ = 0;
        int64_t term_requested_ = 0;
    };

    b0::ServiceServer srv_;
    b0::Publisher beacon_pub_;
    std::map<pid_t, Child> children_;
};

} // namespace process_manager

} // namespace b0

void signalHandler(int signum)
{
    if(b0::process_manager::instance)
        b0::process_manager::instance->killChildren(signum);
}

void registerSignalHandlers()
{
#ifdef HAVE_POSIX_SIGNALS
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    signal(SIGKILL, signalHandler);
    signal(SIGABRT, signalHandler);
    signal(SIGFPE, signalHandler);
    signal(SIGILL, signalHandler);
    signal(SIGSEGV, signalHandler);
#endif
}

int main(int argc, char **argv)
{
    registerSignalHandlers();
    b0::init(argc, argv);
    b0::process_manager::ProcessManager node;
    node.init();
    node.spin();
    node.cleanup();
    return 0;
}

#else // HAVE_BOOST_PROCESS

#include <iostream>

int main()
{
    std::cerr << "boost/process.hpp is needed for process_manager" << std::endl;
    return 1;
}

#endif // HAVE_BOOST_PROCESS
