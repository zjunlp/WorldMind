#include <sstream>
#include <vector>
#include <map>
#include <iostream>
#include <b0/node.h>
#include <b0/service_client.h>
#include <b0/service_server.h>
#include <b0/publisher.h>
#include <b0/subscriber.h>
#include "protocol.h"

namespace b0
{

namespace process_manager
{

class HUB : public b0::Node
{
public:
    HUB()
        : Node("process_manager_hub"),
          srv_(this, "process_manager_hub/control", &HUB::handleRequest, this),
          beacon_sub_(this, "process_manager/beacon", &HUB::onBeacon, this),
          active_nodes_pub_(this, "process_manager_hub/active_nodes")
    {
    }

    ~HUB()
    {
    }

    void onBeacon(const Beacon &beacon)
    {
        auto it = clients_.find(beacon.host_name);
        if(it == clients_.end())
        {
            add(beacon);
            return;
        }
        else
        {
            auto &client = it->second;
            client.last_active_ = timeUSec();
        }
    }

    void handleRequest(const HUBRequest &req, HUBResponse &rsp)
    {
        auto it = clients_.find(req.host_name);
        if(it == clients_.end())
        {
            error("bad request: unknown host");
            rsp.success = false;
            rsp.error_message = "unknown host";
            return;
        }

        auto &client = it->second;
        Request req1(req);
        Response rsp1;
        client.cli_->call(req1, rsp1);
        rsp = rsp1;
        rsp.success = true;
    }

    void add(const Beacon &beacon)
    {
        clients_[beacon.host_name].last_active_ = timeUSec();
        clients_[beacon.host_name].cli_.reset(new b0::ServiceClient(this, beacon.service_name, false));
        info("added new entry: %s -> %s", beacon.host_name, beacon.service_name);
        clients_[beacon.host_name].cli_->init();
    }

    void remove(const std::string &host_name)
    {
        auto it = clients_.find(host_name);
        if(it == clients_.end()) return;
        clients_[host_name].cli_->cleanup();
        clients_.erase(it);
        info("removed entry: %s", host_name);
    }

    void removeInactive()
    {
        std::vector<std::string> removed;
        for(auto it = clients_.begin(); it != clients_.end(); ++it)
        {
            auto &client = it->second;
            int64_t diff = timeUSec() - client.last_active_;
            if(diff > 1000000)
                removed.push_back(it->first);
        }
        for(auto &x : removed)
            remove(x);
    }

    void broadcastActive()
    {
        ActiveNodes msg;
        for(auto it = clients_.begin(); it != clients_.end(); ++it)
        {
            NodeActivity act;
            act.host_name = it->first;
            //TODO: act.node_name = ...;
            //TODO: act.service_name = ...;
            act.last_active = it->second.last_active_;
            msg.nodes.push_back(act);
        }
        active_nodes_pub_.publish(msg);
    }

    void spinOnce()
    {
        Node::spinOnce();
        removeInactive();
        broadcastActive();
    }

protected:
    struct Client
    {
        std::unique_ptr<b0::ServiceClient> cli_;
        int64_t last_active_;
    };

    std::map<std::string, Client> clients_;
    b0::ServiceServer srv_;
    b0::Subscriber beacon_sub_;
    b0::Publisher active_nodes_pub_;
};

} // namespace process_manager

} // namespace b0

int main(int argc, char **argv)
{
    b0::setSpinRate(2.0);
    b0::init(argc, argv);
    b0::process_manager::HUB node;
    node.init();
    node.spin();
    node.cleanup();
    return 0;
}

