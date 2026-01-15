#include <b0/b0.h>

class Node
{
public:
    Node(const std::string &name)
        : node_(new b0::Node(name))
    {
    }

    ~Node()
    {
        delete node_;
    }

    void init()
    {
        node_->init();
    }

    void cleanup()
    {
        node_->cleanup();
    }

    void shutdown()
    {
        node_->shutdown();
    }

    bool shutdownRequested()
    {
        return node_->shutdownRequested();
    }

    void spin()
    {
        node_->spin();
    }

    void spinOnce()
    {
        node_->spinOnce();
    }

    std::string getName()
    {
        return node_->getName();
    }

    int getState()
    {
        return node_->getState();
    }

    int64_t hardwareTimeUSec()
    {
        return node_->hardwareTimeUSec();
    }

    int64_t timeUSec()
    {
        return node_->timeUSec();
    }

private:
    b0::Node *node_;

    friend class Publisher;
    friend class Subscriber;
};

class Publisher
{
public:
    Publisher(Node &node, const std::string &topic)
        : pub_(new b0::Publisher(node.node_, topic))
    {
    }

    ~Publisher()
    {
        delete pub_;
    }

    void publish(const std::string &data)
    {
        pub_->publish(data);
    }

private:
    b0::Publisher *pub_;
};

class SubscriberCallback
{
public:
    void onMessage(const std::string &message) {}
};

class Subscriber
{
public:
    Subscriber(Node &node, const std::string &topic, SubscriberCallback *cb)
        : cb_(cb),
          sub_(new b0::Subscriber(node.node_, topic, [=](const std::string &m) {cb_->onMessage(m);}))
    {
    }

    ~Subscriber()
    {
        delete sub_;
    }

private:
    SubscriberCallback *cb_;
    b0::Subscriber *sub_;
};


