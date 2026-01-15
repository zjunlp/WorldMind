#include <b0/node.h>
#include <b0/subscriber.h>

#include <iostream>

/*! \example publisher_subscriber/subscriber_node_object_manual.cpp
 * This is an advanced usage example of manually reading from a subscriber
 */

//! \cond HIDDEN_SYMBOLS

class TestSubscriberNode : public b0::Node
{
public:
    TestSubscriberNode()
        : Node("subscriber"),
          sub_(this, "A")
    {
    }

    void run()
    {
        while(!shutdownRequested())
        {
            /*
             * In this example we don't call Node::spinOnce() to process
             * incoming messages, but we directly read from this node's sub_'s socket.
             *
             * In general, you should NEVER need to do this, but this is an example
             * of one way to do it, in case you really need to.
             */

            std::string msg;

            // read is blocking
            sub_.readRaw(msg);

            std::cout << "Received: " << msg << std::endl;
        }
    }

private:
    b0::Subscriber sub_;
};

int main(int argc, char **argv)
{
    b0::init(argc, argv);
    TestSubscriberNode node;
    node.init();
    node.run();
    node.cleanup();
    return 0;
}

//! \endcond

