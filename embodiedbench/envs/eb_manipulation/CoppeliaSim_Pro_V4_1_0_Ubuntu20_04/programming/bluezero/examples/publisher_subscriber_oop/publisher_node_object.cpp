#include <b0/node.h>
#include <b0/publisher.h>

#include <iostream>
#include <boost/lexical_cast.hpp>

/*! \example publisher_subscriber_oop/publisher_node_object.cpp
 * This is an example of creating a node by subclassing b0::Node.
 * Useful for overriding some node's behavior.
 *
 * Since b0::Node uses two-phase inizialization, you <b>must not</b> call
 * b0::Node::init() in the constructor, nor b0::Node::cleanup() in the destructor.
 */

//! \cond HIDDEN_SYMBOLS

class TestPublisherNode : public b0::Node
{
public:
    TestPublisherNode()
        : Node("publisher"),
          pub_(this, "A")
    {
    }

    void spinOnce() override
    {
        static int i = 0;
        b0::Node::spinOnce();

        std::string msg = (boost::format("msg-%d") % i++).str();
        std::cout << "Sending: " << msg << std::endl;
        pub_.publish(msg);
    }

private:
    b0::Publisher pub_;
};

int main(int argc, char **argv)
{
    b0::init(argc, argv);
    TestPublisherNode node;
    node.init();
    node.spin();
    node.cleanup();
    return 0;
}

//! \endcond

