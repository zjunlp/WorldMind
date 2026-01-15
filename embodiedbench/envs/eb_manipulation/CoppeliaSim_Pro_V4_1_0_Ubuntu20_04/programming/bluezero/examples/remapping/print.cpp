#include <b0/node.h>
#include <b0/subscriber.h>

#include <iostream>

/*! \example remapping/print.cpp
 * This node print what it receives on its topic
 */

//! \cond HIDDEN_SYMBOLS

class Print : public b0::Node
{
public:
    Print()
        : b0::Node("print"),
          sub_(this, "in", &Print::callback, this)
    {
    }

    void callback(const std::string &msg)
    {
        std::cout << "Received: " << msg << std::endl;
    }

private:
    b0::Subscriber sub_;
};

int main(int argc, char **argv)
{
    b0::init(argc, argv);
    Print node;
    node.init();
    node.spin();
    node.cleanup();
    return 0;
}

//! \endcond

