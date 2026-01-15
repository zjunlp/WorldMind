#include <b0/node.h>
#include <b0/publisher.h>
#include <b0/subscriber.h>

#include <boost/lexical_cast.hpp>

/*! \example remapping/operation.cpp
 * This node performs a mathematical operation on the data received on its two topics
 */

//! \cond HIDDEN_SYMBOLS

class Operation : public b0::Node
{
public:
    Operation(char op)
        : b0::Node("operation"),
          op_(op),
          sub_a_(this, "a", &Operation::callback_a, this),
          sub_b_(this, "b", &Operation::callback_b, this),
          pub_(this, "out"),
          value_a_(0),
          value_b_(0)
    {
        if(op != '+' && op != '-' && op != '*' && op != '/')
            throw std::runtime_error("invalid operator");
    }

    void callback_a(const std::string &msg)
    {
        value_a_ = boost::lexical_cast<int>(msg);
        compute();
    }

    void callback_b(const std::string &msg)
    {
        value_b_ = boost::lexical_cast<int>(msg);
        compute();
    }

    void compute()
    {
        int result = 0;
        switch(op_)
        {
        case '+': result = value_a_ + value_b_; break;
        case '-': result = value_a_ - value_b_; break;
        case '*': result = value_a_ * value_b_; break;
        case '/': result = value_a_ / value_b_; break;
        }
        pub_.publish(boost::lexical_cast<std::string>(result));
    }

private:
    char op_;
    b0::Subscriber sub_a_;
    b0::Subscriber sub_b_;
    b0::Publisher pub_;
    int value_a_;
    int value_b_;
};

int main(int argc, char **argv)
{
    b0::addOptionString("operator,o", "the mathematical operator", nullptr, true);
    b0::init(argc, argv);
    Operation node(b0::getOptionString("operator")[0]);
    node.init();
    node.spin();
    node.cleanup();
    return 0;
}

//! \endcond

