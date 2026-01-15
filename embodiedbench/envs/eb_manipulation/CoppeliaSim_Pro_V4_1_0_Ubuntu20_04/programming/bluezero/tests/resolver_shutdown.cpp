#include <iostream>
#include <b0/resolver/resolver.h>

int main(int argc, char **argv)
{
    b0::init(argc, argv);

    b0::resolver::Resolver node;
    node.init();
    int i;
    while (i++ < 20) {
        node.spinOnce();
        boost::this_thread::sleep_for(boost::chrono::milliseconds{100});
    }
    std::cout << "Resolver cleanup" << std::endl;
    node.cleanup();
    std::cout << "Resolver closing" << std::endl;
}

