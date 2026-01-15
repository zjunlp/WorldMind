#include <string>
#include <iostream>

#include <b0/resolver/resolver.h>

int main(int argc, char **argv)
{
    b0::addOptionInt64("minimum-heartbeat-interval,o", "set the minimum heartbeat interval, in microseconds (an interval of 0us will disable online monitoring)", nullptr, false, 30000000);
    b0::init(argc, argv);

    b0::resolver::Resolver node;
    if(b0::hasOption("minimum-heartbeat-interval"))
    {
        int64_t interval = b0::getOptionInt64("minimum-heartbeat-interval");
        node.setMinimumHeartbeatInterval(interval);
        if(interval == 0)
            node.warn("Online monitoring is disabled");
    }

    node.init();
    node.spin();
    node.cleanup();

    return 0;
}

