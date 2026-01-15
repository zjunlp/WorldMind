# C++ plugin framework for CoppeliaSim

Compile with your C++ project.

Example plugin (uses also [simStubsGen](https://github.com/CoppeliaRobotics/libPlugin/simStubsGen)):

```
#include "simExtPluginSkeletonNG.h"
#include "simPlusPlus/Plugin.h"
#include "stubs.h"

class Plugin : public sim:Plugin
{
public:
    void onStart()
    {
        if(!registerScriptStuff())
            throw std::runtime_error("script stuff initialization failed");
    }

    void test(test_in *in, test_out *out)
    {
        // ...
    }
};

SIM_PLUGIN("PluginSkeletonNG", 1, Plugin)
#include "stubsPlusPlus.cpp"
```

See [simExtPluginSkeletonNG](https://github.com/CoppeliaRobotics/simExtPluginSkeletonNG) for a complete example.
