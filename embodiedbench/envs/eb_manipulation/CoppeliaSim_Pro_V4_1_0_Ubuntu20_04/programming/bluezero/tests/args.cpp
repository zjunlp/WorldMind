#include <iostream>
#include <b0/b0.h>

void fail(const char *arg)
{
    std::cerr << "test failed on argument " << arg << std::endl;
    std::cerr << "call with arguments: -a 1 -a 2 -a 3 -b 0.5 -c 281474976710656 -n 4 w x y z" << std::endl;
    exit(1);
}

void success()
{
    exit(0);
}

int main(int argc, char **argv)
{
    std::vector<int> a;
    b0::addOptionIntVector("aaa,a", "", &a, true);
    double b;
    b0::addOptionDouble("bbb,b", "", &b, true);
    int64_t c;
    b0::addOptionInt64("ccc,c", "", &c, true);
    int n;
    b0::addOptionInt("nnn,n", "", &n, true);
    std::vector<std::string> args;
    b0::addOptionStringVector("args", "", &args, true);
    b0::setPositionalOption("args", -1);
    b0::init(argc, argv);

    if(a.size() != 3) fail("a.size()");
    if(a[0] != 1 || a[1] != 2 || a[2] != 3) fail("a[i]");
    if(b != 0.5) fail("b");
    if(c != 281474976710656) fail("c");
    if(n != 4) fail("n");
    if(args.size() != 4) fail("args.size()");
    if(args[0] != "w" || args[1] != "x" || args[2] != "y" || args[3] != "z") fail("args[i]");
    success();
}

