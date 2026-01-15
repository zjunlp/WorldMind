#include <sstream>
#include <iostream>
#include <iomanip>

#include <b0/compress/compress.h>

std::string generatePayload(size_t size)
{
    std::stringstream ss;
    std::string in = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            "aaaabababababbbbababbbbbbbacccccccccccccccccccccccccccccc";
    for(size_t i = 0; i < size; i++)
        ss << in[i % in.size()];
    return ss.str();
}

int main(int argc, char **argv)
{
    std::string algo;
    b0::addOptionString("algorithm,a", "compression algorithm to test", &algo, true);
    b0::setPositionalOption("algorithm");
    b0::init(argc, argv);

    size_t size[] = {100, 500, 2000, 8000, 25000, 100000, 1000000, 5000000};
    for(int j = 0; j < sizeof(size)/sizeof(size[0]); j++)
    {
        for(int level = -1; level <= 9; level++)
        {
            std::cout << "Testing " << algo << " level " << level << std::endl;
            std::string in = generatePayload(size[j]);
            std::cout << "    in size: " << in.size() << std::endl;
            std::string out = b0::compress::compress(algo, in, level);
            std::cout << "    out size: " << out.size() << std::endl;
            if(in == out)
            {
                std::cerr << "error: compression algorithm returned the string unchanged" << std::endl;
                exit(1);
            }
            if(!(out.size() < in.size()))
            {
                std::cerr << "warning: compression algorithm produced an output bigger than the input" << std::endl;
            }
            std::string in2 = b0::compress::decompress(algo, out);
            std::cout << "    in2 size: " << in2.size() << std::endl;
            if(in != in2)
            {
                std::cerr << "error: compression algorithm does not work correctly (x != decompress(compress(x)))" << std::endl;
                std::cerr << "in: " << in << std::endl;
                std::cerr << "in2: " << in2 << std::endl;
                exit(1);
            }
        }
    }
    exit(0);
}

