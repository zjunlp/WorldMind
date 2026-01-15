# BlueZero v2

BlueZero (in short, "B0") is a cross-platform middleware which provides tools for interconnecting pieces of software running in multiple threads, processes or machines.

It has some similarities with ROS, although it only focuses on providing communication paradigms (client/server and publisher/subscriber) and message transport (based on ZeroMQ), while being agnostic to message serialization format or common protocols and data structures.

This project uses [semantic versioning](https://semver.org).

## Documentation

See the generated `docs/index.html` for the documentation.

## License

See [LICENSE.md](LICENSE.md).

## Dependencies

### Required Libraries

 - ZeroMQ v4.1+
 - Boost v1.54+

### Optional Libraries
 - zlib (for compressing payloads)
 - lz4 (for compressing payloads)
 - Doxygen (for generating documentation)
 - Graphviz (for generating documentation)
 - mscgen (for generating documentation)

### Ubuntu 16

Install the following apt packages:

 - libzmq3-dev
 - libboost-all-dev

### Ubuntu 14

Install the following apt packages:

 - libboost-all-dev

The ZeroMQ apt package is too old. Compile and install it from the git repository:

 - https://github.com/zeromq/libzmq

### Windows (vcpkg)

Install the following vcpkg packages:

 - cmake
 - boost
 - zlib
 - zeromq

### macOS (homebrew)

Install the following:

 - cmake
 - boost
 - zlib
 - zeromq

## Compiling

```bash
git clone --branch v2 --recursive https://github.com/CoppeliaRobotics/bluezero
cd bluezero
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_GUI=ON -DBUILD_EXAMPLES=ON ..
make
```

Optionally, generate the documentation (requires [Doxygen](http://www.doxygen.org), [Graphviz](http://www.graphviz.org) and [mscgen](http://www.mcternan.me.uk/mscgen/) installed):
```
make doc
```

Optionally, run unit tests:
```
make test
```

