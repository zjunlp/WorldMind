#!/bin/sh
set -e -x
if [ ! -d build/coverage ]; then
    mkdir -p build/coverage
fi
cd build/coverage
if [ ! -f CMakeCache.txt ]; then
    cmake \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_TESTS=ON \
        -DBUILD_GUI=OFF \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_CXX_FLAGS="--coverage" \
        ../..
fi
make
find . -iname '*.gcda' -exec rm '{}' ';'
# '*.gcno' files are created at compile time, do not remove
B0_HOST_ID=localhost make test || true
gcov $(find . -name '*.gcno')
