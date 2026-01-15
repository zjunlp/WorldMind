#!/bin/bash

LOCATION_OF_B0_C_API=../../../bluezero/include/b0/bindings
LOCATION_OF_B0_LIB=../../../bluezero/build #libb0.so is normally located in the same folder as coppeliaSim.sh. libb0.dylib is normally located in coppeliaSim.app/Contents/Frameworks/

g++ -std=c++11 -I$LOCATION_OF_B0_C_API -I../msgpack-c/include -I.. ../b0RemoteApi.cpp synchronousImageTransmission.cpp -L$LOCATION_OF_B0_LIB -lb0 -lboost_system -o synchronousImageTransmission
