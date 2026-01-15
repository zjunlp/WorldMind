#!/bin/sh
set -e
set -a

thisscript="$0"
while [ -L "$thisscript" ]; do
        thisscript="`readlink "$thisscript"`"
done

dirname=`dirname "$thisscript"`
if [ $dirname = "." ]; then
        dirname="$PWD"
fi

appname="`basename "$thisscript" | sed 's,\.sh$,,'`"

LD_LIBRARY_PATH="$dirname:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH
ldd "$dirname/libcoppeliaSim.so" #replace libcoppeliaSim.so with the library that is having difficulties to load 

