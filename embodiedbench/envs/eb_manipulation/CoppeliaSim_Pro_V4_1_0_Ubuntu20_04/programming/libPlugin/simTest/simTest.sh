#!/bin/sh

if [ "x$COPPELIASIM_ROOT_DIR" = "x" ]; then
    echo "error: COPPELIASIM_ROOT_DIR environment variable is not set" 1>&2
    exit 1
fi

# figure out simTest's directory:
simTest_dir="$(cd "$(dirname "$0")"; pwd)"

overwrite_output=1

if [ "x$1" = "x" -o "x$1" = "x-h" -o "x$1" = "x--help" ]; then
    echo "usage: $(basename "$0") <input-directory> [ <output-directory> ]"
    if [ "x$1" = "x" ]; then exit 1; else exit 0; fi
fi

if [ "x$1" != "x" ]; then
    input_dir="$(readlink -f "$1")"
fi

if [ "x$2" != "x" ]; then
    output_dir="$(readlink -f "$2")"
else
    output_dir="$(mktemp -t -d "simTest.$(basename "$input_dir").$(date +%Y%m%d%H%MS).XXXX")"
    echo "warning: output directory has not been specified. using $output_dir" 1>&2
fi

if [ ! -d "$output_dir" ]; then
    echo "error: specified output directory $output_dir is not a directory" 1>&2
    exit 1
fi

if [ "$overwrite_output" -gt 0 ]; then
    rm -rf "$output_dir"/*
fi

if [ $(find "$output_dir" | wc -l) -gt 1 ]; then
    echo "error: output directory $output_dir is not empty" 1>&2
    exit 1
fi

export LD_LIBRARY_PATH="$COPPELIASIM_ROOT_DIR:$LD_LIBRARY_PATH"
"$COPPELIASIM_ROOT_DIR/coppeliaSim" \
    -vdebug \
    -xnone \
    -a"$simTest_dir/simTest_addon.lua" \
    -Ginput_dir="$input_dir" \
    -Goutput_dir="$output_dir" \
    -GsimTest_dir="$simTest_dir" \
#    1> "$output_dir/stdout.log" \
#    2> "$output_dir/stderr.log"

if [ -f "$output_dir/result.txt" ]; then
    cat "$output_dir/result.txt"
fi

if [ -f "$output_dir/exitcode.txt" ]; then
    exitcode="$(cat "$output_dir/exitcode.txt")"
    if [ "$exitcode" -eq "$exitcode" ]; then
        exit $exitcode
    else
        echo "error: the test did not produce an INTEGER exitcode" 1>&2
        exit 1
    fi
else
    echo "error: the test did not produce an exitcode" 1>&2
    exit 1
fi
