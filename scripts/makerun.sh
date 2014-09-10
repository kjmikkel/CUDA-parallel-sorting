#!/bin/bash
# Compiles and runs a project.
# Run this like:
# bash scripts/makerun.sh <project> <args>

p=$1
shift 1
rm -r "sdk/projects/$p/obj";
cd sdk && \
make project=$p && \
env "PATH=$PATH:/usr/local/cuda/bin" "LD_LIBRARY_PATH=.:/usr/local/cuda/lib" "./bin/linux/release/$p" $*
