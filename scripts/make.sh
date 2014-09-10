#!/bin/bash
# Compiles and runs a project.
# Run this like:
# bash scripts/makerun.sh <project> <args>

cd sdk && \
make emu=1 project=$1

