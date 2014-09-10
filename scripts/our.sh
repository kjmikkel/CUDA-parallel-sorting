#!/bin/bash
# Lists our own projects (not those that come with the SDK).
# Run this like:
# bash scripts/our.sh

(ls -1 sdk/projects/; cat scripts/original.txt; cat scripts/original.txt) | sort | uniq -u
