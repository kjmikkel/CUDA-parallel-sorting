#include "cutil_inline.h"

extern void runTest(int argc, char** argv);

int main(int argc, char** argv) 
{
    runTest(argc, argv);
    cutilExit(argc, argv);
}

