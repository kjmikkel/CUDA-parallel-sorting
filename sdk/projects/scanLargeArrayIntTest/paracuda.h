#ifndef PARACUDA_H
#define PARACUDA_H

#include <cutil_inline.h>
#include "paracuda_struct.h"
#include "paracuda_map.h"
#include "paracuda_scan.h"
#include "paracuda_permute.h"
//#include "paracuda_segmentedScan.h"


/* Easy initialization of CUDA. It is optional to call this. */
/* You are not required to call this. */
void PARACUDA_INITIALIZE_CUDA(int argc, char** argv) 
{
    if(cutCheckCmdLineFlag(argc, (const char**) argv, "device"))
        cutilDeviceInit(argc, argv);
    else
        cudaSetDevice(cutGetMaxGflopsDeviceId());
}

/* Exits the program, waiting for a keypress. */
/* You are not required to call this. */
void PARACUDA_EXIT_KEYPRESS(int argc, char** argv)
{
    cudaThreadExit();
    cutilExit(argc, argv);
}

/* Calculates the smallest power of two which is greater than or equal to number. */
int PARACUDA_NEXT_POWER_OF_TWO(
    size_t number) 
{
    size_t count = 0;
    while(1 << count < number) count++;
    return count;
}

/* Asserts that a number is a power of two (prints an error and exits otherwise). */
void PARACUDA_ASSERT_POWER_OF_TWO(
    size_t number) 
{
    if(1 << PARACUDA_NEXT_POWER_OF_TWO(number) != number) {
        fprintf(stderr, "Length must be a power of two, but was %d.\n", number);
        exit(1);
    }
}

/* Base 2 integer logarithm. */
size_t PARACUDA_LOG2(size_t n) {
    size_t l = 0;
    while(n > 1) {
        n >>= 1;
        l += 1;
    }
    return l;
}

#define PARACUDA_MAX_THREADS 512

/* Defines an operator. The parenthesis around the arguments and the body are required: */
/* PARACUDA_OPERATOR(name, void, (int x, int y), ({ ... })) */
/* It may create additional definitions that have NAME_ as a prefix. */
#define PARACUDA_OPERATOR(NAME, RETURN, ARGUMENTS, BODY) \
RETURN NAME ARGUMENTS { BODY; } \
__inline__ __device__ RETURN PARACUDA_##NAME##_operator ARGUMENTS { BODY; }

#endif

