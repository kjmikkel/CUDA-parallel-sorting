#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cutil_inline.h>

#include <gold.h>

#include <kernel.cu>
#include <work.cu>

// Hack to get around a seemingly broken part of the emulator
//#define cutilCheckMsg(x) ((void) (x))
// Hack to define element_t temporarily (it's not sufficient to change this)
#define element_t int

__host__ void
printArray(float* array, int length) {
    for(int i = 0; i < length; ++i)
    {
        printf("%d: %f\n", i, array[i]);
    }
}

__host__ void
printArray(int* array, int length) {
    for(int i = 0; i < length; ++i)
    {
        printf("%d: %d\n", i, array[i]);
    }
}

__host__ void
printArray(int* array, bool* flags, int length) {
    for(int i = 0; i < length; ++i)
    {
      printf("%d: %d %d\n", i, array[i], flags[i]);
    }
}

__host__ void
runTest(int argc, char** argv) 
{
    if(cutCheckCmdLineFlag(argc, (const char**)argv, "device"))
        cutilDeviceInit(argc, argv);
    else
        cudaSetDevice(cutGetMaxGflopsDeviceId());

    unsigned int timer = 0;
    cutilCheckError(cutCreateTimer(&timer));
    cutilCheckError(cutStartTimer(timer));

    unsigned int num_threads = 1 << 5;
    unsigned int mem_size = sizeof(element_t) * num_threads;

    element_t h_array[num_threads];
    for(unsigned int i = 0; i < num_threads; ++i) 
    {
        h_array[i] = i;
    }

    element_t reference[num_threads];
    memcpy(reference, h_array, mem_size);

    unsigned int max_threads = 32;

    element_t* h_result;
    int count = packEven(h_array, num_threads, &h_result, max_threads);
    cutilCheckMsg("Kernel execution failed");

    cutilCheckError(cutStopTimer(timer));
    printf("Processing time: %f (ms)\n", cutGetTimerValue(timer));
    cutilCheckError(cutDeleteTimer(timer));

    // compute reference solution
    //compute2(reference, num_threads);

    // debug printing
    printf("Result:\n");
    printArray(h_result, count);

    free(h_result);

    cudaThreadExit();
}

