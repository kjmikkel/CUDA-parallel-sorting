#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cutil_inline.h>

#include <gold.h>

#include <kernel.cu>
#include <work.cu>

// Hack to get around a seemingly broken part of the emulator
#define cutilCheckMsg(x) ((void) (x))
// Hack to define element_t temporarily (it's not sufficient to change this)
#define element_t int

__host__ void
printArray(int* array, int length) {
    for(int i = 0; i < length; ++i)
    {
        printf("%d: %d\n", i, array[i]);
    }
}

__host__ void
printArray(int* array, int* flags, int length) {
    for(int i = 0; i < length; ++i)
    {
      printf("%d: %d %d\n", i, array[i], flags[i]);
    }
}

__host__ void
runTest(int argc, char** argv) 
{
	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	if(cutCheckCmdLineFlag(argc, (const char**)argv, "device"))
		cutilDeviceInit(argc, argv);
	else
        cudaSetDevice(cutGetMaxGflopsDeviceId());

    unsigned int timer = 0;
    cutilCheckError(cutCreateTimer(&timer));
    cutilCheckError(cutStartTimer(timer));

    unsigned int num_threads = 1 << 5;
    unsigned int mem_size = sizeof(element_t) * num_threads;

    // allocate host memory
    element_t* h_array = (element_t*) malloc(mem_size);
    int* h_flags = (int*) malloc(mem_size * 2);
    int* h_result = (int*) malloc(mem_size * 2);
    // initalize the memory
    int value;
    for(unsigned int i = 0; i < num_threads; ++i) 
    {
        h_array[i] = i;

        if (i % 3 == 0) 
	  value = 1;
	else 
	  value = 0;

	h_flags[i] = value;
        h_result[i] = h_flags[i];
    }

    //    for(unsigned int i = 0; i < num_threads; ++i) {
    // printf("%i\n", h_flags[i]);
    //}

    element_t* reference = (element_t*) malloc(mem_size);
    memcpy(reference, h_array, mem_size);

    // allocate device memory
    element_t* d_array;
    cutilSafeCall(cudaMalloc((void**) &d_array, mem_size));
    // copy host memory to device
    cutilSafeCall(cudaMemcpy(d_array, h_array, mem_size,
        cudaMemcpyHostToDevice));
    // allocate device memory
    int* d_flags;
    cutilSafeCall(cudaMalloc((void**) &d_flags, mem_size));
    // copy host memory to device
    cutilSafeCall(cudaMemcpy(d_flags, h_flags, mem_size,
        cudaMemcpyHostToDevice));
    // allocate device memory
    int* d_result;
    cutilSafeCall(cudaMalloc((void**) &d_result, mem_size));
    // copy host memory to device
    cutilSafeCall(cudaMemcpy(d_result, h_result, mem_size,
        cudaMemcpyHostToDevice));

    unsigned int max_threads = 32;

    split<element_t>(d_array, d_result, d_flags, num_threads, max_threads);

    cutilCheckMsg("Kernel execution failed");

    // copy result from device to host
    cutilSafeCall(cudaMemcpy(h_array, d_result, mem_size, cudaMemcpyDeviceToHost));

    cutilCheckError(cutStopTimer(timer));
    printf("Processing time: %f (ms)\n", cutGetTimerValue(timer));
    cutilCheckError(cutDeleteTimer(timer));

    // compute reference solution
    // compute(reference, h_flags, num_threads);

    // debug printing
    printf("Result:\n");
    printArray(h_array, d_flags, num_threads);
    //  printf("Expected:\n");
    //printArray(reference, h_flags, num_threads);

    // check result
    if(cutCheckCmdLineFlag(argc, (const char**) argv, "regression")) 
    {
	    // write file for regression test
	    cutilCheckError(cutWriteFilei("./data/regression.dat",
					    h_array, num_threads, 0.0));
    }
    else 
    {
        // custom output handling when no regression test running
        // in this case check if the result is equivalent to the expected soluion
	    CUTBoolean res = cutComparei(h_array, reference, num_threads);
	    printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");
    }

    // cleanup memory
    free(h_array);
    cutilSafeCall(cudaFree(d_array));

    cudaThreadExit();
}

