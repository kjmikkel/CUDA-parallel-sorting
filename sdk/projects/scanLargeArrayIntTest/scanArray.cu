/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

#ifdef _WIN32
#  define NOMINMAX 
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cutil_inline.h>

// includes, kernels
#include <scan.cu>  // defines prescanArray()

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void scan( int* output, int* intput, int num_elements);

// regression test functionality
extern "C" 
unsigned int compare( const int* reference, const int* data, 
                     const unsigned int len);
extern "C" 
void computeGold( int* reference, int* idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
//int 
//main( int argc, char** argv) 
//{
//    runTest( argc, argv);
//    cutilExit(argc, argv);
//}

////////////////////////////////////////////////////////////////////////////////
//! Run a scan test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
scan(int* output,  int* h_data, int num_elements) 
{
    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
  //  if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
  //      cutilDeviceInit(argc, argv);
  //  else
  //      cudaSetDevice( cutGetMaxGflopsDeviceId() );
    
    unsigned int mem_size = sizeof( int) * num_elements;
    
    unsigned int timerGPU, timerCPU;
    cutilCheckError(cutCreateTimer(&timerCPU));
    cutilCheckError(cutCreateTimer(&timerGPU));

    // compute reference solution
    int* reference = (int*) malloc( mem_size); 
    cutStartTimer(timerCPU);
//    computeGold( reference, h_data, num_elements);
    cutStopTimer(timerCPU);

    // allocate device memory input and output arrays
    int* d_idata = NULL;
    int* d_odata = NULL;

    cutilSafeCall( cudaMalloc( (void**) &d_idata, mem_size));
    cutilSafeCall( cudaMalloc( (void**) &d_odata, mem_size));
    
    // copy host memory to device input array
    cutilSafeCall( cudaMemcpy( d_idata, h_data, mem_size, cudaMemcpyHostToDevice) );
    // initialize all the other device arrays to be safe
    cutilSafeCall( cudaMemcpy( d_odata, h_data, mem_size, cudaMemcpyHostToDevice) );

    printf("Running parallel prefix sum (prescan) of %d elements\n", num_elements);
    printf("This version is work efficient (O(n) adds)\n");
    printf("and has very few shared memory bank conflicts\n\n");

    preallocBlockSums(num_elements);

    // Run the prescan
    cutStartTimer(timerGPU);
    prescanArray(d_odata, d_idata, num_elements);
    
    cutStopTimer(timerGPU);

    deallocBlockSums();    

    // copy result from device to host
    cutilSafeCall(cudaMemcpy( output, d_odata, sizeof(int) * num_elements, 
                               cudaMemcpyDeviceToHost));

    // If this is a regression test write the results to a file
    //if( cutCheckCmdLineFlag( argc, (const char**) argv, "regression")) 
    //{
     //   // write file for regression test 
     //   cutWriteFilei( "./data/result.dat", output, num_elements, 0.0);
    //}
   // else 
    //{
        // custom output handling when no regression test running
        // in this case check if the result is equivalent to the expected soluion
        unsigned int result_regtest = cutComparei( reference, output, num_elements);
        printf( "Test %s\n", (1 == result_regtest) ? "PASSED" : "FAILED");
        printf( "Average GPU execution time: %f ms\n", cutGetTimerValue(timerGPU));
      //   printf( "Average GPU execution time: %f ms\n", cutGetTimerValue(timerGPU) / num_test_iterations);
      //  printf( "CPU execution time:         %f ms\n", cutGetTimerValue(timerCPU) / num_test_iterations);
   // }

    printf("\nCheck out the CUDA Data Parallel Primitives Library for more on scan.\n");
    printf("http://www.gpgpu.org/developer/cudpp\n");
    //memcpy (output, h_data, num_elements);
    // cleanup memory
    cutDeleteTimer(timerCPU);
    cutDeleteTimer(timerGPU);
    //free( h_data);
    free( reference);
    cudaFree( d_odata);
    cudaFree( d_idata);
//    cudaThreadExit();
}
