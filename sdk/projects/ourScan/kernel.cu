#ifndef _KERNEL_H_
#define _KERNEL_H_

#include <stdio.h>

template<typename element_t> __global__ void
evenKernel(element_t* array, element_t* flags, const unsigned int jobs)
{
    const unsigned int offset = threadIdx.x * jobs;
    for(unsigned int job = 0; job < jobs; ++job)
    {
        const unsigned int k = offset + job;
        flags[k] = array[k] % 2 == 0;
    }

    __syncthreads();
}

template<typename element_t> __global__ void
moveKernel(element_t* array, element_t* flags, element_t* positions, element_t* result, const unsigned int jobs)
{
    const unsigned int offset = threadIdx.x * jobs;
    for(unsigned int job = 0; job < jobs; ++job)
    {
        const unsigned int k = offset + job;

        if(flags[k]) {
            result[positions[k]] = array[k];
        }
    }

    __syncthreads();
}

template<typename element_t> __global__ void
upSweepKernel(element_t* array, const unsigned int d, const unsigned int jobs)
{
    const unsigned int offset = threadIdx.x * jobs;       
    const unsigned int step_child = 1 << (d + 1); 
    const unsigned int step_self = 1 << d; 

    for(unsigned int job = 0; job < jobs; ++job)
    {
        const unsigned int k = (offset + job) * step_child;

        const unsigned int self = k + step_self - 1;
        const unsigned int child = k + step_child - 1;

        array[child] += array[self];
    }

    __syncthreads();
}

template<typename element_t> __global__ void
downSweepKernel(element_t* array, const unsigned int d, const unsigned int jobs)
{
    const unsigned int offset = threadIdx.x * jobs;
    const unsigned int step_child = 1 << (d + 1); 
    const unsigned int step_self = 1 << d; 
    
    for(unsigned int job = 0; job < jobs; ++job)
    {
        const unsigned int k = (offset + job) * step_child;

        const unsigned int self = k + step_self - 1;
        const unsigned int child = k + step_child - 1;

        const element_t temp = array[self];
        array[self] = array[child];
        array[child] += temp;
    }

    __syncthreads();
}

#endif
