#ifndef PARACUDA_COPY_H
#define PARACUDA_COPY_H

#include <cutil_inline.h>

/* Defines the parallel algorithm. */
#define PARACUDA_COPY(NAME, TYPE) \
/* The part that is run in parralel. Output and input should be on the device. */ \
/* You don't normally need to call this directly. */ \
__global__ \
 void PARACUDA_##NAME##_kernel(PARACUDA_##TYPE##_struct* output, TYPE in, size_t length) \
{ \
    size_t offset = blockIdx.x * blockDim.x + threadIdx.x; \
    if (length <= offset) return;	\
    PARACUDA_##TYPE##_to_vector(output, &in, offset);\
} \
/* Prepares the kernel and calls it. Output and input should be on the device. */ \
__host__ \
void PARACUDA_##NAME##_run(PARACUDA_##TYPE##_struct* output, TYPE in, size_t length, size_t max_threads) \
{ \
    int numBlocks = length / max_threads;\
    if (numBlocks == 0) numBlocks = 1;\
    if (length < max_threads) max_threads = length;\
    int numThreadsPerBlock = max_threads;\
    dim3 dim(numThreadsPerBlock);\
    PARACUDA_##NAME##_kernel<<<numBlocks, dim>>>(output, in, length);	\
    cutilCheckMsg("Copy"); \
    cudaThreadSynchronize();\
} \
\
/* Executes the algorithm and returns the output. */ \
/* It is OK to have output == input. */ \
/* If output == NULL, memory will be allocated for it. */ \
__host__ \
PARACUDA_##TYPE##_struct* NAME(PARACUDA_##TYPE##_struct* output, TYPE in, size_t length, size_t max_threads)\
{\
  if(output == 0) output = PARACUDA_##TYPE##_allocate_host(length);\
  PARACUDA_##TYPE##_struct* device_output = PARACUDA_##TYPE##_allocate_device(length);\
  PARACUDA_##NAME##_run(device_output, in, length, max_threads);\
  PARACUDA_##TYPE##_copy_device_host(output, device_output, length);\
  PARACUDA_##TYPE##_free_device(device_output);\
  return output;\
}\
 
#endif

