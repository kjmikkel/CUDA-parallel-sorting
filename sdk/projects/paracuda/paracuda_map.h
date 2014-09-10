#ifndef PARACUDA_MAP_H
#define PARACUDA_MAP_H

#include <cutil_inline.h>

/* Defines the parallel algorithm. */
#define PARACUDA_MAP(NAME, OPERATOR, OUTPUT, INPUT) \
/* The part that is run in parralel. Output and input should be on the device. */ \
/* You don't normally need to call this directly. */ \
__global__ \
 void PARACUDA_##NAME##_kernel(PARACUDA_##OUTPUT##_struct* output, PARACUDA_##INPUT##_struct* input, size_t length) \
{ \
  /*__shared__ PARACUDA_##INPUT##_struct s_data[512];*/ \
    size_t nomialOffset = blockIdx.x * blockDim.x + threadIdx.x; \
    OUTPUT out; \
    INPUT in; \
    if (length <= nomialOffset) return;	\
    PARACUDA_##INPUT##_from_vector(&in, input, nomialOffset);\
    PARACUDA_##OPERATOR##_operator(&out, &in);\
    PARACUDA_##OUTPUT##_to_vector(output, &out, nomialOffset);\
} \
/* Prepares the kernel and calls it. Output and input should be on the device. */ \
__host__ \
void PARACUDA_##NAME##_run(PARACUDA_##OUTPUT##_struct* output, PARACUDA_##INPUT##_struct* input, size_t length, size_t max_threads) \
{ \
    int numBlocks = length / max_threads;\
    if (numBlocks == 0) numBlocks = 1;\
    if (length < max_threads) max_threads = length;\
    int numThreadsPerBlock = max_threads;\
    dim3 dim(numThreadsPerBlock);\
    PARACUDA_##NAME##_kernel<<<numBlocks, dim>>>(output, input, length);\
    cutilCheckMsg("Map"); \
    cudaThreadSynchronize();\
} \
\
/* Executes the algorithm and returns the output. */ \
/* It is OK to have output == input. */ \
/* If output == NULL, memory will be allocated for it. */ \
__host__ \
PARACUDA_##OUTPUT##_struct* NAME(PARACUDA_##OUTPUT##_struct* output, PARACUDA_##INPUT##_struct* input, size_t length, size_t max_threads) \
{ \
  /*PARACUDA_ASSERT_POWER_OF_TWO(length);*/\
  if(output == 0) output = PARACUDA_##OUTPUT##_allocate_host(length);\
  PARACUDA_##INPUT##_struct* device_input = PARACUDA_##INPUT##_allocate_device(length); \
  PARACUDA_##INPUT##_copy_host_device(device_input, input, length); \
  if((void*) input == (void*) output)\
  {\
    PARACUDA_##NAME##_run((PARACUDA_##OUTPUT##_struct*) device_input, device_input, length, max_threads); \
    PARACUDA_##OUTPUT##_copy_device_host(output, (PARACUDA_##OUTPUT##_struct*) device_input, length); \
  }\
  else\
  {\
    PARACUDA_##OUTPUT##_struct* device_output = PARACUDA_##OUTPUT##_allocate_device(length); \
    PARACUDA_##NAME##_run(device_output, device_input, length, max_threads); \
    PARACUDA_##OUTPUT##_copy_device_host(output, device_output, length); \
    PARACUDA_##OUTPUT##_free_device(device_output);\
  }\
  PARACUDA_##INPUT##_free_device(device_input); \
  cutilCheckMsg("Map"); \
  return output; \
} \
 
#endif

