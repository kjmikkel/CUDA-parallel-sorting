#ifndef PARACUDA_PERMUTE_H
#define PARACUDA_PERMUTE_H

#include <cutil_inline.h>

/* Defines the parallel algorithm. */
#define PARACUDA_PERMUTE(NAME, TYPE) \
/* The part that is run in parralel. Output and input should be on the device. */ \
/* You don't normally need to call this directly. */ \
__global__ \
 void PARACUDA_##NAME##_kernel(PARACUDA_##TYPE##_struct* output, PARACUDA_##TYPE##_struct* input, int* positions, size_t length) \
{ \
    size_t offset = blockIdx.x * blockDim.x + threadIdx.x; \
    TYPE temp; \
    if (length <= offset) return;	\
    PARACUDA_##TYPE##_from_vector(&temp, input, offset); \
    PARACUDA_##TYPE##_to_vector(output, &temp, positions[offset]); \
} \
/* Prepares the kernel and calls it. Output and input should be on the device. */ \
__host__ \
void PARACUDA_##NAME##_run(PARACUDA_##TYPE##_struct* output, PARACUDA_##TYPE##_struct* input, int* positions, size_t length, size_t max_threads) \
{ \
    int numBlocks = length / max_threads;\
    if (numBlocks == 0) numBlocks = 1;\
    if (length < max_threads) max_threads = length;\
    int numThreadsPerBlock = max_threads;\
    dim3 dim(numThreadsPerBlock);\
    PARACUDA_##NAME##_kernel<<<numBlocks, dim>>>(output, input, positions, length); \
    cutilCheckMsg("Permute"); \
} \
\
/* Executes the algorithm and returns the output. */ \
/* It is OK to have output == input. */ \
/* If output == NULL, memory will be allocated for it. */ \
__host__ \
PARACUDA_##TYPE##_struct* NAME(PARACUDA_##TYPE##_struct* output, PARACUDA_##TYPE##_struct* input, int* positions, size_t length, size_t max_threads) \
{ \
    PARACUDA_ASSERT_POWER_OF_TWO(length); \
    if(output == 0) output = PARACUDA_##TYPE##_allocate_host(length); \
    \
    PARACUDA_##TYPE##_struct* device_output = PARACUDA_##TYPE##_allocate_device(length); \
    PARACUDA_##TYPE##_struct* device_input = PARACUDA_##TYPE##_allocate_device(length); \
    int* device_positions = PARACUDA_int_allocate_device(length); \
    \
    PARACUDA_##TYPE##_copy_host_device(device_input, input, length); \
    PARACUDA_int_copy_host_device(device_positions, positions, length); \
    \
    PARACUDA_##NAME##_run(device_output, device_input, device_positions, length, max_threads); \
    PARACUDA_##TYPE##_copy_device_host(output, device_output, length); \
    PARACUDA_##TYPE##_free_device(device_output);\
    PARACUDA_##TYPE##_free_device(device_input);\
    PARACUDA_int_free_device(device_positions); \
    return output; \
} \


#endif

