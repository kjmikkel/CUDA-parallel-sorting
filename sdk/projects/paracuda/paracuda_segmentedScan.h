
#ifndef PARACUDA_SEGMENTEDSCAN_H
#define PARACUDA_SEGMENTEDSCAN_H
#include "arrayprint.h"

#include <cutil_inline.h>

/* Defines the parallel algorithm. */
#define PARACUDA_SEGMENTEDSCAN(NAME, OPERATOR, NEUTRAL, TYPE)		\
/* The part that is run in parralel. Output and input should be on the device. */ \
/* You don't normally need to call this directly. */ \
__global__ \
 void PARACUDA_##NAME##_upsweep_kernel(PARACUDA_##TYPE##_struct* array, int* flags, int* iFlags, size_t length, size_t d_stop, size_t max_threads) \
{ \
 for(int d = 0; d <= d_stop; d++) {\
    int thread_count = (int) floorf(((length - 1) / (1 << (d + 1)))) + 1; \
    int job_count = 1;\
    while(thread_count > max_threads)\
      {\
	thread_count >>= 1;\
	job_count <<= 1;\
      }\
    \
    size_t offset = threadIdx.x * job_count; \
    size_t step_child = 1 << (d + 1); \
    size_t step_self = 1 << d; \
    for(size_t job = 0; job < job_count; ++job)\
	{\
	  __syncthreads();\
	  size_t k = (offset + job) * step_child;\
          if (k < length) {\
          size_t self = k + step_self - 1;\
	  size_t child = k + step_child - 1;\
	  \
	  TYPE out;\
	  TYPE self_in;\
	  TYPE child_in;\
	  if(!flags[child]) {\
            PARACUDA_##TYPE##_from_vector(&self_in, array, self); \
	    PARACUDA_##TYPE##_from_vector(&child_in, array, child); \
            PARACUDA_##OPERATOR##_operator(&out, &self_in, &child_in);\
	    PARACUDA_##TYPE##_to_vector(array, &out, child);\
	  }\
          flags[child] = flags[self] | flags[child];\
          }\
	}\
    }\
}\
__global__ \
 void PARACUDA_##NAME##_downsweep_kernel(PARACUDA_##TYPE##_struct* sumArray, PARACUDA_##TYPE##_struct* array, int* flags, int* iFlag, int* sumArrayIndex, size_t length, int d_start, int max_threads) \
{\
for(int d = d_start; d >= 0; d--) {\
    __syncthreads();\
    int thread_count = (int) floorf(((length - 1) / (1 << (d + 1)))) + 1; \
    int job_count = 1;\
    while(thread_count > max_threads)\
      {\
	thread_count >>= 1;\
	job_count <<= 1;\
      }\
    \
    size_t offset = threadIdx.x * job_count; \
    size_t step_child = 1 << (d + 1); \
    size_t step_self = 1 << d; \
    for(size_t job = 0; job < job_count; ++job)	\
      {\
	__syncthreads();\
        size_t k = (offset + job) * step_child;\
	if(k < length) { \
	  \
	  size_t self = k + step_self - 1;\
	  size_t child = k + step_child - 1;\
	  \
	  TYPE out;\
	  TYPE child_in;\
          TYPE temp; \
          TYPE backup;\
          \
          PARACUDA_##TYPE##_from_vector(&backup, array, self);	\
	  \
          PARACUDA_##TYPE##_from_vector(&child_in, array, child);\
          PARACUDA_##TYPE##_to_vector(array, &child_in, self);	 \
          if (iFlag[self + 1]) {\
            if(sumArray) { \
              PARACUDA_##TYPE##_from_vector(&child_in, array, child);\
              PARACUDA_##OPERATOR##_operator(&out, &backup, &child_in); \
              PARACUDA_##TYPE##_to_vector(sumArray, &out, sumArrayIndex[self]); \
            } \
            PARACUDA_##NEUTRAL##_operator(&temp);\
            PARACUDA_##TYPE##_to_vector(array, &temp, child);\
          } else if (flags[self]) {\
            PARACUDA_##TYPE##_to_vector(array, &backup, child);\
	  } else {\
	    PARACUDA_##TYPE##_from_vector(&child_in, array, child);\
            PARACUDA_##OPERATOR##_operator(&out, &backup, &child_in); \
            PARACUDA_##TYPE##_to_vector(array, &out, child);\
	  }\
          flags[self] = 0; \
	}\
      }\
  }\
} \
 \
/* Prepares the kernel and calls it. Output and input should be on the device. */ \
/* You don't normally need to call this directly. */ \
__host__ \
void PARACUDA_##NAME##_upsweep(PARACUDA_##TYPE##_struct* array, int* flags, int* iFlags, size_t length, size_t max_threads) \
{\
    if(length == 0) return; \
    if(length < max_threads) max_threads = length; \
    dim3 grid(1, 1, 1); \
    dim3 threads(max_threads, 1, 1); \
    int d_stop = PARACUDA_LOG2(length) -1; \
    PARACUDA_##NAME##_upsweep_kernel<<<grid, threads>>>(array, flags, iFlags, length, d_stop, max_threads); \
} \
void PARACUDA_##NAME##_downsweep(PARACUDA_##TYPE##_struct* sumArray, PARACUDA_##TYPE##_struct* array, int* flags, int* iFlag, int* sumArrayIndex, size_t length, size_t max_threads) \
{ \
    if(length == 0) return; \
    /* TODO: Consider moving the neutralization to the kernel to avoid memcopy from host to device */ \
    \
    TYPE out; \
    NEUTRAL(&out);\
    const int d_start = PARACUDA_LOG2(length) - 1;\
    PARACUDA_##TYPE##_to_vector_poke(array, &out, length - 1); \
    if (length < max_threads) max_threads = length; \
    dim3 grid(1, 1, 1); \
    dim3 threads(max_threads, 1, 1);\
    PARACUDA_##NAME##_downsweep_kernel<<<grid, threads>>>(sumArray, array, flags, iFlag, sumArrayIndex, length, d_start, max_threads); \
} \
 \
/* On-device memory assumed. In place. A copy of flags must be passed as flags_copy, and it will be scrambled. */\
__host__ \
void PARACUDA_##NAME##_run(PARACUDA_##TYPE##_struct* sumArray, PARACUDA_##TYPE##_struct* array, int* flags, int* flags_copy, int* sumArrayIndex, size_t length, size_t max_threads) \
{\
  PARACUDA_##NAME##_upsweep(array, flags_copy, flags, length, max_threads);	\
    PARACUDA_##NAME##_downsweep(sumArray, array, flags_copy, flags, sumArrayIndex, length, max_threads); \
    cutilCheckMsg("Segmented scan");\
}\
/* Executes the algorithm and returns the output. */ \
/* It is OK to have output == input. */ \
/* If output == NULL, memory will be allocated for it. */ \
__host__ \
 PARACUDA_##TYPE##_struct* NAME(PARACUDA_##TYPE##_struct* sumArray, PARACUDA_##TYPE##_struct* output, PARACUDA_##TYPE##_struct* input, int* flags, int* sumArrayIndex, size_t length, size_t max_threads) \
{\
    PARACUDA_ASSERT_POWER_OF_TWO(length);\
    if(output == 0) output = PARACUDA_##TYPE##_allocate_host(length); \
    PARACUDA_##TYPE##_struct* device_array = PARACUDA_##TYPE##_allocate_device(length); \
    \
    int arrayLength = (sumArray) ? length : 0;			\
    int* device_flags      = PARACUDA_int_allocate_device(length);\
    int* device_iFlags     = PARACUDA_int_allocate_device(length);\
    int* device_sumArray   = PARACUDA_int_allocate_device(arrayLength);\
    int* device_arrayIndex = PARACUDA_int_allocate_device(length);\
    if(sumArray) {\
      PARACUDA_##TYPE##_copy_host_device(device_sumArray, sumArray, length); \
      PARACUDA_##TYPE##_copy_host_device(device_arrayIndex, sumArrayIndex, length);\
    }\
    \
    PARACUDA_##TYPE##_copy_host_device(device_array, input, length); \
    PARACUDA_int_copy_host_device(device_flags, flags, length); \
    PARACUDA_int_copy_host_device(device_iFlags, flags, length);\
      \
    \
    PARACUDA_##NAME##_upsweep(device_array, device_flags, device_iFlags, length, max_threads); \
    PARACUDA_##NAME##_downsweep(device_sumArray, device_array, device_flags, device_iFlags, device_arrayIndex, length, max_threads);  \
    \
    PARACUDA_##TYPE##_copy_device_host(output, device_array, length); \
    /*PARACUDA_##TYPE##_copy_device_host(flags, device_iFlags, length);*/ \
    \
    PARACUDA_##TYPE##_free_device(device_array); \
    PARACUDA_int_free_device(device_flags); \
    PARACUDA_int_free_device(device_iFlags);\
    if(sumArrayIndex) {\
        TYPE last;\
        TYPE valueToAdd;\
        PARACUDA_##TYPE##_from_vector_host(&last, output, length - 1); \
        PARACUDA_##TYPE##_from_vector_host(&valueToAdd, input, length -1);\
        OPERATOR(&last, &last, &valueToAdd);\
        PARACUDA_##TYPE##_to_vector_host(device_sumArray, &last,sumArrayIndex[length -1]);\
        PARACUDA_##TYPE##_copy_device_host(sumArray, device_sumArray, length); \
      }\
    PARACUDA_##TYPE##_free_device(device_sumArray);\
    PARACUDA_##TYPE##_free_device(device_arrayIndex);\
    cutilCheckMsg("Kernel execution failed");	\
    return output;\
} \
__host__ \
 PARACUDA_##TYPE##_struct* NAME(PARACUDA_##TYPE##_struct* output, PARACUDA_##TYPE##_struct* input, int* flags, size_t length, size_t max_threads) \
{\
  return NAME(0, output, input, flags, 0, length,  max_threads);	\
}\

#endif
