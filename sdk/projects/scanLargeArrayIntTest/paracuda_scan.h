
#ifndef PARACUDA_SCAN_H
#define PARACUDA_SCAN_H

#include <cutil_inline.h>

/* Defines the parallel algorithm. */
#define PARACUDA_SCAN(NAME, OPERATOR, NEUTRAL, TYPE)\
/* The part that is run in parralel. Output and input should be on the device. */ \
/* You don't normally need to call this directly. */ \
__global__ \
 void PARACUDA_##NAME##_upsweep_kernel(PARACUDA_##TYPE##_struct* array, size_t length, size_t d_stop, size_t max_threads) \
{ \
    for(int d = 0; d < d_stop; ++d) \
    { \
      __syncthreads();					\
      int thread_count = (int) floorf((length - 1) / (1 << (d + 1))) + 1; \
      int job_count = 1;\
      while(thread_count > max_threads) \
        { \
            thread_count >>= 1; \
            job_count <<= 1;\
        } \
      size_t offset = threadIdx.x * job_count;\
      size_t step_child = 1 << (d + 1);\
      size_t step_self =  1 << d; \
      for(size_t job = 0; job < job_count; ++job)\
	{\
	  __syncthreads(); \
	  if (threadIdx.x < thread_count) {\
	  size_t k = (offset + job) * step_child;\
    	 \
	  size_t self = k + step_self - 1;\
	  size_t child = k + step_child - 1;\
	  \
	  TYPE out;\
	  TYPE self_in;\
	  TYPE child_in;\
	  PARACUDA_##TYPE##_from_vector(&self_in, array, self); \
	  PARACUDA_##TYPE##_from_vector(&child_in, array, child); \
	  PARACUDA_##OPERATOR##_operator(&out, &self_in, &child_in);\
	  PARACUDA_##TYPE##_to_vector(array, &out, child);\
	  }\
	}\
    }\
}\
__global__ \
 void PARACUDA_##NAME##_downsweep_kernel(PARACUDA_##TYPE##_struct* array, size_t length, int d_start, int max_threads) \
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
	if(threadIdx.x < thread_count) { \
	  size_t k = (offset + job) * step_child;\
	  \
	  size_t self = k + step_self - 1;\
	  size_t child = k + step_child - 1;\
	  \
	  TYPE out;\
	  TYPE self_in;\
	  TYPE child_in;\
	  PARACUDA_##TYPE##_from_vector(&self_in, array, self);	\
          PARACUDA_##TYPE##_from_vector(&child_in, array, child);\
	  PARACUDA_##OPERATOR##_operator(&out, &self_in, &child_in);\
          PARACUDA_##TYPE##_to_vector(array, &child_in, self);\
          PARACUDA_##TYPE##_to_vector(array, &out, child);\
	}\
      }\
  }\
} \
 \
/* Prepares the kernel and calls it. Output and input should be on the device. */ \
/* You don't normally need to call this directly. */ \
__host__ \
void PARACUDA_##NAME##_upsweep(PARACUDA_##TYPE##_struct* array, size_t length, size_t max_threads) \
{ \
    if(length == 0) return; \
    dim3 grid(1, 1, 1); \
    if(length < max_threads) max_threads = length; \
    dim3 threads(max_threads, 1, 1); \
    int d_stop = PARACUDA_LOG2(length) -1; \
    PARACUDA_##NAME##_upsweep_kernel<<<grid, threads>>>(array, length, d_stop, max_threads); \
    } \
void PARACUDA_##NAME##_downsweep(PARACUDA_##TYPE##_struct* array, size_t length, size_t max_threads) \
{ \
    if(length == 0) return; \
    /* TODO: Consider moving the neutralization to the kernel to avoid memcopy from host to device */ \
    TYPE out; \
    NEUTRAL(&out);\
    PARACUDA_##TYPE##_to_vector_poke(array, &out, length - 1); \
    dim3 grid(1, 1, 1); \
    const int d_start = PARACUDA_LOG2(length) - 1;\
    \
    if (length < max_threads) max_threads = length; \
    dim3 threads(max_threads, 1, 1);\
    PARACUDA_##NAME##_downsweep_kernel<<<grid, threads>>>(array, length, d_start, max_threads); \
} \
 \
/* On-device memory assumed. In place. */\
__host__ \
void PARACUDA_##NAME##_run(PARACUDA_##TYPE##_struct* array, size_t length, size_t max_threads) \
{\
    PARACUDA_##NAME##_upsweep(array, length, max_threads); \
    PARACUDA_##NAME##_downsweep(array, length, max_threads); \
    cutilCheckMsg("Scan"); \
}\
/* Executes the algorithm and returns the output. */ \
/* It is OK to have output == input. */ \
/* If output == NULL, memory will be allocated for it. */ \
__host__ \
PARACUDA_##TYPE##_struct* NAME(TYPE* sum, PARACUDA_##TYPE##_struct* output, PARACUDA_##TYPE##_struct* input, size_t length, size_t max_threads) \
{\
    PARACUDA_ASSERT_POWER_OF_TWO(length);\
    if(sum != 0) PARACUDA_##TYPE##_from_vector_host(sum, input, length - 1); \
    if(output == 0) output = PARACUDA_##TYPE##_allocate_host(length); \
    PARACUDA_##TYPE##_struct* device_array = PARACUDA_##TYPE##_allocate_device(length); \
    PARACUDA_##TYPE##_copy_host_device(device_array, input, length); \
    \
    PARACUDA_##NAME##_run(device_array, length, max_threads); \
    \
    PARACUDA_##TYPE##_copy_device_host(output, device_array, length); \
    PARACUDA_##TYPE##_free_device(device_array);		      \
    if(sum != 0) { \
        TYPE last; \
        PARACUDA_##TYPE##_from_vector_host(&last, output, length - 1); \
        OPERATOR(sum, &last, sum);\
    }\
    return output;\
} \

#endif
