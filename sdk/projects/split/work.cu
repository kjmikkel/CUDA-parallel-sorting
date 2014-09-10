#ifndef _WORK_H_
#define _WORK_H_

// Base 2 logarithm (expects a positive integer)
template<typename int_t> __host__ int_t
log2(int_t n) {
  int_t l = 0;
  while(n > 1) {
    n >>= 1;
    l += 1;
  }
  return l;
}


__host__ void
negate(int* refernece, int* copyTo, int length) {
  for(int i = 0; i < length; i++) {
    if (refernece[i] == 1) {
      copyTo[i] = 0;
    } else {
      copyTo[i] = 1;
    }
  }
}

template<typename element_t> __host__ void
doSplit(element_t* array, int* flags, int* posDown,int* posUp,element_t* result, const unsigned int length, const unsigned int max_threads) {
  if(length == 0) return;
  int thread_count = length;
  dim3 grid(1, 1, 1);
  int job_count = 1;
  
  while(thread_count > max_threads)
    {
      thread_count >>= 1;
      job_count <<= 1;
    }
  
  dim3 threads(thread_count, 1, 1);;  
  splitKernel<element_t><<<grid, threads>>>(array, flags, posDown, posUp, result, job_count);
}

template<typename element_t> __host__ void
upSweep(element_t* array, int length, int max_threads) {
  if(length == 0) return;
  dim3 grid(1, 1, 1);
  const int d_stop = log2(length) - 1;
  for(int d = 0; d < d_stop; ++d) 
    {
      int thread_count = (int) floor(((length - 1) / (1 << (d + 1)))) + 1;
      int job_count = 1;
      while(thread_count > max_threads)
	{
	  thread_count >>= 1;
	  job_count <<= 1;
	}
      dim3 threads(thread_count, 1, 1);
      upSweepKernel<element_t><<<grid, threads>>>(array, d, job_count);
    }
}

template<typename element_t> __host__ void
downSweep(element_t* array, int length, int max_threads) {
    if(length == 0) return;
    array[length - 1] = 0;
    dim3 grid(1, 1, 1);
    const int d_start = log2(length) - 1;

    for(int d = d_start; d >= 0; --d) 
      {
        int thread_count = (int) floor(((length - 1) / (1 << (d + 1)))) + 1;
        int job_count = 1;
        while(thread_count > max_threads)
	  {
	    thread_count >>= 1;
	    job_count <<= 1;
	  }
        dim3 threads(thread_count, 1, 1);
        downSweepKernel<element_t><<<grid, threads>>>(array, d, job_count);
      }
}

template<typename element_t> __host__ element_t
scan(element_t* d_array, int length, int max_threads) {
  element_t last;
  cutilSafeCall(cudaMemcpy(&last, d_array + length - 1, sizeof(element_t), cudaMemcpyDeviceToHost));
  upSweep<element_t>(d_array, length, max_threads);
  downSweep<element_t>(d_array, length, max_threads);
  element_t sum;
  cutilSafeCall(cudaMemcpy(&sum, d_array + length - 1, sizeof(element_t), cudaMemcpyDeviceToHost));
  return sum + last;
}

template<typename element_t> __host__ unsigned int
split(element_t* array, element_t* result, int* flags, int length, int max_threads) {
  
  element_t* d_array;
  element_t* d_notFlags;
  element_t* d_posDown;
  element_t* d_posUp;
  cutilSafeCall(cudaMalloc((void**) &d_array, length * sizeof(element_t)));
  cutilSafeCall(cudaMemcpy(d_array, array, length * sizeof(element_t), cudaMemcpyHostToDevice));
  cutilSafeCall(cudaMalloc((void**) &d_notFlags, length * sizeof(element_t)));
  cutilSafeCall(cudaMalloc((void**) &d_posDown, length * sizeof(element_t)));
  cutilSafeCall(cudaMalloc((void**) &d_posUp, length * sizeof(element_t)));
    
  
  // Setting the values for posUp 
  cutilSafeCall(cudaMemcpy(d_posUp, flags, length * sizeof(element_t), cudaMemcpyDeviceToDevice));  
  
  /*
    Setting up the down positions are a bit more dificult - first we need to negate the flags and then
    copy the values over
    */
  negate(flags, d_notFlags, length);
  cutilSafeCall(cudaMemcpy(d_posDown, d_notFlags, length * sizeof(element_t), cudaMemcpyDeviceToDevice));
  
  // we then need the count from the scan on the positionsDown, and add that to the value of the first element in the up array
  int count = scan(d_posDown, length, max_threads);   
  
  // We need to inset the count at the first address, calculate the scan, and then insert it again.
  // the reason for the second insert is that exlusive scan always sets the zeroeth address to 0 
  d_posUp[0] += count;
  scan(d_posUp, length, max_threads);
  d_posUp[0] += count; 
  // cutilSafeCall(cudaMemcpy(array, d_posUp, length * sizeof(element_t), cudaMemcpyDeviceToHost));
  //return 0;
  // Now we can make the main call
  
  element_t* d_result;
  cutilSafeCall(cudaMalloc((void**) &d_result, length * sizeof(element_t)));
  
  doSplit(d_array, flags, d_posDown, d_posUp, d_result, length, max_threads);
  
  //result = (element_t*) malloc(count * sizeof(element_t));
  
  cutilSafeCall(cudaMemcpy(result, d_result, length * sizeof(element_t), cudaMemcpyDeviceToHost));
  
  cutilSafeCall(cudaFree(d_array));
  cutilSafeCall(cudaFree(d_notFlags));
  cutilSafeCall(cudaFree(d_posDown));
  cutilSafeCall(cudaFree(d_posUp));
  cutilSafeCall(cudaFree(d_result));
  return count;
}

#endif
