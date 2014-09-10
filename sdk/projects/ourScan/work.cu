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
    upSweep(d_array, length, max_threads);
    downSweep(d_array, length, max_threads);
    element_t sum;
    cutilSafeCall(cudaMemcpy(&sum, d_array + length - 1, sizeof(element_t), cudaMemcpyDeviceToHost));
    return sum + last;
}

template<typename element_t> __host__ void
even(element_t* array, element_t* flags, int length, int max_threads) {
    int thread_count = length;
    int job_count = 1;
    while(thread_count > max_threads)
    {
        thread_count >>= 1;
        job_count <<= 1;
    }
    dim3 grid(1, 1, 1);
    dim3 threads(thread_count, 1, 1);
    evenKernel<element_t><<<grid, threads>>>(array, flags, job_count);
}

template<typename element_t> __host__ void
move(element_t* array, element_t* flags, element_t* positions, element_t* result, int length, int max_threads) {
    int thread_count = length;
    int job_count = 1;
    while(thread_count > max_threads)
    {
        thread_count >>= 1;
        job_count <<= 1;
    }
    dim3 grid(1, 1, 1);
    dim3 threads(thread_count, 1, 1);
    moveKernel<element_t><<<grid, threads>>>(array, flags, positions, result, job_count);
}

template<typename element_t> __host__ unsigned int
packEven(element_t* array, int length, element_t** result, int max_threads) {
    element_t* d_array;
    element_t* d_flags;
    element_t* d_positions;
    cutilSafeCall(cudaMalloc((void**) &d_array, length * sizeof(element_t)));
    cutilSafeCall(cudaMemcpy(d_array, array, length * sizeof(element_t), cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMalloc((void**) &d_flags, length * sizeof(element_t)));
    cutilSafeCall(cudaMalloc((void**) &d_positions, length * sizeof(element_t)));

    even(d_array, d_flags, length, max_threads);

    cutilSafeCall(cudaMemcpy(d_positions, d_flags, length * sizeof(element_t), cudaMemcpyDeviceToDevice));

    int count = scan(d_positions, length, max_threads);

    element_t* d_result;
    cutilSafeCall(cudaMalloc((void**) &d_result, length * sizeof(element_t)));

    move(d_array, d_flags, d_positions, d_result, length, max_threads);
   
    *result = (element_t*) malloc(count * sizeof(element_t));
    cutilSafeCall(cudaMemcpy(*result, d_result, count * sizeof(element_t), cudaMemcpyDeviceToHost));

    cutilSafeCall(cudaFree(d_array));
    cutilSafeCall(cudaFree(d_flags));
    cutilSafeCall(cudaFree(d_positions));
    cutilSafeCall(cudaFree(d_result));
    return count;
}

#endif

