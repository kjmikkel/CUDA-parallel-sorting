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
upSweep(element_t* array, bool* flags, bool* intermediate, int length, int max_threads) {
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
        upSweepKernel<element_t><<<grid, threads>>>(array, flags, intermediate, d, job_count);
    }
}

template<typename element_t> __host__ void
downSweep(element_t* array, bool* flags, bool* intermediate, int length, int max_threads) {
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
       // downSweepKernel<element_t><<<grid, threads>>>(array, flags, intermediate, d, job_count);
    }
}

#endif

