#include <cstdio>

#include "paracuda.h"
#include "kernel.cu"
#include "splitNvidia.h"
#include "arrayprint.h"
#include "copy.h"
#include "radixNvidia.h"
#include "arrayprint.h"

void testRadix(int* data, int* gold, int length, unsigned int timer)
{
    for(int i = 0; i < length; i++) {
        data[i] = length - i - 1;
    }
    
    //radixNvidia(gold, length, PARACUDA_MAX_THREADS); // warmup
    
    unsigned int timer2 = 0;
    cutilCheckError(cutCreateTimer(&timer2));
    cutilCheckError(cutStartTimer(timer2));
    for(int i = 0; i < length; i++) {
        gold[i] = i;
    }
    cutilCheckError(cutStopTimer(timer2));
    //printf("iterative solution took: %f ms\n", cutGetTimerValue(timer2));
    cutilCheckError(cutStartTimer(timer));
    
    radixNvidia(data, length, PARACUDA_MAX_THREADS);
    
    cutilCheckError(cutStopTimer(timer));
}

void runTestRadix(int argc, char** argv) {
    PARACUDA_INITIALIZE_CUDA(argc, argv);
    unsigned int timer = 0;
    cutilCheckError(cutCreateTimer(&timer));
    int length = 1024 * 1024;
    int* data = PARACUDA_int_allocate_host(length);
    int* gold = PARACUDA_int_allocate_host(length);
    testRadix(data, gold, length, timer);
    printf("\n");
    printf("Data:\t\tGold:\n");
    bool same = true;
    for(int i = 0; i < length; i++) {
      if(data[i] == gold[i]) {
	if (i < 10)
	  printf("%d\t==\t%d\n", data[i], gold[i]);
        } else {
        if (i < 10) 
	  printf("%d\t!=\t%d\t!!!\n", data[i], gold[i]);
	same = false;
      }
    }
    if(same) {
        printf("\nThey are the SAME! Processing time: %f ms, average: %f ms.\n", 
            cutGetTimerValue(timer), cutGetAverageTimerValue(timer));
    } else {
        printf("\nThey are DIFFERENT! - and took %f ms\n", cutGetTimerValue(timer));
    }
    cutilCheckError(cutDeleteTimer(timer));
    PARACUDA_EXIT_KEYPRESS(argc, argv);
}
