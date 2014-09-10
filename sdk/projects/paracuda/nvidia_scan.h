#ifndef NVIDIA_SCAN
#define NVIDIA_SCAN

extern void preallocBlockSums(unsigned int maxNumElements);
extern void deallocBlockSums();
extern void prescanArray(int *outArray, int *inArray, int numElements);

void nvidia_scan(int* output, int* input, size_t length) {
    preallocBlockSums(length);
    prescanArray(output, input, length);
    deallocBlockSums();    
}

#endif
