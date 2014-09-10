#include <stdio.h>
#include <gold.h>

void
compute(int* array, bool* flags, const unsigned int len) 
{
    int sum = 0;
    for(unsigned int i = 0; i < len; ++i) 
    {
        if(flags[i]) sum = 0;
        int current = array[i];
        array[i] = sum;
        sum += current;
    }
}

