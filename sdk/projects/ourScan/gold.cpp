#include <stdio.h>
#include <gold.h>

void
compute2(int* array, const unsigned int len) 
{
    int sum = 0;
    for(unsigned int i = 0; i < len; ++i) 
    {
        int current = array[i];
        array[i] = sum;
        sum += current;
    }
}
