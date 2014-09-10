
#include "arrayprint.h"

void radix(int* array, int length, int max_threads)
{ 
  // The flags we are going to use again and again
  int* flags            = (int*) malloc(length * sizeof(int));
  int* numbers          = (int*) malloc(length * sizeof(int)); 
  int* internalNumbers  = (int*) malloc(length * sizeof(int));
  for(int i = 0; i < 32; ++i) {
    // The datastructure we are going to use to calculate the flags
    bitwise_vector_t input;
    int num = (1 << i);
    // We copy the numbers
    copy(numbers, num, length);
    input.number = numbers;
    // We set the numbers in order to calculate the flags 
    int_transfer_map(internalNumbers, array, length, max_threads);   
    input.integer = internalNumbers;
    int_bitwise_map(flags, &input, length, max_threads); 
    split(array, flags, length);
  }

  free(flags);
  free(numbers);
  free(internalNumbers); 
}
