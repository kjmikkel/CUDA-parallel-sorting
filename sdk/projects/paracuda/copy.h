
#include "arrayprint.h"

int* copy(int* array, int num, int length)			
{ 
  // Clear the array
  memset(array, 0, length * sizeof(int));
  // Scan 
  array[0] = num;
  plus_scan(0, array, array, length, PARACUDA_MAX_THREADS); 
  array[0] = num;
  return array;
}

