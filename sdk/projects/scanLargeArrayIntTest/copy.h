
#include "arrayprint.h"

int* copy(int* array, int num, int length)			
{ 
  free(array);
  array  = (int*)calloc(length, sizeof(int));
  
  array[0] = num;
  plus_scan(&num, array, array, length, PARACUDA_MAX_THREADS); 
  array[0] = num;
  return array;
}

