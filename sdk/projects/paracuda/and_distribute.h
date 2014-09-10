
#include "arrayprint.h"

int* and_distribute(int* array, int length)			
{ 
  // we calculate the distributation from the array
  int endResult = 1;
  //PRINT(array);
  and_distribute_scan(&endResult, array, array, length, PARACUDA_MAX_THREADS);
  //PRINT(array);
  // I copy the value
  array = copy(array, endResult, length);  
  return array;
}

