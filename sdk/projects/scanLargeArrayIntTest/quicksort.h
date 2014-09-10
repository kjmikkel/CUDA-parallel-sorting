
#include "arrayprint.h"
#include "segmentedCopy.h"
#include "segmentedSplit.h"

bool is_sorted(int* array, int* flags, int length, int max_threads) {
  int* tempArray        = (int*) malloc(length * sizeof(int)); 
  int* numbers          = (int*) malloc(length * sizeof(int)); 
  int* internalNumbers  = (int*) malloc(length * sizeof(int));
  pair_vector_t* numberCompare = PARACUDA_pair_t_allocate_host(length);
  int* tempFlags        = (int*) malloc(length * sizeof(int));
  tempArray = copy(tempArray, 1, length);
  int* testArray = (int*) array + 1;
  
  int_transfer_map(tempArray, array, length, max_threads); 
  int_transfer_map(numbers, testArray, length - 1, max_threads);   
  
  numberCompare->x = numbers;
  numberCompare->y = tempArray;
  compare_map(internalNumbers, numberCompare, length, max_threads);

  internalNumbers[length -1] = 1;
  PRINT(internalNumbers);
  //  int_transfer_map(tempFlags, flags, length, max_threads);
  
  //int_transfer_map(internalNumbers, tempFlags, length, max_threads);
  PRINT(internalNumbers);
  internalNumbers = and_distribute(internalNumbers, length);
  bool value = internalNumbers[length - 1];
  PRINT(internalNumbers);
  printf("sorted?: %i",value);
  free(tempArray);
  free(numbers);
  free(internalNumbers);
  free(numberCompare);   
  return value;
}

void quicksort(int* array, int length, int max_threads)
{ 
  // The flags we are going to use again and again
  int* flags            = (int*) calloc(length, sizeof(int));
  int* pivot            = (int*) calloc(length, sizeof(int));
  int* pivotFlags       = (int*) calloc(length, sizeof(int));
  int* numbers          = (int*) malloc(length * sizeof(int));
  pair_vector_t* numberCompare = PARACUDA_pair_t_allocate_host(length);
  
  flags[0] = 1; 
  bool sorted = is_sorted(array, flags, length, max_threads);
  
  while(!sorted) {
  int_transfer_map(numbers, array, length, max_threads);   
  int_transfer_map(pivot, array, length, max_threads);   
  
  segmentedCopy(pivot, flags, length);  
  //compare value with pivot
  numberCompare->x = numbers;
  numberCompare->y = pivot;
  
  compare_map(pivotFlags, numberCompare, length, max_threads);  
  segmentedSplit(array, pivotFlags, flags, length);
  int* adJustedArray        = (int*) malloc(length * sizeof(int));
  int* newIndex             = (int*) malloc(length * sizeof(int));
  int* adjustedArrayPointer = (int*) array - 1; 
  int_transfer_map(adJustedArray, adjustedArrayPointer, length, max_threads); 
  adJustedArray[0] = INT_MIN; 
 
  split_vector_t* compareStructure = PARACUDA_split_t_allocate_host(length);

  compareStructure->flags = pivot;
  compareStructure->left = array;
  compareStructure->right = adJustedArray;
  
  //PRINT(pivot);
  //PRINT(adJustedArray);
  
  
  
  find_new_array_map(newIndex, compareStructure, length, max_threads);
  numberCompare->x = flags;
  numberCompare->y = newIndex;
  or_map(flags, numberCompare, length, max_threads);
  
  
  //  PRINT(newIndex);
  //PRINT(flags);
   PRINT(array); 

  sorted = is_sorted(array, flags, length, max_threads);
  //<sorted = 1;
  free(adJustedArray);
  free(newIndex);
  
  free(compareStructure); 
 }

  free(flags);
  free(pivot);
  free(pivotFlags);
  // return array;   
}


