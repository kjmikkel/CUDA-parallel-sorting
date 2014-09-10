
#include "arrayprint.h"

void segmentedEnumerate(int* arrayFlags, int* pivotFlags, int* posDown, int*posUp, int length, int max_threads)			
{ 
  int* negatedFlags = (int*) malloc(length * sizeof(int));
  negate_map(negatedFlags, pivotFlags, length, PARACUDA_MAX_THREADS);
  
  pair_vector_t* data = PARACUDA_pair_t_allocate_host(length);
  split_vector_t* split = PARACUDA_split_t_allocate_host(length);
  // To make sure we can do the segmentedScan right, we must retain some information in a array.

  // First we find the offsets, which contains the offset for each logical vector from the start
  int* offsets = (int*) malloc(length * sizeof(int));
  offsets = copy(offsets, 1, length);
  plus_scan(0, offsets, offsets, length, PARACUDA_MAX_THREADS);
  
  data->x = arrayFlags;
  data->y = offsets;  

  int_copy_pivot_map(offsets, data, length, max_threads);
  // Now that we know the offset of each array, we need to copy it across the entire array
  segmentedCopy(offsets, arrayFlags, length);
  
  // We need to retain some information in order to be able to get the correctly segmented values for posUp
  int* offSetArray = (int*) calloc(length, sizeof(int));
  // posDown
  segmented_plus_scan(offSetArray, posDown, negatedFlags, arrayFlags, offsets, length, PARACUDA_MAX_THREADS);    
  data->x = offSetArray;
  data->y = pivotFlags;   
  PRINT(offSetArray);
  PRINT(pivotFlags);
  map_add(pivotFlags, data, length, max_threads);
  //posUp
  
  split->flags = arrayFlags;
  split->left = posUp;
  split->right = offSetArray;
  int_add_pivot_map(posUp, split, length, max_threads);
  segmented_plus_scan(posUp, pivotFlags, arrayFlags, length, PARACUDA_MAX_THREADS);
  PRINT(posUp);

 
  PRINT(posUp);
       //split->right = offsets;
       //int_add_pivot_map(posUp, split, length, max_threads); 
  data->x = pivotFlags;
  data->y = offSetArray;
  // I restore the pivot Flags
  map_minus(pivotFlags, data, length, max_threads);   
  
  printf("\nPermute:\n");
  PRINT(arrayFlags);
  printf("value: %i\n", offSetArray[arrayFlags[0]]);
  PRINT(posUp);
  PRINT(posDown);
  PRINT(offsets);
  PRINT(offSetArray);
 
  // I add the offset array to both posUp and posDown - so that they now point to the absolute index in the array
  // not just the offset from their logical array    
  data->x = offsets;
  data->y = posUp;
  map_add(posUp, data, length, max_threads);
  
  data->x = offsets;
  data->y = posDown;
  map_add(posDown, data, length, max_threads);
  
  free(offsets);
  free(data);
  free(negatedFlags);
  free(offSetArray);
}

