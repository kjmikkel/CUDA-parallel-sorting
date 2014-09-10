#include "arrayprint.h"

void segmentedSplit(int* array, int* flags, int*arrayFlags, int length)
{ 
  PRINT(flags);
  PRINT(arrayFlags);
  int* negatedFlags = (int*) malloc(length * sizeof(int));
  int* posDown = (int*) malloc(length * sizeof(int));
  int* posUp = (int*) malloc(length * sizeof(int));
  int* positions = (int*) malloc(length * sizeof(int));
  
  // I make a array and fills it with 1's to calculate the offsets
  int* offsets = (int*) malloc(length * sizeof(int));
  copy(offsets, 1, length);
  plus_scan(0, offsets, offsets, length, 0, PARACUDA_MAX_THREADS);
  segmentedCopy(offsets, arrayFlags, length);

  pair_vector_t* data = PARACUDA_pair_t_allocate_host(length);

  // To make sure we can do the segmentedScan right, we must retain some information in a array.
  // The length of the array is the amount of subarrays in the array - which we calculate here,
  int arrayLength;
  int* tempArray = (int*) malloc(length * sizeof(int));
  plus_scan(&arrayLength, tempArray, arrayFlags, length, PARACUDA_MAX_THREADS);
  free(tempArray);
  int* segmentedInformation
  printf("arrayLength: %i\n", arrayLength);

  negate_map(negatedFlags, flags, length, PARACUDA_MAX_THREADS);
  int computed_sum;
  
  segmented_plus_scan(&computed_sum, posDown, negatedFlags, arrayFlags, length, PARACUDA_MAX_THREADS);
  //  flags[0] += computed_sum;
  segmented_plus_scan(0, posUp, flags, arrayFlags, length, PARACUDA_MAX_THREADS);
  //flags[0] -= computed_sum;
  //posUp[0] += computed_sum;

  PRINT(posDown);
  PRINT(posUp);
  PRINT(offsets);

  data->x = posDown;
  data->y = offsets;

  map_add(posDown, data, length, PARACUDA_MAX_THREADS);

  data->x = posUp;
  map_add(posUp, data, length, PARACUDA_MAX_THREADS); 

  PRINT(flags);
  split_vector_t input;
  input.flags = flags;
  input.left = posDown;
  input.right = posUp;
 
  split_map(positions, &input, length, PARACUDA_MAX_THREADS);
  PRINT(flags);
  PRINT(positions);
  int_permute(array, array, positions, length, PARACUDA_MAX_THREADS);

  free(data);
  free(negatedFlags);
  free(posDown);
  free(posUp);
  free(positions);
}
