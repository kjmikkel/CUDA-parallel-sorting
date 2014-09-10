#include "arrayprint.h"
#include "segmentedEnumerate.h"

void segmentedSplit(int* array, int* flags, int* arrayFlags, int length)
{ 
  int* posDown   = (int*) calloc(length, sizeof(int));
  int* posUp     = (int*) calloc(length, sizeof(int));
  int* positions = (int*) malloc(length * sizeof(int));
  // I make a array and fills it with 1's to calculate the offsets
  int* offsets = (int*) malloc(length * sizeof(int));
  
  offsets = copy(offsets, 1, length);
  
  plus_scan(0, offsets, offsets, length, PARACUDA_MAX_THREADS); 
  segmentedCopy(offsets, arrayFlags, length);
  segmentedEnumerate(arrayFlags, flags, posDown, posUp,  length, PARACUDA_MAX_THREADS);
  
  split_vector_t input;
  input.flags = flags;
  input.left = posDown;
  input.right = posUp;

  //  PRINT(flags);
  PRINT(arrayFlags);
  PRINT(posDown);
  PRINT(posUp);
  split_map(positions, &input, length, PARACUDA_MAX_THREADS);
  PRINT(positions);
  //printf("permute\n");
  int_permute(array, array, positions, length, PARACUDA_MAX_THREADS);
  
  //printf("time to free");
  free(offsets); 
  free(posDown);
  free(posUp);
  //free(printf("done with segmented split\n");
}
