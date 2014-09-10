#include "arrayprint.h"

extern void scan(int * output, int* input, int num_elements);

void splitNvidia(int* array, int* flags, int length)
{ 
  int* negatedFlags = (int*) malloc(length * sizeof(int));
  int* backupFlags  = (int*) malloc(length * sizeof(int));
  int* posDown      = (int*) malloc(length * sizeof(int));
  int* posUp        = (int*) malloc(length * sizeof(int));
  int* positions    = (int*) malloc(length * sizeof(int));
  
  negate_map(negatedFlags, flags, length, PARACUDA_MAX_THREADS);
  int computed_sum = negatedFlags[length - 1];
  
  scan(posDown, negatedFlags, length);
  
  computed_sum += posDown[length - 1];
  flags[0] += computed_sum;   
  scan(posUp, flags, length);
    
  flags[0] -= computed_sum;
  posUp[0] += computed_sum;

  split_vector_t input;
  input.flags = flags;
  input.left = posDown;
  input.right = posUp;

  split_map(positions, &input, length, PARACUDA_MAX_THREADS);
  int_permute(array, array, positions, length, PARACUDA_MAX_THREADS);

  free(negatedFlags);
  free(posDown);
  free(posUp);
  free(positions);
}
