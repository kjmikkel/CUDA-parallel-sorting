
#include "arrayprint.h"

void segmentedCopy(int* array, int* flags, int length)			
{ 
  int* copyArray = (int*) malloc(length * sizeof(int));
  int_transfer_map(copyArray, array, length, PARACUDA_MAX_THREADS);  
  
  pair_vector_t* data = PARACUDA_pair_t_allocate_host(length);
  data->x = flags;
  data->y = array; 

  int_copy_pivot_map(array, data, length, PARACUDA_MAX_THREADS);
  max_segmented_scan(array, array, flags, length, PARACUDA_MAX_THREADS); 
  split_vector_t* data2 = PARACUDA_split_t_allocate_host(length);
  data2->flags = flags;
  data2->left  = copyArray;
  data2->right = array;
  
  // I make sure that the first elements in each segment also has the pivot element  
  int_insert_pivot_map(array, data2, length, PARACUDA_MAX_THREADS);
  free(data);
  free(data2);
  free(copyArray);
}

