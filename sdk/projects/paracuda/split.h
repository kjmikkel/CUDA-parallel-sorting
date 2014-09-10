#include "arrayprint.h"
#include "nvidia_scan.h"

#ifdef USE_NVIDIA_SCAN
#define SCAN(ARRAY) nvidia_scan(ARRAY, ARRAY, length);
#else
#define SCAN(ARRAY) PARACUDA_plus_scan_run(0, ARRAY, length, PARACUDA_MAX_THREADS);
#endif

void split(int* array, int* flags, int length)
{  
  int* posDown      = PARACUDA_int_allocate_device(length);
  int* posUp        = PARACUDA_int_allocate_device(length);
  int* positions    = PARACUDA_int_allocate_device(length);
  pair_vector_t* pair = PARACUDA_pair_t_shallow_allocate_device();
  split_vector_t* input = PARACUDA_split_t_shallow_allocate_device();

  int before;
  int computed_sum;
  PARACUDA_negate_map_run(posDown, flags, length, PARACUDA_MAX_THREADS);
  PARACUDA_int_peek(&before, posDown, length - 1);
  SCAN(posDown);
  PARACUDA_int_peek(&computed_sum, posDown, length - 1);
  computed_sum += before;
  PARACUDA_int_copy_run(positions, computed_sum, length, PARACUDA_MAX_THREADS);

  PARACUDA_int_copy_device_device(posUp, flags, length);
  SCAN(posUp);

  pair_vector_t host_pair;
  host_pair.x = positions;
  host_pair.y = posUp;
  PARACUDA_pair_t_shallow_copy_host_device(pair, &host_pair);
  PARACUDA_map_add_run(posUp, pair, length, PARACUDA_MAX_THREADS);

  split_vector_t host_input;
  host_input.flags = flags;
  host_input.left = posDown; 
  host_input.right = posUp;
  PARACUDA_split_t_shallow_copy_host_device(input, &host_input);
  PARACUDA_split_map_run(positions, input, length, PARACUDA_MAX_THREADS);
  
  PARACUDA_int_permute_run(posDown, array, positions, length, PARACUDA_MAX_THREADS);
  PARACUDA_int_copy_device_device(array, posDown, length);

  PARACUDA_pair_t_shallow_free_device(pair);
  PARACUDA_split_t_shallow_free_device(input);
  PARACUDA_int_free_device(posDown);
  PARACUDA_int_free_device(posUp);
  PARACUDA_int_free_device(positions);
}
