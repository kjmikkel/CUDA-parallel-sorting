#import "arrayprint.h"

void radix(int* array, int length, int max_threads)
{ 
  int* t_numbers = PARACUDA_int_allocate_device(length);
  int* t_flags = PARACUDA_int_allocate_device(length);
  bitwise_vector_t* in = PARACUDA_bitwise_t_shallow_allocate_device();
  for(int i = 0; i < 32; ++i) {
    int num = (1 << i);
    PARACUDA_int_copy_run(t_numbers, num, length, max_threads);
    bitwise_vector_t input;
    input.number = t_numbers;
    input.integer = array;
    PARACUDA_bitwise_t_shallow_copy_host_device(in, &input);
    PARACUDA_int_bitwise_map_run(t_flags, in, length, max_threads); 
    split(array, t_flags, length);
  }
  PARACUDA_bitwise_t_shallow_free_device(in);
  PARACUDA_int_free_device(t_numbers);
  PARACUDA_int_free_device(t_flags);
}
