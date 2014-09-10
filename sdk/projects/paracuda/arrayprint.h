#ifndef ARRAYPRINT_H
#define ARRAYPRINT_H

#include "stdio.h"

#define PRINT_HOST(NAME) print_array(#NAME, NAME, length)

#define PRINT_DEVICE(NAME, TYPE) ({\
  PARACUDA_##TYPE##_struct* foo_992_avaq1 = PARACUDA_##TYPE##_allocate_host(length);\
  PARACUDA_##TYPE##_copy_device_host(foo_992_avaq1, NAME, length);\
  print_array(#NAME, foo_992_avaq1, length);\
  PARACUDA_##TYPE##_free_host(foo_992_avaq1);\
})\

void print_array(char* name, int* array, int length)
{
  printf("%s: ", name);
  for(int i = 0; i < length; i++) printf("%d ", array[i]);
  printf("\n");
}

#endif
