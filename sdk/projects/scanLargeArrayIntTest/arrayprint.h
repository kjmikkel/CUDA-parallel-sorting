#ifndef ARRAYPRINT_H
#define ARRAYPRINT_H

#include "stdio.h"

#define PRINT(NAME) print_array(#NAME, NAME, length)

void print_array(char* name, int* array, int length)
{
  printf("%s: ", name);
  for(int i = 0; i < length; i++) printf("%d ", array[i]);
  printf("\n");
}

#endif
