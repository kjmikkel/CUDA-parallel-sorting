#import "cpu_radix.h"
#import "cpu_split.h"
#import "cpu_scan.h"
#import "cpu_map.h"
#import <stdlib.h>
#import <stdio.h>
#import <string.h>
#import <math.h>
#include <time.h>



int main(int argc, char** argv)
{
  clock_t start, end;
  double elapsed;
  srand(clock());
  int length = 1024 * 1024;
  int iterations = 10;
  int print = 20;
  int* array = (int*) malloc(length * sizeof(int));
  int* flags = (int*) malloc(length * sizeof(int));
  int* store = (int*) malloc(length * sizeof(int));

  printf("CLOCKS/S: %d\n", CLOCKS_PER_SEC);

  printf("CPU radix: ");
  for(int i = 0; i < length; i++)
  {
    array[i] = rand() % 100;
  }
  start = clock();
  for(int t = 0; t < iterations; t++) cpu_radix(array, store, length);
  end = clock();
  elapsed = (((double) (end - start)) / iterations) / (CLOCKS_PER_SEC / 1000.0);
  int sorted = 1;
  printf("(%d ms) ", (int) elapsed);
  for(int i = 0; i < length; i++)
  {
    if(i < 20) printf("%d ", array[i]);
    sorted &= i == 0 || array[i - 1] <= array[i];
  }
  if(sorted) printf(" SORTED!\n");
  else printf(" GARBAGE!\n");

  printf("CPU split: ");
  for(int i = 0; i < length; i++)
  {
    array[i] = i;
    flags[i] = i % 3 == 0;
  }
  start = clock();
  for(int t = 0; t < iterations; t++) cpu_split(array, flags, store, length);
  end = clock(); 
  elapsed = (((double) (end - start)) / iterations) / (CLOCKS_PER_SEC / 1000.0);
  printf("(%d ms) ", (int) elapsed);
  for(int i = 0; i < length; i++)
  {
    if(i < 20) printf("%d ", array[i]);
  }
  printf("\n");

  printf("CPU scan: ");
  for(int i = 0; i < length; i++)
  {
    array[i] = i;
  }
  start = clock();
  for(int t = 0; t < iterations; t++) cpu_scan(array, length);
  end = clock();
  elapsed = (((double) (end - start)) / iterations) / (CLOCKS_PER_SEC / 1000.0);
  printf("(%d ms) ", (int) elapsed);
  for(int i = 0; i < length && i < 32; i++)
  {
    if(i < 20) printf("%d ", array[i]);
  }
  printf("\n");

  printf("CPU map: ");
  for(int i = 0; i < length; i++)
  {
    array[i] = i;
  }
  start = clock();
  for(int t = 0; t < iterations; t++) cpu_map(array, length);
  end = clock();
  elapsed = (((double) (end - start)) / iterations) / (CLOCKS_PER_SEC / 1000.0);
  printf("(%d ms) ", (int) elapsed);
  for(int i = 0; i < length && i < 32; i++)
  {
    if(i < 20) printf("%d ", array[i]);
  }
  printf("\n");

  free(array);
  free(flags);
  free(store);
}

