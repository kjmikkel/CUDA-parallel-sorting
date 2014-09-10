void cpu_scan(int* array, int length)
{
  for(int i = 2; i < length; i++)
  {
    array[i] = array[i - 2] + array[i - 1];
  }
  array[1] = array[0];
  array[0] = 0;
}
