void cpu_split(int* array, int* flags, int* store, int length)
{
  int left = 0;
  int right = 0;
  int temp;
  for(int current = 0; current < length; current += 1)
  {
    if(flags[current])
    {
      store[right] = array[current];
      right += 1;
    }
    else
    {
      temp = array[current];
      array[current] = array[left];
      array[left] = temp;
      left += 1;
    }
  }
  for(int current = left; current < length; current += 1)
  {
    array[current] = store[current - left];
  }
}
