void cpu_radix(int* array, int* store, int length)
{
  for(int i = 0; i < 32; ++i) 
  {
    int num = 1 << i;
    int left = 0;
    int right = 0;
    for(int current = 0; current < length; current += 1)
    {
      if(array[current] & num)
      {
        store[right] = array[current];
        right += 1;
      }
      else
      {
        int t = array[current];
        array[current] = array[left];
        array[left] = t;
        left += 1;
      }
    }
    for(int current = left; current < length; current += 1)
    {
      array[current] = store[current - left];
    }
  }
}
