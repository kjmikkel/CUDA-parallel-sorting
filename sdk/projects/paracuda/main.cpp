extern void runTest(int argc, char** argv);
extern void runQuicksort(int argc, char** argv);
extern void runTestRadix(int argc, char** argv);

#include <cstdio>
int main(int argc, char** argv) 
{
  //runQuicksort(argc, argv);
  runTest(argc, argv);
  //runTestRadix(argc, argv);
  return 0;
}

