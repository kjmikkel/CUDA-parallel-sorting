The library only really consists of paracuda.h, the rest is just tests.
#include "paracuda.h"
should do the trick (that, and a lot of custom functions, as required
for each custom algorithm)
Remember to put the operators in a .cu file, or they won't be able to
run on the GPU.
