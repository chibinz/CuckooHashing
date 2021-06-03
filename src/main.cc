#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "cuda.h"

#include "bench.h"
#include "device.h"
#include "host.h"
#include "multi.h"

int main() {

  srand(time(NULL));
  // test();
  // insertion();
  // lookup();
  stress();
  // evict();

  return 0;
}
