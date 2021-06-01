#include <stdio.h>
#include <stdlib.h>

#include "cuda.h"

#include "bench.h"
#include "device.h"
#include "host.h"
#include "multi.h"

int main() {

  test();
  // insertion(); lookup();
  stress();

  return 0;
}
