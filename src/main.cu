#include <stdio.h>
#include <stdlib.h>

#include "cuda.h"

#include "DeviceTable.h"
#include "HostTable.h"
#include "MultilevelTable.h"

int main(int argc, char **argv) {
  u32 numEntries = 1 << 24;

  auto t = new DeviceTable(1 << 25, numEntries);

  u32 *array;
  cudaMallocManaged(&array, sizeof(u32) * numEntries);
  randomizeGPU(array, numEntries);

  t->insert(array);
  t->lookup(array);

  printf("Total number of collisions: %u\n", t->collision);
  syncCheck();

  cudaFree(array);
  delete t;

  return 0;
}
