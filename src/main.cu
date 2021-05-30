#include <stdio.h>
#include <stdlib.h>

#include "cuda.h"

#include "DeviceTable.h"
#include "HostTable.h"
#include "MultilevelTable.h"

int main(int argc, char **argv) {
  if (argc != 3) {
    printf("Usage: %s <width> <load-factor>", argv[0]);
    return -1;
  }

  int width = atoi(argv[1]);
  double load = atof(argv[2]);
  auto t = new DeviceTable(1 << 25, 1 << width);

  u32 *array;
  cudaMallocManaged(&array, sizeof(u32) * t->size);
  randomizeDevice(array, t->size);
  syncCheck();

  t->insert(array);
  t->lookup(array);

  printf("Total number of collisions: %u\n", t->collision);
  syncCheck();

  cudaFree(array);
  delete t;

  return 0;
}
