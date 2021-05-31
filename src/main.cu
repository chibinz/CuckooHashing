#include <stdio.h>
#include <stdlib.h>

#include "cuda.h"

#include "device.h"
#include "host.h"
#include "multi.h"

int main(int argc, char **argv) {
  if (argc != 3) {
    printf("Usage: %s <width> <load-factor>\n", argv[0]);
    return -1;
  }

  auto width = atoi(argv[1]);
  auto load = atof(argv[2]);
  auto entry = (u32)((1 << width) * load);
  // auto t = new DeviceTable(1 << width, entry);
  auto t = new MultilevelTable(1 << width, entry);

  u32 *array, *set;
  cudaMallocManaged(&array, sizeof(u32) * entry);
  cudaMallocManaged(&set, sizeof(u32) * entry);
  cudaMemset(set, 0, sizeof(u32) * entry);
  randomizeDevice(array, entry);
  syncCheck();

  t->insert(array);
  t->lookup(array, set);

  syncCheck();
  syncCheck();

  // printf("Total number of collisions: %u\n", t->collision);
  syncCheck();

  cudaFree(array);
  delete t;

  return 0;
}
