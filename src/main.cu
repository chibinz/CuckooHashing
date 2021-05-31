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
  // auto t = new DeviceTable(1 << width);
  auto t = new MultilevelTable(1 << width);

  u32 *array, *set;
  cudaMalloc(&array, sizeof(u32) * entry);
  cudaMalloc(&set, sizeof(u32) * entry);
  cudaMemset(set, 0, sizeof(u32) * entry);
  randomizeDevice(array, entry);
  syncCheck();

  t->insert(array, entry);
  t->lookup(array, set, entry);

  syncCheck();

  cudaFree(array);
  delete t;

  return 0;
}
