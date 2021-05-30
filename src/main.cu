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

  auto width = atoi(argv[1]);
  auto load = atof(argv[2]);
  auto entry = (u32)((1 << width) * load);
  auto t = new DeviceTable(1 << width, entry);

  u32 *array;
  cudaMalloc(&array, sizeof(u32) * entry);
  randomizeDevice(array, entry);
  syncCheck();

  t->insert(array);
  t->lookup(array);

  printf("Total number of collisions: %u\n", t->collision);
  syncCheck();

  cudaFree(array);
  delete t;

  return 0;
}
