#include <stdio.h>
#include <stdlib.h>

#include "cuda.h"

#include "HostTable.h"
#include "Types.h"
#include "xxHash.h"

__global__ void hello() {
  printf("blockIdx.x=%d/%d blocks, threadIdx.x=%d/%d threads\n", blockIdx.x,
         gridDim.x, threadIdx.x, blockDim.x);
}

int main() {
  auto t = HostTable(10, 2);

  for (usize i = 0; i < 10; i += 1) {
    t.insert(rand());
    t.write(stdout);
    putchar('\n');
  }

  hello<<<1, 1024>>>();

  cudaDeviceSynchronize();

  cudaError_t err = cudaGetLastError(); // Get error code

  if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
    exit(-1);
  }

  return 0;
}
