#include <cstdio>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Common.h"
#include "DeviceTable.h"
#include "Types.h"
#include "xxHash.h"

__global__ void randomizeArray(u32 *array, u32 n) {
  u32 id = threadIdx.x + blockIdx.x * blockDim.x;

  if (id < n) {
    u32 acc = xxhash(array[id], n);
    acc = xxhash(acc, threadIdx.x);
    acc = xxhash(acc, blockIdx.x);
    acc = xxhash(acc, blockDim.x);
    array[id] = (u32)(acc);
  }
}

__global__ void printArray(u32 *array, u32 n) {
  u32 id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < n) {
    printf("%08x: %x\n", id, array[id]);
  }
}

__global__ void setEmpty(u32 *val, u32 capacity) {
  u32 id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < capacity) {
    val[id] = empty;
  }
}

__global__ void batchedInsert(DeviceTable *t, u32 *array, u32 n) {
  u32 id = threadIdx.x + blockIdx.x * blockDim.x;

  if (id < n) {
    u32 v = array[id];

    for (u32 i = 0; i < t->threshold && v != empty; i += 1) {
      u32 b = i % t->dim;
      u32 key = xxhash(t->seed[b], v) % t->len;
      v = atomicExch(&t->val[b * t->len + key], v);
    }

    // Record number of collisions
    if (v != empty) {
      atomicAdd(&t->collision, 1);
    }
  }
}

__global__ void batchedLookup(DeviceTable *t, u32 *keys, u32 n) {
  u32 id = threadIdx.x + blockIdx.x * blockDim.x;

  if (id < n) {
    u32 v = keys[id];

    for (u32 i = 0; i < t->dim; i += 1) {
      u32 key = xxhash(t->seed[i], v) % t->len;
      if (t->val[i * t->len + key] == v) {
        break;
      }
    }
  }
}

void tableInit(DeviceTable *t, u32 dim, u32 len) {
  t->dim = dim;
  t->len = len;
  t->threshold = 4 * bit_width(dim * len);
  t->collision = 0;

  cudaMemset(t->val, -1, sizeof(u32) * dim * len);
  randomize(t->seed, dim);
  syncCheck();
}

void wrapper() {
  u32 dim = 3;
  u32 len = 1 << 23;
  u32 numEntries = 1 << 24;
  u32 numThreads = 1024;
  u32 entryBlocks = numEntries / numThreads;

  auto t = new DeviceTable(dim, len);

  u32 *array;
  cudaMallocManaged(&array, sizeof(u32) * numEntries);
  randomize(array, numEntries);

  batchedInsert<<<entryBlocks, numThreads>>>(t, array, numEntries);
  syncCheck();
  while (t->collision > 0) {
    tableInit(t, dim, len);
    batchedInsert<<<entryBlocks, numThreads>>>(t, array, numEntries);
    syncCheck();
  }
  batchedLookup<<<entryBlocks, numThreads>>>(t, array, numEntries);
  syncCheck();

  printf("Total number of collisions: %u\n", t->collision);
  syncCheck();

  cudaFree(array);
  delete t;

  syncCheck();
}
