#include <cstdio>

#include "cuda.h"

#include "Common.h"
#include "DeviceTable.h"
#include "Types.h"
#include "xxHash.h"

constexpr u32 empty = (u32)(-1);

__global__ void hello() {
  printf("blockIdx.x=%d/%d blocks, threadIdx.x=%d/%d threads\n", blockIdx.x,
         gridDim.x, threadIdx.x, blockDim.x);
}

__global__ void randomizeArray(u32 *array, u32 n) {
  u32 id = threadIdx.x + blockIdx.x * blockDim.x;

  if (id < n) {
    u32 acc = xxhash(0, n);
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
  if (id == 0) {
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
      atomicAdd(&t->val[0], 1);
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

void syncCheck() {
  cudaDeviceSynchronize();
  auto err = cudaGetLastError(); // Get error code
  if (err != cudaSuccess) {
    printf("Error: %s!\n", cudaGetErrorString(err));
    exit(-1);
  }
}

DeviceTable *tableNew(u32 dim, u32 len) {
  DeviceTable *t;
  cudaMallocManaged(&t, sizeof(DeviceTable));
  cudaMallocManaged(&t->val, sizeof(u32) * dim * len);
  cudaMallocManaged(&t->seed, sizeof(u32) * dim);

  t->dim = dim;
  t->len = len;
  t->threshold = bit_width(4 * dim * len);

  u32 numThreads = 256;
  u32 numBlocks = dim * len / numThreads;

  setEmpty<<<numBlocks, numThreads>>>(t->val, dim * len);
  syncCheck();
  randomizeArray<<<1, 1>>>(t->seed, dim);
  syncCheck();

  return t;
}

void tableFree(DeviceTable *t) {
  cudaFree(t->val);
  cudaFree(t->seed);
}

void wrapper() {
  u32 dim = 2;
  u32 len = 1 << 24;
  u32 numEntries = 1 << 24;
  u32 numThreads = 1024;
  u32 entryBlocks = numEntries / numThreads;

  DeviceTable *t = tableNew(dim, len);

  u32 *array, *numCollisions;
  cudaMallocManaged(&array, sizeof(u32) * numEntries);
  cudaMallocManaged(&numCollisions, sizeof(u32));
  *numCollisions = 0;

  randomizeArray<<<entryBlocks, numThreads>>>(array, numEntries);
  syncCheck();
  batchedInsert<<<entryBlocks, numThreads>>>(t, array, numEntries);
  syncCheck();
  batchedLookup<<<entryBlocks, numThreads>>>(t, array, numEntries);
  syncCheck();
  printArray<<<1, 1>>>(t->val, 1); // Print number of collisions
  syncCheck();

  cudaFree(array);
  tableFree(t);

  syncCheck();
}
