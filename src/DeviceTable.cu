#include <cstdio>

#include "cuda.h"
#include "curand.h"

#include "DeviceTable.h"
#include "Types.h"
#include "xxHash.h"

constexpr i32 empty = INT32_MIN;

__global__ void hello() {
  printf("blockIdx.x=%d/%d blocks, threadIdx.x=%d/%d threads\n", blockIdx.x,
         gridDim.x, threadIdx.x, blockDim.x);
}

__global__ void genRandArray(i32 *array, u32 n) {
  u32 id = threadIdx.x + blockIdx.x * blockDim.x;

  if (id < n) {
    u32 acc = xxhash((u32)(usize)(array), n);
    acc = xxhash(acc, threadIdx.x);
    acc = xxhash(acc, blockIdx.x);
    acc = xxhash(acc, blockDim.x);
    array[id] = (i32)(acc);
  }
}

__global__ void printArray(i32 *array, u32 n) {
  u32 id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < n) {
    printf("%08x: %x\n", id, array[id]);
  }
}

__global__ void tableInit(i32 *table, u32 capacity) {
  u32 id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id == 0) {
    table[0] = 0;
  } else if (id < capacity) {
    table[id] = empty;
  }
}

__global__ void batchedInsert(i32 *array, u32 n, i32 *table, u32 capacity) {
  u32 id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < n) {
    u32 key = xxhash(0, array[id]) % capacity;
    i32 old = atomicCAS(&table[key], empty, array[id]);
    if (old != empty) {
      atomicAdd(&table[0], 1);
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

void wrapper() {
  u32 capacity = 2048;
  u32 numEntries = 1024;

  u32 numThreads = 64;
  u32 numBlocks = numEntries / numThreads;
  u32 tableBlocks = capacity / numThreads;

  i32 *array, *table;
  cudaMallocManaged(&array, sizeof(u32) * numEntries);
  cudaMallocManaged(&table, sizeof(u32) * capacity);

  genRandArray<<<numBlocks, numThreads>>>(array, numEntries);
  tableInit<<<tableBlocks, numThreads>>>(table, capacity);
  batchedInsert<<<1, numThreads>>>(array, numEntries, table, capacity);
  printArray<<<1, numThreads>>>(table, 256);

  cudaFree(array);
  cudaFree(table);

  syncCheck();
}
