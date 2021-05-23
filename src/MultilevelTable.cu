#include <cstdio>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"

#include "Common.h"
#include "MultilevelTable.h"
#include "Types.h"
#include "xxHash.h"

__global__ void bucketInput(MultilevelTable *t, u32 *array, u32 n) {
  u32 id = threadIdx.x + blockDim.x * blockIdx.x;

  if (id < n) {
    u32 b = array[id] % t->bucket;
    u32 old = atomicAdd(&t->bucketSize[b], 1);
    if (old < t->bucketCapacity) {
      t->bucketData[b * t->bucketCapacity + old] = array[id];
    } else {
      printf("Bucket overflow! %u\n", blockIdx.x);
    }
  }
}

__global__ void bucketInsert(MultilevelTable *t) {
  // Declare fixed size shared memory
  __shared__ u32 local[3 * 128];
  // Initialize shared memory
  for (u32 i = threadIdx.x; i < 3 * 128; i += blockDim.x)
    local[i] = empty;
  __syncthreads();

  u32 bid = blockIdx.x;
  u32 tid = threadIdx.x;

  if (tid < t->bucketSize[bid]) {
    u32 v = t->bucketData[bid * t->bucketCapacity + tid];

    for (u32 i = 0; i < t->threshold && v != empty; i += 1) {
      u32 d = i % t->dim;
      u32 key = xxhash(t->seed[d], v) % t->len;
      v = atomicExch(&local[d * t->len + key], v);
    }

    // Record number of collisions
    if (v != empty) {
      atomicAdd(&t->collision, 1);
    } else {
      // Copy value from shared memory to global memory
      for (u32 i = threadIdx.x; i < 3 * 128; i += blockDim.x) {
        t->val[bid * t->len * t->dim + i] = local[i];
      }
    }
  }
}

__global__ void bucketLookup(MultilevelTable *T) {}

void test() {
  u32 dim = 3;
  u32 len = 128;
  u32 bucket = 1 << 17;
  u32 bucketCapacity = 256;
  u32 numEntries = 1 << 24;
  u32 numThreads = 1024;
  u32 entryBlocks = numEntries / numThreads;

  auto t = new MultilevelTable(dim, len, bucket, bucketCapacity);

  u32 *array;
  cudaMallocManaged(&array, sizeof(u32) * numEntries);
  randomize(array, numEntries);
  syncCheck();

  bucketInput<<<entryBlocks, numThreads>>>(t, array, numEntries);
  syncCheck();

  bucketInsert<<<bucket, bucketCapacity>>>(t);
  syncCheck();

  printf("Total number of collisions: %u\n", t->collision);
  cudaFree(array);
  delete t;

  syncCheck();
}
