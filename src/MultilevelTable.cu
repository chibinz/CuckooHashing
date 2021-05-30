#include <cstdio>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"

#include "Common.h"
#include "MultilevelTable.h"
#include "Types.h"
#include "xxHash.h"

namespace {

__global__ void divideKernel(MultilevelTable *t, u32 *array, u32 n) {
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

__global__ void insertKernel(MultilevelTable *t) {
  // Declare fixed size shared memory
  __shared__ u32 local[3 * 192];
  // Initialize shared memory
  for (u32 i = threadIdx.x; i < 3 * 192; i += blockDim.x)
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
      // v = atomicExch(&t->val[bid * t->len * t->dim + d * t->len + key], v);
    }

    // Record number of collisions
    if (v != empty) {
      atomicAdd(&t->collision, 1);
    } else {
      // Copy value from shared memory to global memory
      for (u32 i = threadIdx.x; i < 3 * 192; i += blockDim.x) {
        t->val[bid * t->len * t->dim + i] = local[i];
      }
    }
  }
}

__global__ void lookupKernel(MultilevelTable *T, u32 *k) {}

} // namespace

MultilevelTable::MultilevelTable(u32 capacity, u32 entry) {
  dim = 3;
  len = 192;
  size = entry;
  collision = 0;
  bucketCapacity = 512;
  bucket = ceil(capacity, bucketCapacity);
  thread = bucketCapacity;
  block = ceil(entry, thread);
  threshold = 4 * bit_width(dim * len);

  cudaMalloc(&val, sizeof(u32) * dim * len * bucket);
  cudaMalloc(&seed, sizeof(u32) * dim * bucket);
  cudaMalloc(&bucketSize, sizeof(u32) * bucket);
  cudaMalloc(&bucketData, sizeof(u32) * bucketCapacity * bucket);

  cudaMemset(val, -1, sizeof(u32) * dim * len * bucket);
  cudaMemset(bucketSize, 0, sizeof(u32) * bucket);
  randomizeDevice(seed, dim * bucket);
}

MultilevelTable::~MultilevelTable() {
  cudaFree(bucketSize);
  cudaFree(bucketData);
}

void MultilevelTable::insert(u32 *v) {
  divideKernel<<<block, thread>>>(this, v, size);
  syncCheck();

  do {
    reset();
    insertKernel<<<block, thread>>>(this);
    syncCheck();
  } while (collision > 0);
}

void MultilevelTable::lookup(u32 *k) {
  lookupKernel<<<block, thread>>>(this, k);
  syncCheck();
}
