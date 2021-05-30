#include <cstdio>
#include <cstdlib>

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
    u32 b = xxhash(t->bucketSeed, array[id]) % t->bucket;
    u32 old = atomicAdd(&t->bucketSize[b], 1);
    if (old < t->bucketCapacity) {
      t->bucketData[b * t->bucketCapacity + old] = array[id];
    } else {
      printf("Bucket overflow! %u\n", b);
      atomicAdd(&t->collision, 1);
    }
  }
}

__global__ void insertKernel(MultilevelTable *t) {
  // Declare fixed size shared memory
  __shared__ u32 local[3 * 192];
  // Initialize shared memory
  for (u32 i = threadIdx.x; i < t->dim * t->len; i += blockDim.x)
    local[i] = empty;
  __syncthreads();

  u32 bid = blockIdx.x;
  u32 tid = threadIdx.x;

  if (tid < t->bucketSize[bid]) {
    u32 k = t->bucketData[bid * t->bucketCapacity + tid];

    for (u32 i = 0; i < t->threshold && k != empty; i += 1) {
      u32 d = i % t->dim;
      u32 key = xxhash(t->seed[bid * t->dim + d], k) % t->len;
      k = atomicExch(&local[d * t->len + key], k);
      // k = atomicExch(&t->val[bid * t->len * t->dim + d * t->len + key], k);
    }

    // Record number of collisions
    if (k != empty) {
      atomicAdd(&t->collision, 1);
    } else {
      // Copy value from shared memory to global memory
      for (u32 i = threadIdx.x; i < 3 * 192; i += blockDim.x) {
        t->val[bid * t->len * t->dim + i] = local[i];
      }
    }
  }
}

__global__ void lookupKernel(MultilevelTable *t, u32 *keys, u32 *set, u32 n) {
  u32 id = threadIdx.x + blockDim.x * blockIdx.x;

  if (id < n) {
    u32 k = keys[id];
    u32 b = keys[id] % t->bucket;

    for (u32 d = 0; d < t->dim; d += 1) {
      u32 key = xxhash(t->seed[b * t->dim + d], k) % t->len;
      if (k = t->val[b * t->len * t->dim + d * t->len + key]) {
        set[id] = 1;
      }
    }
  }
}

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

void MultilevelTable::insert(u32 *k) {
  do {
    bucketSeed = rand();
    cudaMemset(bucketSize, 0, sizeof(u32) * bucket);
    divideKernel<<<block, thread>>>(this, k, size);
    syncCheck();
  } while (collision > 0);

  do {
    reset();
    insertKernel<<<block, thread>>>(this);
    syncCheck();
  } while (collision > 0);
}

void MultilevelTable::lookup(u32 *k, u32 *s) {
  lookupKernel<<<block, thread>>>(this, k, s, size);
  syncCheck();
}
