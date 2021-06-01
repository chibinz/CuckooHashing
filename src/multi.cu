#include <cstdio>
#include <cstdlib>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"

#include "common.h"
#include "multi.h"
#include "types.h"
#include "xxhash.h"

namespace {

__global__ void divideKernel(MultilevelTable *t, u32 *key, u32 n) {
  u32 id = threadIdx.x + blockDim.x * blockIdx.x;

  if (id < n) {
    u32 b = xxhash(t->bucketSeed, key[id]) % t->bucket;
    u32 old = atomicAdd(&t->bucketSize[b], 1);
    if (old < t->bucketCapacity) {
      t->bucketData[b * t->bucketCapacity + old] = key[id];
    } else {
      // printf("Bucket overflow! %u\n", b);
      atomicAdd(&t->collision, 1);
    }
  }
}

__global__ void insertKernel(MultilevelTable *t) {
  // Declare shared memory size of `t->dim * t->len`
  extern __shared__ u32 local[];

  // Initialize shared memory
  for (u32 i = threadIdx.x; i < t->dim * t->len; i += blockDim.x)
    local[i] = empty;
  __syncthreads();

  u32 bid = blockIdx.x;
  u32 tid = threadIdx.x;

  if (tid < t->bucketSize[bid]) {
    u32 k = t->bucketData[bid * t->bucketCapacity + tid];

    do {
      // Record collision in shared memory
      local[t->dim * t->len] = 0;

      for (u32 i = 0; i < t->threshold && k != empty; i += 1) {
        u32 d = i % t->dim;
        u32 key = xxhash(t->seed[bid * t->dim + d], k) % t->len;
        // k = atomicExch(&local[d * t->len + key], k);
        k = atomicExch(&t->val[bid * t->len * t->dim + d * t->len + key], k);
        // u32 old = local[d * t->len + key];
        // __syncthreads();
        // local[d * t->len + key] = k;
        // __syncthreads();
        // if (k == local[d * t->len + key]) {
        //   k = old;
        // }
        // __syncthreads();
      }

      // Guard to avoid bank conflict
      if (local[t->dim * t->len] == 0 && k != empty) {
        local[t->dim * t->len] = 1;
        for (u32 d = 0; d < t->dim; d += 1) {
          t->seed[bid * t->dim + d] = xxhash(tid, t->seed[bid * t->dim + d]);
        }
      }
      __syncthreads();

    } while (local[t->dim * t->len] != 0);
  }

  __syncthreads();
  // Copy value from shared memory to global memory
  for (u32 i = threadIdx.x; i < t->dim * t->len; i += blockDim.x) {
    // t->val[bid * t->len * t->dim + i] = local[i];
  }
}

__global__ void lookupKernel(MultilevelTable *t, u32 *key, u32 *set, u32 n) {
  u32 id = threadIdx.x + blockDim.x * blockIdx.x;

  if (id < n) {
    u32 k = key[id];
    u32 b = xxhash(t->bucketSeed, key[id]) % t->bucket;

    for (u32 d = 0; d < t->dim; d += 1) {
      u32 offset = xxhash(t->seed[b * t->dim + d], k) % t->len;
      if (k == t->val[b * t->len * t->dim + d * t->len + offset]) {
        set[id] = 1;
      }
    }
  }
}

} // namespace

MultilevelTable::MultilevelTable(u32 capacity) {
  dim = 3;
  len = 192;
  collision = 0;
  bucketCapacity = 512;
  bucket = ceil(capacity, bucketCapacity);
  thread = bucketCapacity;
  block = bucket;
  threshold = 4 * bit_width(dim * len);

  cudaMalloc(&val, sizeof(u32) * dim * len * bucket);
  cudaMalloc(&seed, sizeof(u32) * dim * bucket);
  cudaMalloc(&bucketSize, sizeof(u32) * bucket);
  cudaMalloc(&bucketData, sizeof(u32) * bucketCapacity * bucket);

  cudaMemset(val, -1, sizeof(u32) * dim * len * bucket);
  cudaMemset(bucketSize, 0, sizeof(u32) * bucket);
  randomizeDevice(seed, dim * bucket);
  syncCheck();
}

MultilevelTable::~MultilevelTable() {
  cudaFree(bucketSize);
  cudaFree(bucketData);
}

void MultilevelTable::insert(u32 *k, u32 n) {
  do {
    collision = 0;
    bucketSeed = rand();
    cudaMemset(bucketSize, 0, sizeof(u32) * bucket);
    divideKernel<<<block, thread>>>(this, k, n);
    syncCheck();
  } while (collision > 0);

  insertKernel<<<block, thread, sizeof(u32) * (dim * len + 1)>>>(this);
  syncCheck();
}

void MultilevelTable::lookup(u32 *k, u32 *s, u32 n) {
  lookupKernel<<<block, thread>>>(this, k, s, n);
  syncCheck();
}
