
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "cuda.h"
#include "cuda_runtime.h"
#include "curand.h"
#include "device_launch_parameters.h"
#include "thrust/device_ptr.h"
#include "thrust/sort.h"

#include "common.h"
#include "stash.h"
#include "types.h"
#include "xxhash.h"

namespace {

__global__ void insertKernel(StashTable *t, u32 *key, u32 n) {
  u32 id = threadIdx.x + blockIdx.x * blockDim.x;

  if (id < n) {
    u32 k = key[id];

    for (u32 i = 0; i < t->threshold && k != empty; i += 1) {
      u32 d = i % t->dim;
      u32 key = xxhash(t->seed[d], k) % t->len;
      k = atomicExch(&t->val[d * t->len + key], k);
    }

    // Shovel conflicting keys into stash
    if (k != empty) {
      u32 old = atomicAdd(&t->stashSize, 1);
      t->stash[old] = k;
    }
  }
}

__global__ void lookupKernel(StashTable *t, u32 *key, u32 *set, u32 n) {
  u32 id = threadIdx.x + blockIdx.x * blockDim.x;

  if (id < n) {
    u32 k = key[id];

    for (u32 d = 0; d < t->dim; d += 1) {
      u32 offset = xxhash(t->seed[d], k) % t->len;
      if (k == t->val[d * t->len + offset]) {
        set[id] = 1;
        break;
      }
    }

    if (k != empty) {
      for (u32 i = 0; i < t->stashSize; i += 1) {
        if (k == t->stash[i]) {
          set[id] = 1;
          break;
        }
      }
    }
  }
}

} // namespace

StashTable::StashTable(u32 capacity) : DeviceTable(capacity) {
  stashSize = 0;
  stashCapacity = ceil(capacity, 8);
  thread = 256;
  block = ceil(capacity, thread);
  threshold = 1 * bit_width(capacity);
  cudaMallocManaged(&stash, sizeof(u32) * stashCapacity);

  syncCheck();
}

StashTable::~StashTable() { cudaFree(stash); }

void StashTable::insert(u32 *k, u32 n) {
  reset();
  insertKernel<<<block, thread>>>(this, k, n);
  syncCheck();
  collision = stashSize;
  syncCheck();
}

void StashTable::lookup(u32 *k, u32 *s, u32 n) {
  lookupKernel<<<block, thread>>>(this, k, s, n);
  syncCheck();
}
