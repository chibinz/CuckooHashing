#include <cstdio>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Common.h"
#include "Types.h"
#include "xxHash.h"
#include "MultilevelTable.h"


__global__ void bucketInput(MultilevelTable *t, u32 *array, u32 n) {
    u32 id = threadIdx.x + blockDim.x * blockIdx.x;

    if (id < n) {
        u32 b = array[id] % t->bucket;
        u32 old = atomicAdd(&t->bucketSize[b], 1);
        if (old < t->bucketCapacity) {
            t->bucketData[b * t->bucketCapacity + old] = array[id];
        } else {
            printf("Bucket overflow!\n");
        }
    }
}

__global__ void bucketInsert(MultilevelTable *t) {
  u32 bid = blockIdx.x;
  u32 tid = threadIdx.x;

  if (tid < t->bucketSize[bid]) {
    u32 v = t->bucketData[bid * t->bucketCapacity + tid];

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
