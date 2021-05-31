#include <cstdio>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "common.h"
#include "device.h"
#include "types.h"
#include "xxhash.h"

namespace {

__global__ void randomizeKernel(u32 *array, u32 n) {
  u32 id = threadIdx.x + blockIdx.x * blockDim.x;

  if (id < n) {
    u32 acc = xxhash(array[id], n);
    acc = xxhash(acc, threadIdx.x);
    acc = xxhash(acc, blockIdx.x);
    acc = xxhash(acc, blockDim.x);
    array[id] = (u32)(acc);
  }
}

__global__ void insertKernel(DeviceTable *t, u32 *array, u32 n) {
  u32 id = threadIdx.x + blockIdx.x * blockDim.x;

  if (id < n) {
    u32 k = array[id];

    for (u32 i = 0; i < t->threshold && k != empty; i += 1) {
      u32 d = i % t->dim;
      u32 key = xxhash(t->seed[d], k) % t->len;
      k = atomicExch(&t->val[d * t->len + key], k);
    }

    // Record number of collisions
    if (k != empty) {
      atomicAdd(&t->collision, 1);
    }
  }
}

__global__ void lookupKernel(DeviceTable *t, u32 *keys, u32 *set, u32 n) {
  u32 id = threadIdx.x + blockIdx.x * blockDim.x;

  if (id < n) {
    u32 k = keys[id];

    for (u32 d = 0; d < t->dim; d += 1) {
      u32 key = xxhash(t->seed[d], k) % t->len;
      if (k == t->val[d * t->len + key]) {
        set[id] = 1;
      }
    }
  }
}

} // namespace

void randomizeDevice(u32 *array, u32 n) {
  randomizeKernel<<<ceil(n, 256), 256>>>(array, n);
}

void *DeviceTable::operator new(usize size) {
  void *ret = nullptr;
  cudaMallocManaged(&ret, size);
  return ret;
}

void DeviceTable::operator delete(void *p) { cudaFree(p); }

DeviceTable::DeviceTable(u32 capacity, u32 entry) {
  dim = 3;
  len = ceil(ceil(capacity, dim), 32) * 32;
  size = entry;
  collision = 0;
  thread = 1024;
  block = ceil(entry, thread);
  threshold = 4 * bit_width(capacity);

  cudaMalloc(&val, sizeof(u32) * dim * len);
  cudaMalloc(&seed, sizeof(u32) * dim);
  syncCheck();
}

DeviceTable::~DeviceTable() {
  cudaFree(val);
  cudaFree(seed);
}

void DeviceTable::reset() {
  printf("Reset!\n");
  collision = 0;
  cudaMemset(val, -1, sizeof(u32) * dim * len);
  randomizeDevice(seed, dim);
  syncCheck();
}

void DeviceTable::insert(u32 *k) {
  do {
    reset();
    insertKernel<<<block, thread>>>(this, k, size);
    syncCheck();
  } while (collision > 0);
}

void DeviceTable::lookup(u32 *k, u32 *s) {
  lookupKernel<<<block, thread>>>(this, k, s, size);
  syncCheck();
}
