#include <cstdio>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Common.h"
#include "DeviceTable.h"
#include "Types.h"
#include "xxHash.h"

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

__global__ void lookupKernel(DeviceTable *t, u32 *keys, u32 n) {
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

} // namespace

void randomizeDevice(u32 *array, u32 n) {
  randomizeKernel<<<n / 256 + 1, 256>>>(array, n);
}

void *DeviceTable::operator new(usize size) {
  void *ret = nullptr;
  cudaMallocManaged(&ret, size);
  return ret;
}

void DeviceTable::operator delete(void *p) { cudaFree(p); }

DeviceTable::DeviceTable(u32 capacity, u32 entry) {
  dim = 3;
  len = capacity / dim;
  size = entry;
  threshold = 4 * bit_width(capacity);
  collision = 0;
  thread = 1024;
  block = ceil(entry, thread);

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

void DeviceTable::insert(u32 *v) {
  do {
    reset();
    insertKernel<<<block, thread>>>(this, v, size);
    syncCheck();
  } while (collision > 0);
}

void DeviceTable::lookup(u32 *k) {
  lookupKernel<<<block, thread>>>(this, k, size);
  syncCheck();
}
