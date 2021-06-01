#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "cuda.h"
#include "cuda_runtime.h"
#include "curand.h"
#include "device_launch_parameters.h"

#include "common.h"
#include "device.h"
#include "types.h"
#include "xxhash.h"

namespace {

__global__ void randomizeKernel(u32 *a, u32 n) {
  u32 id = threadIdx.x + blockIdx.x * blockDim.x;

  if (id < n) {
    u32 acc = xxhash(a[id], n);
    acc = xxhash(acc, threadIdx.x);
    acc = xxhash(acc, blockIdx.x);
    acc = xxhash(acc, blockDim.x);
    a[id] = (u32)(acc);
  }
}

__global__ void insertKernel(DeviceTable *t, u32 *key, u32 n) {
  u32 id = threadIdx.x + blockIdx.x * blockDim.x;

  if (id < n) {
    u32 k = key[id];

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

__global__ void lookupKernel(DeviceTable *t, u32 *key, u32 *set, u32 n) {
  u32 id = threadIdx.x + blockIdx.x * blockDim.x;

  if (id < n) {
    u32 k = key[id];

    for (u32 d = 0; d < t->dim; d += 1) {
      u32 offset = xxhash(t->seed[d], k) % t->len;
      if (k == t->val[d * t->len + offset]) {
        set[id] = 1;
      }
    }
  }
}

} // namespace

void randomizeDevice(u32 *a, u32 n) {
  auto tmp = new u32[n];
  randomizeHost(tmp, n);
  cudaMemcpy(a, tmp, sizeof(u32) *n, cudaMemcpyHostToDevice);
  delete[] tmp;
}

void *DeviceTable::operator new(usize size) {
  void *ret = nullptr;
  cudaMallocManaged(&ret, size);
  return ret;
}

void DeviceTable::operator delete(void *p) { cudaFree(p); }

DeviceTable::DeviceTable(u32 capacity) {
  dim = 3;
  len = ceil(ceil(capacity, dim), 32) * 32;
  collision = 0;
  thread = 1024;
  block = ceil(capacity, thread);
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
  // printf("Reset!\n");
  collision = 0;
  cudaMemset(val, -1, sizeof(u32) * dim * len);
  randomizeDevice(seed, dim);
  syncCheck();
}

void DeviceTable::insert(u32 *k, u32 n) {
  do {
    reset();
    insertKernel<<<block, thread>>>(this, k, n);
    syncCheck();
  } while (collision > 0);
}

void DeviceTable::lookup(u32 *k, u32 *s, u32 n) {
  lookupKernel<<<block, thread>>>(this, k, s, n);
  syncCheck();
}
