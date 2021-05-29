#pragma once

#include <cstdlib>
#include <cstring>

#include "cuda.h"
#include "cuda_runtime.h"

#include "Common.h"
#include "HostTable.h"
#include "Types.h"

void randomizeGPU(u32 *array, u32 n);

/// Convenience struct to pass around as function parameter
struct DeviceTable {
  /// Actual data
  u32 *val;
  /// Seed for each subtable hash function
  u32 *seed;
  /// Number of sub-tables / hash functions
  u32 dim;
  /// Length of a single subtable
  u32 len;
  /// Number of expected input entries
  u32 size;
  /// Number of iterations before rehash happens
  u32 threshold;
  /// Total number of collisions occurred
  u32 collision;
  /// Total number of blocks
  u32 block;
  /// Number of thread per block
  u32 thread;

  DeviceTable() = default;

  DeviceTable(u32 capacity, u32 inputSize) {
    dim = 3;
    len = capacity / dim;
    size = inputSize;
    threshold = 4 * (dim * len);
    collision = 0;
    thread = 1024;
    block = inputSize / thread;

    cudaMallocManaged(&val, sizeof(u32) * dim * len);
    cudaMallocManaged(&seed, sizeof(u32) * dim);
    cudaMemset(val, -1, sizeof(u32) * dim * len);
    randomizeGPU(seed, dim);

    syncCheck();
  }

  ~DeviceTable() {
    cudaFree(val);
    cudaFree(seed);
  }

  void *operator new(usize size) {
    void *ret = nullptr;
    cudaMallocManaged(&ret, size);
    return ret;
  }

  void operator delete(void *p) { cudaFree(p); }

  void reset() {
    cudaMemset(val, -1, sizeof(u32) * dim * len);
    randomizeGPU(seed, dim);
    syncCheck();
  }

  void insert(u32 *v);
  void lookup(u32 *k);
};
