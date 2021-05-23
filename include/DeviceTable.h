#pragma once

#include <cstdlib>
#include <cstring>

#include "cuda.h"
#include "cuda_runtime.h"

#include "Types.h"
#include "Common.h"

void wrapper();

/// Convenience struct to pass around as function parameter
/// No member function implementation defined
struct DeviceTable {
  /// Actual values
  u32 *val;
  /// Array of unique hashers
  u32 *seed;
  /// Length of a single subtable
  u32 len;
  /// Number of sub-tables / hash functions
  u32 dim;
  /// Number of occupied entries
  u32 size;
  /// Number of iterations before rehash happens
  u32 threshold;
  /// Total number of collisions occurred
  u32 collision;

  DeviceTable() = default;

  DeviceTable(u32 dim, u32 len) : len(len), dim(dim) {
    cudaMallocManaged(&val, sizeof(u32) * dim * len);
    cudaMallocManaged(&seed, sizeof(u32) * dim);
    threshold = 4 * bit_width(dim * len);
    collision = 0;

    cudaMemset(val, -1, sizeof(u32) * dim * len);
    randomize(seed, dim);

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

  void operator delete(void *p) {
    cudaFree(p);
  }

  void reset() {
    cudaMemset(val, -1, sizeof(u32) * dim * len);
    randomize(seed, dim);
    syncCheck();
  }
};
