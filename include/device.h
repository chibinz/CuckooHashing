#pragma once

#include <cstdlib>
#include <cstring>

#include "cuda.h"
#include "cuda_runtime.h"

#include "common.h"
#include "host.h"
#include "types.h"

/// Convenience struct to pass around as function parameter
struct DeviceTable {
  /// Actual data
  u32 *val;
  /// Seed for each subtable hash function
  u32 *seed;
  /// Length of a single subtable
  u32 len;
  /// Number of sub-tables / hash functions
  u32 dim;
  /// Number of iterations before rehash happens
  u32 threshold;
  /// Total number of collisions occurred
  u32 collision;
  /// Total number of blocks
  u32 block;
  /// Number of thread per block
  u32 thread;

  /// Overriden by CudaMallocManaged to make use of UMA
  void *operator new(usize size);
  /// Overriden by CudaFree
  void operator delete(void *p);

  DeviceTable() = default;
  /// Static hashing with input size known before hand
  DeviceTable(u32 capacity);
  /// Free `val` and `seed`
  ~DeviceTable();
  /// Generate new `seed` and set `val` to `empty`
  void reset();
  /// Batched insert
  void insert(u32 *k, u32 n);
  /// Batched lookup
  void lookup(u32 *k, u32 *s, u32 n);
};
