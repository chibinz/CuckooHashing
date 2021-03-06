#pragma once

#include "device.h"
#include "types.h"

struct MultilevelTable : public DeviceTable {
  /// Total number of buckets
  u32 bucket;
  u32 bucketSeed;
  u32 bucketCapacity;
  u32 *bucketSize;
  u32 *bucketData;

  MultilevelTable() = default;
  /// Self adjusting bucket and bucket capacity;
  MultilevelTable(u32 capacity);
  /// Destructor of `DeviceTable` will free `val` and `seed`
  ~MultilevelTable();
  /// Divide input into buckets.
  /// And then run Cuckoo Hashing on each bucket.
  void insert(u32 *k, u32 n);
  /// Lookup keys
  void lookup(u32 *k, u32 *s, u32 n);
};
