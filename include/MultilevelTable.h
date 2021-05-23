#pragma once

#include "DeviceTable.h"
#include "Types.h"

void test();

struct MultilevelTable : public DeviceTable {
  /// Total number of buckets
  u32 bucket;
  u32 bucketCapacity;
  u32 *bucketSize;
  u32 *bucketData;

  MultilevelTable() = default;

  MultilevelTable(u32 dim, u32 len, u32 bucket, u32 bucketCapacity)
      : bucket(bucket), bucketCapacity(bucketCapacity) {
    this->dim = dim;
    this->len = len;
    threshold = 4 * bit_width(dim * len);
    collision = 0;

    cudaMallocManaged(&val, sizeof(u32) * dim * len * bucket);
    cudaMallocManaged(&seed, sizeof(u32) * dim * bucket);
    cudaMallocManaged(&bucketSize, sizeof(u32) * bucket);
    cudaMallocManaged(&bucketData, sizeof(u32) * bucketCapacity * bucket);

    cudaMemset(val, -1, sizeof(u32) * dim * len * bucket);
    randomize(seed, dim * bucket);
  }

  /// Destructor of `DeviceTable` will free `val` and `seed`
  ~MultilevelTable() {
    cudaFree(bucketSize);
    cudaFree(bucketData);
  }
};
