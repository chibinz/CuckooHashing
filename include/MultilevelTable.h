#pragma once

#include "Types.h"
#include "DeviceTable.h"

struct MultilevelTable : public DeviceTable {
    /// Total number of buckets
    u32 bucket;
    u32 bucketCapacity;
    u32 *bucketSize;
    u32 *bucketData;
    /// Seed used in respective bucket hash function
    u32 *bucketSeed;
};
