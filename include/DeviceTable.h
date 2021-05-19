#pragma once

#include "Types.h"

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
};
