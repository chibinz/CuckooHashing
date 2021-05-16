#pragma once

#include "types.h"
#include "hasher.h"

typedef struct cuckoo_hash_table {
  /// Actual values
  i32 *v;
  /// Array of unique hashers
  hasher *h;
  /// Length of a single subtable
  u32 len;
  /// Number of sub-tables / hash functions
  u32 dim;
  /// Number of occupied entries
  usize size;
} table;
