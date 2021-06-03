#pragma once

#include <cstdio>

#include "types.h"
#include "xxhash.h"

struct HostTable {
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

  HostTable() = default;
  HostTable(u32 len, u32 dim);
  ~HostTable();
  u32 capacity();
  void cleanup();
  void insert(u32 v);
  void rehash(u32 v);
  void write(FILE *f);
};
