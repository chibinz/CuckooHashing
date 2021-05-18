#pragma once

#include <cstdio>

#include "Types.h"
#include "xxHash.h"

class HostTable {
private:
  /// Actual values
  i32 *val;
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

  static const i32 empty;

  auto rehash(i32 v) -> void;

public:
  HostTable(u32 len, u32 dim);
  ~HostTable();
  auto capacity() -> u32;
  auto find(u32 k) -> i32 *;
  auto insert(i32 v) -> void;
  auto remove(u32 k) -> void;
  auto write(FILE *f) -> void;
};
