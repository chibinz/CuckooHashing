#pragma once

#include "device.h"

struct StashTable : public DeviceTable {
  /// Stash to store conflicting keys
  u32 *stash;
  u32 stashSize;
  u32 stashCapacity;

  StashTable() = default;
  StashTable(u32 capacity);
  ~StashTable();
  /// Try to insert keys, if not possible, store into stash
  void insert(u32 *k, u32 n);
  /// Lookup keys
  void lookup(u32 *k, u32 *s, u32 n);
};
