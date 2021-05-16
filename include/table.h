#pragma once

#include <stdio.h>

#include "types.h"
#include "xxhash.h"

typedef struct cuckoo_hash_table {
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
} table;

table *table_new(u32 len, u32 dim);

u32 table_capacity(table *t);

void table_write(table *t, FILE *f);

void table_insert(table *t, i32 v);

void table_rehash(table *t, i32 v);

void table_free(table *t);
