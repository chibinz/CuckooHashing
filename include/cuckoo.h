#include "types.h"

typedef u32 (*hash_func)(i32);

typedef struct cuckoo_hash_table {
  /// Actual values
  i32 *val;
  /// Length of a single subtable
  u32 len;
  /// Number of subtable / hash functions
  u32 dim;
} table;
