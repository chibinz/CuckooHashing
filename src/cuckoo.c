#include <stdlib.h>

#include "cuckoo.h"
#include "types.h"

table *table_new(u32 len, u32 dim) {
  i32 *val = malloc(sizeof(i32) * len * dim);
  table *ret = malloc(sizeof(table));

  *ret = (table){val, len, dim};

  return ret;
}

void table_free(table *t) {
  free(t->val);
  free(t);
}
