#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>

#include "cuckoo.h"
#include "types.h"
#include "hasher.h"

static bool is_power_of_two(u32 x) {
  return (x & (x - 1)) == 0;
}

table *table_new(u32 len, u32 dim) {
  assert(is_power_of_two(len));
  assert(is_power_of_two(dim));

  i32 *v = malloc(sizeof(i32) * len * dim);
  hasher *h = malloc(sizeof(hasher) * dim);
  table *ret = malloc(sizeof(table));

  // Initialize hasher fields to random value
  for (usize i = 0; i < dim; i += 1) {
    hasher_init(h, len * dim - 1);
  }

  *ret = (table){v, (hasher *)(h), len, dim, 0};

  return ret;
}

u32 table_capacity(table *t) { return t->len * t->dim; }

void table_free(table *t) {
  free(t->v);
  free(t);
}
