#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>

#include "cuckoo.h"
#include "hasher.h"
#include "types.h"

#define EMPTY (i32)(-1)
#define THRESHOLD 16

static bool is_power_of_two(u32 x) { return (x & (x - 1)) == 0; }

table *table_new(u32 len, u32 dim) {
  assert(is_power_of_two(len));
  assert(is_power_of_two(dim));

  i32 *v = malloc(sizeof(i32) * len * dim);
  hasher *h = malloc(sizeof(hasher) * dim);
  table *ret = malloc(sizeof(table));

  // Initialize hasher fields to random value
  for (usize i = 0; i < dim; i += 1) {
    hasher_init(h);
  }

  *ret = (table){v, h, len, dim, 0};

  return ret;
}

u32 table_capacity(table *t) { return t->len * t->dim; }

static void swap(i32 *a, i32 *b) {
  i32 temp = *a;
  *a = *b;
  *b = temp;
}

void table_insert(table *t, i32 v) {
  t->size += 1;

  for (usize i = 0; i < THRESHOLD && v != EMPTY; i += 2) {
    u32 b = i & (t->dim - 1);
    u32 key = hasher_hash(&t->h[b], v) & (t->len - 1);
    swap(&v, &t->v[b * t->len + key]);
  }

  if (v != EMPTY) {
    assert(!"Rehash needed!");
  }
}

void table_free(table *t) {
  free(t->v);
  free(t);
}
