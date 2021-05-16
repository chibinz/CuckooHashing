#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "table.h"
#include "types.h"
#include "xxhash.h"

#define EMPTY INT32_MIN

static u32 bit_width(u32 x) {
  u32 w = 0;

  while (x > 0) {
    w += 1;
    x >>= 1;
  }

  return w;
}

static void swap(i32 *a, i32 *b) {
  i32 temp = *a;
  *a = *b;
  *b = temp;
}

static void randomize(u32 *seed, u32 n) {
  // Benign uninitialized read here
  for (usize i = 0; i < n; i += 1) {
    seed[i] = xxhash(i, seed[i]);
  }
}

table *table_new(u32 len, u32 dim) {
  i32 *val = malloc(sizeof(i32) * len * dim);
  u32 *seed = malloc(sizeof(u32) * dim);
  table *ret = malloc(sizeof(table));

  randomize(seed, dim);

  for (usize i = 0; i < dim * len; i += 1) {
    val[i] = EMPTY;
  }

  *ret = (table){val, seed, len, dim, 0, bit_width(dim * len)};

  return ret;
}

u32 table_capacity(table *t) { return t->len * t->dim; }

void table_write(table *t, FILE *f) {
  for (usize i = 0; i < t->dim; i += 1) {
    for (usize j = 0; j < t->len; j += 1) {
      fprintf(f, "%12x", t->val[i * t->len + j]);
    }
    fprintf(f, "\n");
  }
}

void table_insert(table *t, i32 v) {
  assert(t->size < table_capacity(t));
  assert(v != EMPTY);

  t->size += 1;

  for (usize i = 0; i < t->threshold && v != EMPTY; i += 1) {
    u32 b = i % t->dim;
    u32 key = xxhash(t->seed[b], v) % t->len;
    swap(&v, &t->val[b * t->len + key]);
  }

  // Mutually recursive call to rehash
  if (v != EMPTY) {
    table_rehash(t, v);
  }
}

void table_rehash(table *t, i32 v) {
  randomize(t->seed, t->dim);

  for (usize i = 0; i < table_capacity(t); i += 1) {
    if (t->val[i] != EMPTY) {
      table_insert(t, t->val[i]);
    }
  }

  table_insert(t, v);
}

void table_free(table *t) {
  free(t->val);
  free(t->seed);
  free(t);
}
