#include <cassert>
#include <cstdint>
#include <cstdio>

#include "HostTable.h"
#include "Types.h"
#include "xxHash.h"

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

const i32 HostTable::empty = INT32_MIN;

HostTable::HostTable(u32 len, u32 dim): len(len), dim(dim), size(0) {
  val = new i32[len * dim];
  seed = new u32[dim];
  threshold = bit_width(dim * len);

  randomize(seed, dim);

  for (usize i = 0; i < dim * len; i += 1) {
    val[i] = empty;
  }
}

HostTable::~HostTable() {
  delete[] val;
  delete[] seed;
}

u32 HostTable::capacity() { return len * dim; }

void HostTable::write(FILE *f) {
  for (usize i = 0; i < dim; i += 1) {
    for (usize j = 0; j < len; j += 1) {
      fprintf(f, "%12x", val[i * len + j]);
    }
    fprintf(f, "\n");
  }
}

void HostTable::insert(i32 v) {
  assert(size < capacity());
  assert(v != empty);

  size += 1;

  for (usize i = 0; i < threshold && v != empty; i += 1) {
    u32 b = i % dim;
    u32 key = xxhash(seed[b], v) % len;
    swap(&v, &val[b * len + key]);
  }

  // Mutually recursive call to rehash
  if (v != empty) {
    rehash(v);
  }
}

void HostTable::rehash(i32 v) {
  randomize(seed, dim);

  for (usize i = 0; i < capacity(); i += 1) {
    if (val[i] != empty) {
      insert(val[i]);
    }
  }

  insert(v);
}
