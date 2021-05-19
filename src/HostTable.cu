#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>

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

static void swap(u32 *a, u32 *b) {
  u32 temp = *a;
  *a = *b;
  *b = temp;
}

static void randomize(u32 *seed, u32 n) {
  // Benign uninitialized read here
  for (usize i = 0; i < n; i += 1) {
    seed[i] = xxhash(i, seed[i]);
  }
}

const u32 HostTable::empty = (u32)(-1);

HostTable::HostTable(u32 len, u32 dim): len(len), dim(dim), size(0) {
  val = new u32[len * dim];
  seed = new u32[dim];
  threshold = bit_width(dim * len);

  randomize(seed, dim);
  memset(val, (i32)(empty), sizeof(u32) * len * dim);
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

void HostTable::insert(u32 v) {
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

void HostTable::rehash(u32 v) {
  u32 *old = val;
  val = new u32[capacity()];

  randomize(seed, dim);
  memset(val, (i32)(empty), sizeof(u32) * dim * len);

  for (usize i = 0; i < capacity(); i += 1) {
    if (old[i] != empty) {
      insert(old[i]);
    }
  }

  insert(v);
}
