#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "Common.h"
#include "HostTable.h"
#include "Types.h"
#include "xxHash.h"

HostTable::HostTable(u32 len, u32 dim) : len(len), dim(dim), size(0) {
  val = new u32[len * dim];
  seed = new u32[dim];
  threshold = 4 * bit_width(dim * len);

  randomizeHost(seed, dim);
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

  randomizeHost(seed, dim);
  memset(val, (i32)(empty), sizeof(u32) * dim * len);

  for (usize i = 0; i < capacity(); i += 1) {
    if (old[i] != empty) {
      insert(old[i]);
    }
  }

  insert(v);
}
