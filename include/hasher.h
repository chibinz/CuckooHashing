#pragma once

#include "types.h"

typedef struct {
  u32 r;
  u32 x;
  u32 m;
} hasher;

void hasher_init(hasher *h);
u32 hasher_hash(hasher *h, i32 v);
