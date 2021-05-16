#pragma once

#include "types.h"

typedef struct {
    u32 r;
    u32 x;
    u32 m;
    u32 mask;
} hasher;

void hasher_init(hasher *h, u32 mask);
