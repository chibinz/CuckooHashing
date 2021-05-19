#pragma once

#include "Types.h"

static u32 bit_width(u32 x) {
  u32 w = 0;

  while (x > 0) {
    w += 1;
    x >>= 1;
  }

  return w;
}
