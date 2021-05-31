#include <stdlib.h>

#include "types.h"
#include "xxhash.h"

static const u32 PRIME1 = 0x9E3779B1U;
static const u32 PRIME2 = 0x85EBCA77U;
static const u32 PRIME3 = 0xC2B2AE3DU;
static const u32 PRIME4 = 0x27D4EB2FU;
static const u32 PRIME5 = 0x165667B1U;

__host__ __device__ static u32 rotate_left(u32 v, u32 n) {
  return (v << n) | (v >> (32 - n));
}

__host__ __device__ u32 xxhash(u32 seed, u32 v) {
  u32 acc = seed + PRIME5;

  acc = acc + v * PRIME3;
  acc = rotate_left(acc, 17) * PRIME4;

  u8 *byte = (u8 *)(&v);
  for (u32 i = 0; i < 4; i += 1) {
    acc = acc + byte[i] * PRIME5;
    acc = rotate_left(acc, 11) * PRIME1;
  }

  acc ^= acc >> 15;
  acc *= PRIME2;
  acc ^= acc >> 13;
  acc *= PRIME3;
  acc ^= acc >> 16;

  return acc;
}
