#pragma once

#include <cstdlib>

#include "cuda.h"
#include "cuda_runtime.h"

#include "types.h"
#include "xxhash.h"

constexpr u32 empty = (u32)(-1);

namespace {

#define syncCheck()                                                            \
  do {                                                                         \
    cudaDeviceSynchronize();                                                   \
    auto err = cudaGetLastError();                                             \
    if (err != cudaSuccess) {                                                  \
      printf("%s:%d %s!\n", __FILE__, __LINE__, cudaGetErrorString(err));      \
      exit(-1);                                                                \
    }                                                                          \
    /* printf("%s:%d Pass!\n", __FILE__, __LINE__); */                         \
  } while (0);

void swap(u32 *a, u32 *b) {
  u32 temp = *a;
  *a = *b;
  *b = temp;
}

u32 ceil(u32 a, u32 b) { return a / b + !!(a % b); }

u32 bit_width(u32 x) {
  u32 w = 0;

  while (x > 0) {
    w += 1;
    x >>= 1;
  }

  return w;
}

void randomizeHost(u32 *a, u32 n) {
  for (usize i = 0; i < n; i += 1) {
    a[i] = rand();
  }
}

} // namespace
